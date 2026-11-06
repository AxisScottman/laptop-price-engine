import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import FEATURES, parse_raw

BASE = Path(__file__).parent
RAW_COLS = ["Company", "TypeName", "Inches", "ScreenResolution", "Cpu", "Ram", "Memory", "Gpu", "OpSys", "Weight"]

st.set_page_config(page_title="Laptop Fair-Price & Value Engine", page_icon="💻", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE / "models" / "price_pipeline.joblib")
    tiers = joblib.load(BASE / "models" / "tier_model.joblib")
    meta = json.loads((BASE / "models" / "meta.json").read_text())
    return model, tiers, meta


model, tier_art, meta = load_artifacts()
ch = meta["choices"]


def assign_tier(X: pd.DataFrame) -> pd.Series:
    cols = tier_art["cols"]
    Z = tier_art["scaler"].transform(X[cols].fillna(pd.Series(tier_art["fill"])))
    ranks = [tier_art["rank_of"][int(c)] for c in tier_art["kmeans"].predict(Z)]
    return pd.Series([tier_art["names"][r] for r in ranks], index=X.index)


def verdict(quoted: float, fair: float) -> str:
    ratio = quoted / fair
    if ratio < meta["deal_low_ratio"]:
        return "🟢 Good deal"
    if ratio > meta["deal_high_ratio"]:
        return "🔴 Overpriced"
    return "🟡 Fair price"


st.title("💻 Laptop Fair-Price & Value Engine")
st.caption(f"Model: {meta['best_model']} · typical error ≈ {meta['median_pct_err']}% · "
           f"test MAE ≈ {meta['test_mae']:,.0f} price units · R² = {meta['test_r2']:.2f}")

tab1, tab2, tab3 = st.tabs(["🔎 Check one machine", "🗂️ Check a whole fleet (CSV)", "ℹ️ About"])

# ---------------------------------------------------------------- single machine
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        company = st.selectbox("Brand", ch["Company"], index=ch["Company"].index("Dell"))
        type_name = st.selectbox("Type", ch["TypeName"], index=ch["TypeName"].index("Notebook"))
        inches = st.selectbox("Screen size (inches)", meta["inches"], index=meta["inches"].index(15.6))
        res = st.selectbox("Screen resolution", meta["resolutions"], index=meta["resolutions"].index("1920x1080"))
        weight = st.number_input("Weight (kg)", 0.5, 5.0, 2.0, 0.1)
    with c2:
        cpu_brand = st.selectbox("CPU family", ch["Cpu_brand"], index=ch["Cpu_brand"].index("Intel Core i5"))
        ghz = st.number_input("CPU clock (GHz)", 0.9, 4.5, 2.5, 0.1)
        ram = st.selectbox("RAM (GB)", meta["ram"], index=meta["ram"].index(8.0))
        gpu_brand = st.selectbox("GPU brand", ch["Gpu_brand"], index=ch["Gpu_brand"].index("Intel"))
        os_group = st.selectbox("Operating system", ch["OS_group"], index=ch["OS_group"].index("Windows"))
    with c3:
        ssd = st.number_input("SSD (GB)", 0, 2000, 256, 128)
        hdd = st.number_input("HDD (GB)", 0, 4000, 0, 500)
        flash = st.number_input("Flash storage (GB)", 0, 1000, 0, 32)
        hybrid = st.number_input("Hybrid drive (GB)", 0, 4000, 0, 500)
        touch = st.checkbox("Touchscreen")
        ips = st.checkbox("IPS panel")
        quoted = st.number_input("Quoted price (optional, 0 = skip)", 0.0, 1_000_000.0, 0.0, 1000.0)

    if st.button("Estimate fair price", type="primary"):
        if ssd + hdd + flash + hybrid == 0:
            st.error("Please enter at least one storage drive (SSD, HDD, flash or hybrid).")
        else:
            x, y = (int(v) for v in res.split("x"))
            row = pd.DataFrame([{
                "Company": company, "TypeName": type_name, "Cpu_brand": cpu_brand, "Gpu_brand": gpu_brand,
                "OS_group": os_group, "Inches": inches, "Ram_GB": ram, "Weight_kg": weight, "Cpu_GHz": ghz,
                "SSD_GB": ssd, "HDD_GB": hdd, "Hybrid_GB": hybrid, "Flash_GB": flash,
                "PPI": np.hypot(x, y) / inches, "Touchscreen": int(touch), "IPS": int(ips),
                "Dedicated_GPU": int(gpu_brand != "Intel"),
            }])[FEATURES]
            fair = float(model.predict(row)[0])
            tier = assign_tier(row).iloc[0]
            lo, hi = fair * meta["deal_low_ratio"], fair * meta["deal_high_ratio"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Fair price estimate", f"{fair:,.0f}")
            m2.metric("Typical range", f"{lo:,.0f} – {hi:,.0f}")
            m3.metric("Hardware tier", tier)
            if quoted > 0:
                st.subheader(f"{verdict(quoted, fair)}  ·  quoted {quoted:,.0f} is {quoted / fair * 100:.0f}% of the fair price")
            if not (meta["price_min"] <= fair <= meta["price_max"]):
                st.warning("This configuration is outside the price range the model was trained on, so treat the estimate with caution.")
            st.caption("Estimates reflect new retail laptop prices in the training snapshot; used machines and condition are not modelled.")

# ---------------------------------------------------------------- fleet CSV
with tab2:
    st.write("Upload a CSV of machines. Accepted formats: **(A)** the raw spec columns "
             f"`{', '.join(RAW_COLS)}` or **(B)** the cleaned feature columns. An optional `Price` column adds a deal verdict.")
    template = pd.DataFrame([
        ["Dell", "Notebook", 15.6, "Full HD 1920x1080", "Intel Core i5 7200U 2.5GHz", "8GB", "256GB SSD",
         "Intel HD Graphics 620", "Windows 10", "2.0kg", 45000]], columns=RAW_COLS + ["Price"])
    st.download_button("Download CSV template", template.to_csv(index=False), "fleet_template.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            if df.empty:
                st.error("The file has no rows.")
            else:
                if set(RAW_COLS).issubset(df.columns):
                    X = parse_raw(df)
                elif set(FEATURES).issubset(df.columns):
                    X = df[FEATURES].copy()
                else:
                    missing = sorted(set(RAW_COLS) - set(df.columns))
                    raise ValueError(f"Missing columns for the raw format: {missing}")
                out = df.copy()
                out["Fair_price"] = model.predict(X).round(0)
                out["Tier"] = assign_tier(X).values
                if "Price" in df.columns:
                    out["Verdict"] = [verdict(q, f) for q, f in zip(pd.to_numeric(df["Price"], errors="coerce"), out["Fair_price"])]
                st.dataframe(out, use_container_width=True)
                st.bar_chart(out["Tier"].value_counts())
                st.download_button("Download results", out.to_csv(index=False), "fleet_results.csv", "text/csv")
        except Exception as e:  # wrong input must never crash the app
            st.error(f"Could not process this file: {e}")

# ---------------------------------------------------------------- about
with tab3:
    st.markdown(f"""
**What it does.** Estimates the fair price of a laptop from its specs, labels a quoted price as a good deal / fair / overpriced,
and places the machine in one of four hardware tiers (K-Means on specs).

**How good is it?** On 255 held-out laptops the model's median error is about **{meta['median_pct_err']}%**;
errors are larger for the most expensive machines.

**"Good deal" rule.** Compared with the model's fair price, the cheapest quarter of real laptops are priced below
{meta['deal_low_ratio']:.2f}× and the most expensive quarter above {meta['deal_high_ratio']:.2f}×.

**Limits.** ~1,270 unique laptops from one snapshot in time, new retail prices only, no age or condition.
This is a decision aid, not a valuation.
""")
