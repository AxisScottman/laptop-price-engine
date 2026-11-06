"""Row-wise parsing of the raw laptop spec strings into clean numeric/categorical features.

Every step here looks at ONE row at a time (no statistics are learned from the data),
so applying it before the train/test split cannot leak information.
All learned steps (imputing, scaling, encoding) live inside the scikit-learn Pipeline.
"""
import re
import numpy as np
import pandas as pd

NUM_COLS = ["Inches", "Ram_GB", "Weight_kg", "Cpu_GHz", "SSD_GB", "HDD_GB",
            "Hybrid_GB", "Flash_GB", "PPI", "Touchscreen", "IPS", "Dedicated_GPU"]
CAT_COLS = ["Company", "TypeName", "Cpu_brand", "Gpu_brand", "OS_group"]
FEATURES = CAT_COLS + NUM_COLS


def _cpu_brand(cpu: str) -> str:
    for tier in ("i3", "i5", "i7"):
        if f"Core {tier}" in cpu:
            return f"Intel Core {tier}"
    if cpu.startswith("Intel"):
        return "Other Intel"
    if cpu.startswith("AMD"):
        return "AMD"
    return "Other"


def _storage(memory: str) -> dict:
    out = {"SSD_GB": 0.0, "HDD_GB": 0.0, "Hybrid_GB": 0.0, "Flash_GB": 0.0}
    for part in str(memory).split("+"):
        m = re.search(r"([\d.]+)\s*(TB|GB)\s*(.*)", part.strip())
        if not m:
            continue
        size = float(m.group(1)) * (1000 if m.group(2) == "TB" else 1)
        kind = m.group(3).lower()
        if "ssd" in kind:
            out["SSD_GB"] += size
        elif "hdd" in kind:
            out["HDD_GB"] += size
        elif "hybrid" in kind:
            out["Hybrid_GB"] += size
        elif "flash" in kind:
            out["Flash_GB"] += size
    return out


def _os_group(os_name: str) -> str:
    o = str(os_name).lower()
    if "windows" in o:
        return "Windows"
    if "mac" in o:
        return "macOS"
    if "linux" in o:
        return "Linux"
    return "Other / No OS"


def parse_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw Kaggle-style columns into the modelling feature table."""
    out = pd.DataFrame(index=df.index)
    out["Company"] = df["Company"].astype(str)
    out["TypeName"] = df["TypeName"].astype(str)
    out["Inches"] = pd.to_numeric(df["Inches"], errors="coerce")
    out["Ram_GB"] = df["Ram"].astype(str).str.replace("GB", "", regex=False).astype(float)
    out["Weight_kg"] = df["Weight"].astype(str).str.replace("kg", "", regex=False).astype(float)

    res = df["ScreenResolution"].astype(str)
    out["Touchscreen"] = res.str.contains("Touchscreen").astype(int)
    out["IPS"] = res.str.contains("IPS").astype(int)
    xy = res.str.extract(r"(\d{3,4})x(\d{3,4})").astype(float)
    out["PPI"] = np.sqrt(xy[0] ** 2 + xy[1] ** 2) / out["Inches"]

    cpu = df["Cpu"].astype(str)
    out["Cpu_brand"] = cpu.map(_cpu_brand)
    out["Cpu_GHz"] = cpu.str.extract(r"([\d.]+)GHz")[0].astype(float)

    out = out.join(pd.DataFrame([_storage(m) for m in df["Memory"]], index=df.index))

    gpu = df["Gpu"].astype(str)
    out["Gpu_brand"] = gpu.str.split().str[0]
    out["Dedicated_GPU"] = (~gpu.str.startswith("Intel")).astype(int)
    out["OS_group"] = df["OpSys"].map(_os_group)
    return out[FEATURES]


def rule_based_score(X: pd.DataFrame) -> pd.Series:
    """A hand-written 'hardware ranking' (like a rule-based PC score) used as a comparison."""
    cpu_pts = X["Cpu_brand"].map({"Intel Core i7": 3, "Intel Core i5": 2, "AMD": 1.5,
                                  "Intel Core i3": 1}).fillna(0.5)
    return (X["Ram_GB"] / 32 * 30 + (X["SSD_GB"] / 1000).clip(upper=1) * 25
            + cpu_pts / 3 * 25 + X["Dedicated_GPU"] * 15 + X["Cpu_GHz"] / 3.6 * 5)
