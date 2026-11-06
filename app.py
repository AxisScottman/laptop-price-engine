import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import FEATURES, parse_raw

BASE = Path(__file__).parent
RAW_COLS = ["Company", "TypeName", "Inches", "ScreenResolution", "Cpu", "Ram", "Memory", "Gpu", "OpSys", "Weight"]
DEFAULT_RATE = 0.58   # EGP per 1 INR (early Nov 2026 mid-market is roughly 0.55-0.59)

st.set_page_config(page_title="Laptop Fair-Price Engine", page_icon="💻", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE / "models" / "price_pipeline.joblib")
    tiers = joblib.load(BASE / "models" / "tier_model.joblib")
    meta = json.loads((BASE / "models" / "meta.json").read_text())
    res_path = BASE / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    return model, tiers, meta, results


model, tier_art, meta, results = load_artifacts()
ch = meta["choices"]

# ------------------------------------------------------------------ language
with st.sidebar:
    lang_choice = st.radio("اللغة / Language", ["العربية", "English"], index=1, horizontal=True)
AR = lang_choice == "العربية"

TXT = {
    "title": ("💻 محرّك السعر العادل للّابتوب", "💻 Laptop Fair-Price & Value Engine"),
    "intro": ("**اللابتوب ده يستاهل سعره؟** اكتب مواصفات أي لابتوب، ونموذج تعلّم آلي (Machine Learning) هيقدّر سعره العادل "
              "بالجنيه، ويقولك لو العرض اللي جالك كويس ولا غالي، ويحدد الجهاز ده من أنهي مستوى.",
              "**Is this laptop worth its price?** Enter a laptop's specs and a machine-learning model estimates what it "
              "*should* cost (in EGP), tells you if a quote is a good deal, and groups it into a hardware tier."),
    "caption": ("النموذج: {m} · متوسط الخطأ حوالي {e}% · معامل التحديد R² = {r2:.2f} على {n} لابتوب النموذج عمره ما شافهم وقت التدريب",
                "Model: {m} · typical error ≈ {e}% · R² = {r2:.2f} on {n} laptops never seen during training"),
    "tabs": (["🔎 افحص جهاز واحد", "🗂️ افحص مجموعة أجهزة (CSV)", "ℹ️ إزاي بيشتغل؟"],
             ["🔎 Check one machine", "🗂️ Check a whole fleet (CSV)", "ℹ️ How it works"]),
    "steps": ("**الخطوة 1:** اختار مثال جاهز (أو اكتب مواصفاتك تحت) ← **الخطوة 2:** دوس على *احسب السعر العادل* ← **الخطوة 3:** اقرا النتيجة.",
              "**Step 1:** pick an example (or fill in your own below) → **Step 2:** click *Estimate fair price* → **Step 3:** read the result."),
    "body": ("##### الشكل والشاشة", "##### Body & screen"),
    "proc": ("##### المعالج والرام والكارت", "##### Processor, memory & graphics"),
    "stor": ("##### التخزين وحاجات إضافية", "##### Storage & extras"),
    "brand": ("الماركة", "Brand"), "type": ("النوع", "Type"), "inches": ("حجم الشاشة (بوصة)", "Screen size (inches)"),
    "res": ("دقة الشاشة", "Screen resolution"), "weight": ("الوزن (كجم)", "Weight (kg)"),
    "cpu": ("فئة المعالج", "CPU family"), "ghz": ("سرعة المعالج (GHz)", "CPU clock (GHz)"), "ram": ("الرام (GB)", "RAM (GB)"),
    "gpu": ("ماركة كارت الشاشة", "GPU brand"), "os": ("نظام التشغيل", "Operating system"),
    "ssd": ("SSD (GB)", "SSD (GB)"), "hdd": ("هارد HDD (GB)", "HDD (GB)"), "flash": ("ذاكرة فلاش (GB)", "Flash storage (GB)"),
    "hybrid": ("هارد هجين (GB)", "Hybrid drive (GB)"),
    "touch": ("شاشة لمس", "Touchscreen"), "ips": ("شاشة IPS (زوايا رؤية أحسن)", "IPS panel (better viewing angles)"),
    "quoted": ("السعر اللي اتعرض عليك بالجنيه (اختياري، 0 = تخطّي)", "Price you were quoted, in EGP (optional, 0 = skip)"),
    "btn": ("احسب السعر العادل", "Estimate fair price"),
    "nostore": ("من فضلك اختار على الأقل وحدة تخزين واحدة (SSD أو HDD أو فلاش أو هجين).",
                "Please enter at least one storage drive (SSD, HDD, flash or hybrid)."),
    "result": ("النتيجة", "Result"),
    "fair": ("السعر العادل المتوقع (جنيه)", "Estimated fair price (EGP)"),
    "range": ("النطاق المعتاد (جنيه)", "Typical range (EGP)"),
    "tier": ("أقرب مجموعة أجهزة", "Closest hardware group"),
    "tip": ("نصيحة: اكتب السعر اللي اتعرض عليك فوق عشان تعرف لو غالي ولا كويس.",
            "Tip: enter a quoted price above to get a good-deal / overpriced verdict."),
    "v_good": ("🟢 صفقة كويسة", "🟢 Good deal"), "v_fair": ("🟡 سعر عادل", "🟡 Fair price"), "v_over": ("🔴 غالي", "🔴 Overpriced"),
    "v_msg": ("{v} — السعر المعروض {q:,.0f} جنيه يعادل {p:.0f}% من السعر العادل اللي النموذج قدّره.",
              "{v} — the quoted price of {q:,.0f} EGP is {p:.0f}% of the model's fair price."),
    "lg_range": ("النطاق المعتاد", "Typical range"), "lg_fair": ("السعر العادل", "Fair price"), "lg_quoted": ("السعر المعروض", "Quoted price"),
    "outside": ("المواصفات دي برّه نطاق الأسعار اللي النموذج اتدرّب عليه، فاعتبر التقدير تقريبي جدًا.",
                "This configuration is outside the price range the model was trained on, so treat the estimate with caution."),
    "price_note": ("⚠️ الأسعار هنا **تقديرية**: الداتا الأصلية غالبًا بالروبية الهندية وأسعارها قديمة نسبيًا، واتحوّلت للجنيه بسعر صرف تقريبي "
                   "(1 روبية = {r} جنيه، تقدر تغيّره من القائمة الجانبية). استخدمها للمقارنة والتقدير، مش كسعر السوق المصري النهاردة. "
                   "النموذج اتعلّم من لابتوبات جديدة، فالمستعمل والحالة مش محسوبين.",
                   "⚠️ Prices are **approximate**: the original data is most likely in Indian rupees with older prices, converted to EGP at a rough "
                   "rate (1 INR = {r} EGP, adjustable in the sidebar). Use them for comparison, not as today's Egyptian market prices. "
                   "The model learned from new laptops, so used machines and condition are not modelled."),
    "side_rate": ("سعر الصرف: 1 روبية هندية = كام جنيه؟", "Exchange rate: 1 Indian rupee = how many EGP?"),
    "side_help": ("آخر قراءات في أوائل نوفمبر 2026 كانت حوالي 0.55 – 0.59. غيّره لو عايز.",
                  "Early-November 2026 readings were roughly 0.55 – 0.59. Change it if you like."),
    # fleet
    "fleet_intro": ("ارفع ملف CSV فيه جهاز في كل صف. استخدم القالب عشان أسماء الأعمدة تبقى مظبوطة. "
                    "عمود `Price` (اختياري، **بالجنيه**) بيضيف حكم صفقة كويسة/غالي لكل جهاز.",
                    "Upload a CSV with one machine per row. Use the template so the column names match. "
                    "An optional `Price` column (in **EGP**) adds a good-deal verdict for each machine."),
    "dl_tpl": ("⬇️ حمّل قالب CSV", "⬇️ Download CSV template"), "upload": ("ارفع ملف CSV", "Upload your CSV"),
    "empty": ("الملف فاضي.", "The file has no rows."),
    "badfile": ("مقدرتش أعالج الملف ده: {e}", "Could not process this file: {e}"),
    "dl_res": ("⬇️ حمّل النتائج", "⬇️ Download results"),
    # how it works
    "h_problem": ("### المشكلة اللي بيحلّها", "### What problem does this solve?"),
    "p_problem": ("كتير من مديري الـ IT والمشترين مش عارفين العرض اللي جاي لهم لابتوب سعره عادل ولا لأ. التطبيق ده اتعلّم من "
                  "{n:,} لابتوب حقيقي إزاي المواصفات (الرام، المعالج، التخزين، الشاشة، الماركة…) بتأثر على السعر، "
                  "وبعدين بيقدّر السعر العادل لأي تجميعة مواصفات.",
                  "IT managers and buyers often cannot tell whether a laptop quote is fair. This app learned from {n:,} real laptop listings "
                  "how specs (RAM, CPU, storage, screen, brand…) relate to price, then estimates a fair price for any configuration."),
    "h_build": ("### اتبنى إزاي؟", "### How was it built?"),
    "p_build": ("1. **نضّفنا** الداتا (نصوص زي `256GB SSD + 1TB HDD` اتحوّلت لأرقام، وشِلنا المكرر).\n"
                "2. **استكشفنا** الداتا: الرام وحجم SSD ووضوح الشاشة وسرعة المعالج هم أكتر حاجات بتأثر على السعر.\n"
                "3. **قسمنا** الداتا 80/20 *قبل* أي تعلّم، عشان لابتوبات الاختبار تفضل جديدة على النموذج.\n"
                "4. **قارنّا** خط أساس (يخمّن المتوسط دايمًا) مع Ridge وRandom Forest وGradient Boosting، وظبطنا الإعدادات بـ cross-validation (5 أجزاء).\n"
                "5. **جمّعنا** اللابتوبات في 4 مستويات أجهزة بـ K-Means (على المواصفات بس).",
                "1. **Cleaned** the data (text like `256GB SSD + 1TB HDD` turned into numbers; duplicates removed).\n"
                "2. **Explored** it: RAM, SSD size, screen sharpness and CPU speed drive price most.\n"
                "3. **Split** the data 80/20 *before* any learning, so the test laptops stay unseen.\n"
                "4. **Compared** a baseline (always guess the average) with Ridge, Random Forest and Gradient Boosting, tuned with 5-fold cross-validation.\n"
                "5. **Grouped** laptops into 4 hardware tiers with K-Means (on specs only)."),
    "h_res": ("### النتائج على لابتوبات النموذج ماشافهاش", "### Results on unseen laptops"),
    "cols": (["متوسط الخطأ (جنيه)", "RMSE (جنيه)", "R²"], ["Avg. error (EGP)", "RMSE (EGP)", "R²"]),
    "read": ("**إزاي تقراها:** خط الأساس بيغلط في المتوسط حوالي {b:,.0f} جنيه، والنموذج المختار ({m}) بيغلط حوالي {a:,.0f} جنيه. "
             "و{w}% من التوقعات بتيجي في حدود 20% من السعر الحقيقي.",
             "**Reading it:** the baseline is off by ~{b:,.0f} EGP on average; the chosen model ({m}) by ~{a:,.0f} EGP. "
             "{w}% of predictions land within 20% of the real price."),
    "rank": ("وكمان بيرتّب الأجهزة حسب السعر أحسن من نظام نقاط مكتوب بإيد (معامل الترتيب {a} مقابل {b}).",
             "It also ranks machines by price better than a hand-written points score (rank correlation {a} vs {b})."),
    "cap1": ("السعر المتوقع مقابل الحقيقي (الرسم بالإنجليزي، والأرقام بالوحدات الأصلية للداتا)", "Predicted vs actual price on unseen laptops"),
    "cap2": ("أهم المواصفات المؤثرة على السعر", "Which specs matter most"),
    "h_tiers": ("### مستويات الأجهزة الأربعة", "### The four hardware tiers"),
    "tier_note": ("المستويات بتجمّع الأجهزة حسب تشابه المواصفات مش حسب السعر، فممكن جهاز يقع في مستوى أقل من اللي سعره يوحي بيه.",
                  "Tiers group machines by similar hardware, not by price, so a machine can sit in a lower tier than its price suggests."),
    "h_lim": ("### حدود النموذج", "### Limits"),
    "p_lim": ("حوالي {n:,} لابتوب مختلف من فترة زمنية واحدة، أسعار أجهزة جديدة بس، من غير عمر الجهاز أو حالته، والخطأ بيزيد في الأجهزة الغالية جدًا. "
              "ده أداة مساعدة للقرار مش تقييم رسمي.",
              "About {n:,} unique laptops from one snapshot in time, new retail prices only, no age or condition, and larger errors on very "
              "expensive machines. This is a decision aid, not a valuation."),
}


def t(key, **kw):
    s = TXT[key][0 if AR else 1]
    return s.format(**kw) if kw else s


TIER_NAMES = {
    "Tier 1 · Entry": ("المستوى 1 · أساسي", "Tier 1 · Entry"),
    "Tier 2 · Mainstream": ("المستوى 2 · متوسط", "Tier 2 · Mainstream"),
    "Tier 3 · Slim Premium": ("المستوى 3 · نحيف مميّز", "Tier 3 · Slim Premium"),
    "Tier 4 · High-Performance": ("المستوى 4 · أداء عالي", "Tier 4 · High-Performance"),
}
TIER_INFO = {
    "Tier 1 · Entry": ("أجهزة أساسية: غالبًا رام 4 GB وهارد عادي وكارت شاشة مدمج.",
                       "Basic machines: typically ~4 GB RAM, a classic hard drive, integrated graphics."),
    "Tier 2 · Mainstream": ("أجهزة للاستخدام اليومي: غالبًا رام 8 GB، SSD مع هارد كبير، وكتير بكارت شاشة مستقل.",
                            "Everyday machines: typically ~8 GB RAM, SSD + large hard drive, often a dedicated GPU."),
    "Tier 3 · Slim Premium": ("نحيف وخفيف (حوالي 1.4 كجم) بشاشة واضحة وتخزين SSD، وأغلبه كارت شاشة مدمج.",
                              "Thin and light (about 1.4 kg) with a sharp screen and SSD storage, mostly integrated graphics."),
    "Tier 4 · High-Performance": ("قوي وأتقل: غالبًا رام 16 GB وكارت شاشة مستقل ومعالج أسرع.",
                                  "Powerful and heavier: typically ~16 GB RAM, dedicated graphics, faster CPU."),
}
TYPE_AR = {"2 in 1 Convertible": "2 في 1 (بيتحوّل لتابلت)", "Gaming": "جيمنج", "Netbook": "نت بوك (صغير)",
           "Notebook": "لابتوب عادي", "Ultrabook": "ألترابوك (نحيف)", "Workstation": "ورك ستيشن (شغل ثقيل)"}
OS_AR = {"Linux": "لينكس", "Other / No OS": "بدون نظام / أخرى", "Windows": "ويندوز", "macOS": "ماك (macOS)"}


def tier_label(name):
    return TIER_NAMES[name][0 if AR else 1]


PRESETS = {
    ("🏢 لابتوب مكتبي أساسي", "🏢 Basic office laptop"): dict(
        company="Acer", type_name="Notebook", inches=15.6, res="1366x768", weight=2.2, cpu_brand="Intel Core i3", ghz=2.0,
        ram=4.0, gpu_brand="Intel", os_group="Windows", ssd=0, hdd=500, flash=0, hybrid=0, touch=False, ips=False, quoted=0.0),
    ("💼 لابتوب شغل يومي", "💼 Everyday work laptop"): dict(
        company="Dell", type_name="Notebook", inches=15.6, res="1920x1080", weight=2.0, cpu_brand="Intel Core i5", ghz=2.5,
        ram=8.0, gpu_brand="Intel", os_group="Windows", ssd=256, hdd=0, flash=0, hybrid=0, touch=False, ips=True, quoted=0.0),
    ("✈️ ألترابوك نحيف", "✈️ Slim ultrabook"): dict(
        company="Lenovo", type_name="Ultrabook", inches=13.3, res="1920x1080", weight=1.3, cpu_brand="Intel Core i7", ghz=1.8,
        ram=16.0, gpu_brand="Intel", os_group="Windows", ssd=512, hdd=0, flash=0, hybrid=0, touch=False, ips=True, quoted=0.0),
    ("🎮 لابتوب جيمنج", "🎮 Gaming laptop"): dict(
        company="Asus", type_name="Gaming", inches=15.6, res="1920x1080", weight=2.6, cpu_brand="Intel Core i7", ghz=2.8,
        ram=16.0, gpu_brand="Nvidia", os_group="Windows", ssd=256, hdd=1000, flash=0, hybrid=0, touch=False, ips=True, quoted=0.0),
}
for k, v in list(PRESETS.values())[1].items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("rate", DEFAULT_RATE)


def load_preset(vals):
    for k, v in vals.items():
        st.session_state[k] = v


with st.sidebar:
    st.number_input(t("side_rate"), 0.20, 2.00, step=0.01, format="%.2f", key="rate", help=t("side_help"))
rate = float(st.session_state["rate"])

# ------------------------------------------------------------------ styling (RTL for Arabic)
if AR:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
html, body, [class*="css"], .stApp, button, input, select, textarea { font-family: 'Cairo', sans-serif !important; }
[data-testid="stAppViewContainer"], [data-testid="stSidebar"], .stMarkdown, [data-testid="stMetric"], label, .stAlert { direction: rtl; text-align: right; }
input[type=number], input[type=text] { direction: ltr; text-align: right; }
[data-testid="stTabs"] { direction: rtl; }
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def assign_tier(X: pd.DataFrame) -> pd.Series:
    cols = tier_art["cols"]
    Z = tier_art["scaler"].transform(X[cols].fillna(pd.Series(tier_art["fill"])))
    ranks = [tier_art["rank_of"][int(c)] for c in tier_art["kmeans"].predict(Z)]
    return pd.Series([tier_art["names"][r] for r in ranks], index=X.index)


def verdict_key(quoted_units, fair_units):
    ratio = quoted_units / fair_units
    return "v_good" if ratio < meta["deal_low_ratio"] else "v_over" if ratio > meta["deal_high_ratio"] else "v_fair"


def range_bar(lo, fair, hi, quoted):
    vals = [lo, hi] + ([quoted] if quoted > 0 else [])
    vmin, vmax = min(vals) * 0.85, max(vals) * 1.08
    pos = lambda v: (v - vmin) / (vmax - vmin) * 100
    q = (f'<div style="position:absolute;left:{pos(quoted):.1f}%;top:-6px;width:4px;height:30px;background:#dc2626;border-radius:2px"></div>'
         if quoted > 0 else "")
    legend = (f'<span style="color:#93c5fd">■</span> {t("lg_range")} &nbsp; <span style="color:#2563eb">■</span> {t("lg_fair")}'
              + (f' &nbsp; <span style="color:#dc2626">■</span> {t("lg_quoted")}' if quoted > 0 else ""))
    st.markdown(f"""
<div style="direction:ltr;margin:18px 6px 6px 6px">
  <div style="position:relative;height:18px;background:rgba(148,163,184,0.35);border-radius:9px">
    <div style="position:absolute;left:{pos(lo):.1f}%;width:{pos(hi)-pos(lo):.1f}%;height:18px;background:#93c5fd;border-radius:9px"></div>
    <div style="position:absolute;left:{pos(fair):.1f}%;top:-6px;width:4px;height:30px;background:#2563eb;border-radius:2px"></div>
    {q}
  </div>
  <div style="display:flex;justify-content:space-between;font-size:13px;color:#94a3b8;margin-top:6px">
    <span>{vmin:,.0f}</span><span>{vmax:,.0f}</span></div>
</div>
<div style="text-align:center;font-size:14px;margin-bottom:8px">{legend}</div>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ header
st.title(t("title"))
st.markdown(t("intro"))
st.caption(t("caption", m=meta["best_model"], e=meta["median_pct_err"], r2=meta["test_r2"], n=results.get("n_test", 255)))

tab1, tab2, tab3 = st.tabs(TXT["tabs"][0 if AR else 1])

# ------------------------------------------------------------------ single machine
with tab1:
    st.markdown(t("steps"))
    pcols = st.columns(len(PRESETS))
    for col, (names, vals) in zip(pcols, PRESETS.items()):
        col.button(names[0 if AR else 1], on_click=load_preset, args=(vals,), width="stretch")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(t("body"))
        st.selectbox(t("brand"), ch["Company"], key="company")
        st.selectbox(t("type"), ch["TypeName"], key="type_name", format_func=(lambda x: TYPE_AR.get(x, x)) if AR else str)
        st.selectbox(t("inches"), meta["inches"], key="inches")
        st.selectbox(t("res"), meta["resolutions"], key="res")
        st.number_input(t("weight"), 0.5, 5.0, step=0.1, key="weight")
    with c2:
        st.markdown(t("proc"))
        st.selectbox(t("cpu"), ch["Cpu_brand"], key="cpu_brand")
        st.number_input(t("ghz"), 0.9, 4.5, step=0.1, key="ghz")
        st.selectbox(t("ram"), meta["ram"], key="ram")
        st.selectbox(t("gpu"), ch["Gpu_brand"], key="gpu_brand")
        st.selectbox(t("os"), ch["OS_group"], key="os_group", format_func=(lambda x: OS_AR.get(x, x)) if AR else str)
    with c3:
        st.markdown(t("stor"))
        st.number_input(t("ssd"), 0, 2000, step=128, key="ssd")
        st.number_input(t("hdd"), 0, 4000, step=500, key="hdd")
        st.number_input(t("flash"), 0, 1000, step=32, key="flash")
        st.number_input(t("hybrid"), 0, 4000, step=500, key="hybrid")
        st.checkbox(t("touch"), key="touch")
        st.checkbox(t("ips"), key="ips")
        st.number_input(t("quoted"), 0.0, 1_000_000.0, step=500.0, key="quoted")

    if st.button(t("btn"), type="primary"):
        s = st.session_state
        if s.ssd + s.hdd + s.flash + s.hybrid == 0:
            st.error(t("nostore"))
        else:
            x, y = (int(v) for v in s.res.split("x"))
            row = pd.DataFrame([{
                "Company": s.company, "TypeName": s.type_name, "Cpu_brand": s.cpu_brand, "Gpu_brand": s.gpu_brand,
                "OS_group": s.os_group, "Inches": s.inches, "Ram_GB": s.ram, "Weight_kg": s.weight, "Cpu_GHz": s.ghz,
                "SSD_GB": s.ssd, "HDD_GB": s.hdd, "Hybrid_GB": s.hybrid, "Flash_GB": s.flash,
                "PPI": np.hypot(x, y) / s.inches, "Touchscreen": int(s.touch), "IPS": int(s.ips),
                "Dedicated_GPU": int(s.gpu_brand != "Intel"),
            }])[FEATURES]
            fair_u = float(model.predict(row)[0])           # model units (dataset currency)
            fair = fair_u * rate                              # EGP
            lo, hi = fair * meta["deal_low_ratio"], fair * meta["deal_high_ratio"]
            tier = assign_tier(row).iloc[0]
            quoted = float(s.quoted)                          # already EGP

            st.divider()
            st.subheader(t("result"))
            m1, m2, m3 = st.columns(3)
            m1.metric(t("fair"), f"{fair:,.0f}")
            m2.metric(t("range"), f"{lo:,.0f} – {hi:,.0f}")
            m3.metric(t("tier"), tier_label(tier))
            st.caption(f"ℹ️ {TIER_INFO[tier][0 if AR else 1]}")

            if quoted > 0:
                vk = verdict_key(quoted, fair)
                msg = t("v_msg", v=t(vk), q=quoted, p=quoted / fair * 100)
                (st.success if vk == "v_good" else st.error if vk == "v_over" else st.warning)(msg)
            else:
                st.info(t("tip"))
            range_bar(lo, fair, hi, quoted)

            if not (meta["price_min"] <= fair_u <= meta["price_max"]):
                st.warning(t("outside"))
            st.caption(t("price_note", r=f"{rate:.2f}"))

# ------------------------------------------------------------------ fleet CSV
with tab2:
    st.write(t("fleet_intro"))
    template = pd.DataFrame([
        ["Dell", "Notebook", 15.6, "Full HD 1920x1080", "Intel Core i5 7200U 2.5GHz", "8GB", "256GB SSD",
         "Intel HD Graphics 620", "Windows 10", "2.0kg", 26000]], columns=RAW_COLS + ["Price"])
    st.download_button(t("dl_tpl"), template.to_csv(index=False), "fleet_template.csv", "text/csv")
    up = st.file_uploader(t("upload"), type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            if df.empty:
                st.error(t("empty"))
            else:
                if set(RAW_COLS).issubset(df.columns):
                    X = parse_raw(df)
                elif set(FEATURES).issubset(df.columns):
                    X = df[FEATURES].copy()
                else:
                    raise ValueError(f"Missing columns: {sorted(set(RAW_COLS) - set(df.columns))}")
                out = df.copy()
                fair_egp = model.predict(X) * rate
                out["Fair_price_EGP"] = fair_egp.round(0)
                out["Tier"] = [tier_label(n) for n in assign_tier(X).values]
                if "Price" in df.columns:
                    q = pd.to_numeric(df["Price"], errors="coerce")
                    out["Verdict"] = [t(verdict_key(a, b)) if pd.notna(a) else "" for a, b in zip(q, fair_egp)]
                st.dataframe(out, width="stretch")
                st.bar_chart(out["Tier"].value_counts())
                st.download_button(t("dl_res"), out.to_csv(index=False), "fleet_results.csv", "text/csv")
        except Exception as e:  # wrong input must never crash the app
            st.error(t("badfile", e=e))

# ------------------------------------------------------------------ how it works
with tab3:
    n_rows = results.get("rows_after_dedup", 1274)
    st.markdown(t("h_problem")); st.write(t("p_problem", n=n_rows))
    st.markdown(t("h_build")); st.markdown(t("p_build"))
    if results.get("test"):
        st.markdown(t("h_res"))
        tbl = pd.DataFrame(results["test"]).T[["MAE", "RMSE", "R2"]].copy()
        tbl["MAE"] *= rate; tbl["RMSE"] *= rate
        tbl.columns = TXT["cols"][0 if AR else 1]
        st.dataframe(tbl.style.format({tbl.columns[0]: "{:,.0f}", tbl.columns[1]: "{:,.0f}", tbl.columns[2]: "{:.2f}"}), width="stretch")
        st.write(t("read", b=results["test"]["Baseline (mean)"]["MAE"] * rate, m=meta["best_model"],
                   a=meta["test_mae"] * rate, w=results.get("within_20pct", "")))
        if "spearman_model" in results:
            st.write(t("rank", a=results["spearman_model"], b=results["spearman_rule"]))
    a, b = st.columns(2)
    if (BASE / "assets" / "eval.png").exists():
        a.image(str(BASE / "assets" / "eval.png"), caption=t("cap1"))
    if (BASE / "assets" / "importance.png").exists():
        b.image(str(BASE / "assets" / "importance.png"), caption=t("cap2"))
    st.markdown(t("h_tiers"))
    for name in TIER_INFO:
        st.markdown(f"- **{tier_label(name)}** – {TIER_INFO[name][0 if AR else 1]}")
    st.caption(t("tier_note"))
    st.markdown(t("h_lim")); st.write(t("p_lim", n=n_rows))
    st.caption(t("price_note", r=f"{rate:.2f}"))
