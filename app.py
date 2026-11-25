"""Customer Churn Prediction: retention console (Streamlit demo).

Run:  streamlit run app.py

The model ranks customers by churn risk and explains why. The retention team decides
who to call and what to offer; the threshold comes from the business case, not from 0.5.
"""
import json
from html import escape
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.data import load
from src.explain import describe
from src.service import ChurnService, risk_band
from src.threshold import assumptions, curve, evaluate_policy, recommended_threshold
from src.train import FEATURES

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
st.set_page_config(page_title="Customer Churn Prediction", page_icon="◉", layout="wide")

# palette: black #0C0C14 · plum #44142C · wine #642444 · mist #E4E4EC · white  (+ rose #C24A73, a lighter wine for highlights)
BLACK, PANEL, PLUM, WINE, ROSE, MIST, MUTED = "#0C0C14", "#15151F", "#44142C", "#642444", "#C24A73", "#E4E4EC", "#9A9AAE"
BAND = {"High": ROSE, "Medium": "#B98AA0", "Low": "#6E6E84"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp, .stApp p, .stApp label, .stMarkdown {{ font-family:'Inter',sans-serif; }}
.stApp {{ background: radial-gradient(900px 420px at 100% -10%, rgba(100,36,68,.45), transparent 60%),
                      radial-gradient(700px 360px at -10% 0%, rgba(68,20,44,.40), transparent 60%), {BLACK}; }}
[data-testid="stSidebar"] {{ background:linear-gradient(180deg, #110F18, {PLUM}); border-right:1px solid #2A1A24; }}
[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:1.4rem; max-width:1380px; }}
[data-tag] {{ background:{WINE} !important; }}
[data-tag], [data-tag] * {{ color:#fff !important; fill:#fff !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap:.3rem; border-bottom:1px solid #2A1A24; }}
.stTabs [data-baseweb="tab"] {{ background:transparent; padding:.55rem .95rem; color:{MUTED}; }}
.stTabs [aria-selected="true"] {{ color:#fff !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{ROSE}; }}

.hero {{ display:flex; align-items:center; gap:1.3rem; padding:1.1rem 1.4rem; border-radius:18px; margin-bottom:1rem;
         background:linear-gradient(120deg, {PANEL} 0%, {PLUM} 75%, {WINE} 100%); border:1px solid #3A1C2C;
         box-shadow:0 10px 40px rgba(0,0,0,.45); }}
.hero h1 {{ margin:0; font-size:1.75rem; font-weight:800; color:#fff; letter-spacing:.01em; padding:0; }}
.hero p {{ margin:.25rem 0 0; color:#D8C9D2; font-size:.92rem; }}
.pulse {{ animation:pulse 2s ease-in-out infinite; transform-box:fill-box; transform-origin:center; }}
@keyframes pulse {{ 0%,100% {{ transform:scale(1); }} 50% {{ transform:scale(1.18); }} }}
.h {{ font-size:1.12rem; font-weight:700; color:#fff; margin:.5rem 0 .6rem; }}
.h small {{ color:{MUTED}; font-weight:400; font-size:.82rem; margin-left:.4rem; }}
.note {{ color:{MUTED}; font-size:.87rem; line-height:1.55; }}
.note b {{ color:{MIST}; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(175px, 1fr)); gap:.7rem; margin:.3rem 0 1rem; }}
.kpi {{ background:{PANEL}; border:1px solid #2A2333; border-radius:14px; padding:.85rem 1rem; }}
.kpi .l {{ color:{MUTED}; font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; font-weight:600; }}
.kpi .v {{ font-size:1.6rem; font-weight:800; color:#fff; margin-top:.1rem; }}
.kpi .d {{ color:{MUTED}; font-size:.78rem; }}
.kpi.hi {{ background:linear-gradient(135deg, {WINE}, {ROSE}); border:none; }}
.kpi.hi .l, .kpi.hi .d {{ color:#F6DCE6; }}
.card {{ background:{PANEL}; border:1px solid #2A2333; border-left:5px solid {ROSE}; border-radius:12px;
         padding:.8rem 1rem; margin-bottom:.5rem; }}
.card.Medium {{ border-left-color:#B98AA0; }} .card.Low {{ border-left-color:#6E6E84; }}
.top {{ display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; }}
.cid {{ font-weight:700; color:#fff; font-size:1rem; }}
.meta {{ color:{MUTED}; font-size:.8rem; margin-top:.1rem; }}
.score {{ text-align:right; }}
.score .n {{ font-size:1.35rem; font-weight:800; color:#fff; }}
.score .l {{ font-size:.62rem; letter-spacing:.14em; text-transform:uppercase; color:{MUTED}; }}
.bar {{ height:6px; background:#26202E; border-radius:99px; overflow:hidden; margin:.5rem 0 .45rem; }}
.bar > div {{ height:100%; border-radius:99px; background:linear-gradient(90deg, {WINE}, {ROSE}); }}
.reason {{ font-size:.84rem; color:{MIST}; margin-top:.15rem; }}
.reason::before {{ content:"▲ "; color:{ROSE}; font-size:.7rem; }}
.good {{ font-size:.84rem; color:{MUTED}; margin-top:.15rem; }}
.good::before {{ content:"▼ "; color:#8FA3B8; font-size:.7rem; }}
.pill {{ display:inline-block; padding:.12rem .55rem; border-radius:99px; font-size:.68rem; font-weight:700; color:#fff;
         letter-spacing:.06em; text-transform:uppercase; }}
.flow {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(165px, 1fr)); gap:.6rem; }}
.step {{ background:{PANEL}; border:1px solid #2A2333; border-top:4px solid {ROSE}; border-radius:12px; padding:.85rem;
         font-size:.84rem; color:{MUTED}; }}
.step b {{ display:block; color:#fff; margin-bottom:.25rem; }}
</style>
""", unsafe_allow_html=True)

HERO = f"""
<div class="hero">
  <svg width="74" height="64" viewBox="0 0 74 64">
    <rect x="4" y="10" width="50" height="40" rx="8" fill="none" stroke="{MIST}" stroke-width="3"/>
    <circle cx="20" cy="26" r="6" fill="{MIST}"/><path d="M11 42 q9 -10 18 0" fill="none" stroke="{MIST}" stroke-width="3" stroke-linecap="round"/>
    <path d="M35 24 h12 M35 32 h9" stroke="#B98AA0" stroke-width="3" stroke-linecap="round"/>
    <g class="pulse"><circle cx="56" cy="12" r="10" fill="{ROSE}"/>
    <text x="56" y="17" text-anchor="middle" font-size="14" font-weight="800" fill="#fff">!</text></g>
  </svg>
  <div><h1>Customer Churn Prediction</h1>
  <p>Find the subscribers most likely to cancel, see why, and decide who the retention team should call.</p></div>
</div>"""


@st.cache_resource(show_spinner="Loading model ...")
def get_service():
    return ChurnService()


@st.cache_data(show_spinner=False)
def report(name):
    p = REPORTS / name
    return json.loads(p.read_text()) if name.endswith(".json") else pd.read_csv(p)


@st.cache_data(show_spinner=False)
def val_scores():
    svc = get_service()
    v = load("val")
    return v, svc.score(v)


@st.cache_data(show_spinner=False)
def reasons_table(n):
    """Top-3 reasons for the n riskiest customers (vectorised contributions)."""
    svc = get_service()
    d = svc.customers.head(n)
    contrib = svc.explainer.contributions(d)
    out = []
    for i, r in d.iterrows():
        c = contrib.loc[i].sort_values(ascending=False)
        out.append([describe(g, r) for g, v in c.items() if v > 0.05][:3])
    return out


def show_chart(chart, grid=True):
    st.altair_chart(chart.configure(background=PANEL, padding=14)
                    .configure_axis(grid=grid, gridColor="#2E2838", gridDash=[3, 3], domainColor="#3A3446",
                                    tickColor="#3A3446", labelColor=MIST, titleColor=MUTED, labelFontSize=11)
                    .configure_view(strokeWidth=0)
                    .configure_legend(labelColor=MIST, titleColor=MUTED, orient="top"),
                    width="stretch", theme=None)


def pill(band):
    return f'<span class="pill" style="background:{BAND[band]}">{band} risk</span>'


svc = get_service()
T = svc.threshold

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div style="font-weight:800;font-size:1.1rem;color:#fff">Customer Churn Prediction</div>'
                f'<div style="color:{MUTED};font-size:.75rem;letter-spacing:.12em;text-transform:uppercase">Retention console</div>',
                unsafe_allow_html=True)
    st.caption("UneeQ Interns · ML Task: Customer Churn Prediction Pipeline · IBM Telco data")
    st.divider()
    st.markdown("**Model**")
    st.markdown("Logistic regression + engineered features")
    st.caption(f"Contact threshold: **{T}** (chosen from the retention business case on validation data).")
    st.divider()
    st.caption("The model only ranks customers and explains the score. People decide who to call and what to offer.")

st.markdown(HERO, unsafe_allow_html=True)
tabs = st.tabs(["At-risk customers", "Customer explorer", "Threshold & value", "Model comparison", "Insights",
                "How it works"])

# ------------------------------------------------------------------ at-risk list
with tabs[0]:
    cust = svc.customers
    flagged = cust[cust["score"] >= T]
    pol = evaluate_policy(cust, cust["score"].values, T, assumptions())
    st.markdown(f"""<div class="kpis">
      <div class="kpi"><div class="l">Customers scored</div><div class="v">{len(cust):,}</div><div class="d">held-out validation + test customers</div></div>
      <div class="kpi hi"><div class="l">To contact</div><div class="v">{len(flagged):,}</div><div class="d">{len(flagged)/len(cust):.0%} of customers, score ≥ {T}</div></div>
      <div class="kpi"><div class="l">Churners reached</div><div class="v">{pol['recall']:.0%}</div><div class="d">{pol['churners_caught']} of {int(cust['churn'].sum())} who actually left</div></div>
      <div class="kpi"><div class="l">Hit rate</div><div class="v">{pol['precision']:.0%}</div><div class="d">of contacted customers really were leaving</div></div>
      <div class="kpi"><div class="l">Expected net value</div><div class="v">${pol['net_value']/1000:,.0f}K</div><div class="d">saved revenue minus offer costs</div></div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    band_f = c1.multiselect("Risk band", ["High", "Medium"], default=["High", "Medium"])
    contract_f = c2.multiselect("Contract", sorted(cust["Contract"].unique()), default=sorted(cust["Contract"].unique()))
    n_show = c3.slider("Show", 5, 40, 12)
    st.markdown('<div class="h">Call list <small>riskiest first, with the reasons behind each score</small></div>',
                unsafe_allow_html=True)
    reasons = reasons_table(len(flagged))
    shown = 0
    for i, r in flagged.iterrows():
        band = risk_band(r["score"], T)
        if band not in band_f or r["Contract"] not in contract_f:
            continue
        rs = "".join(f'<div class="reason">{escape(x)}</div>' for x in reasons[i])
        st.markdown(f"""<div class="card {band}"><div class="top">
            <div><div class="cid">{r['customerID']} &nbsp;{pill(band)}</div>
            <div class="meta">{r['Contract']} · {int(r['tenure'])} months · ${r['MonthlyCharges']:.0f}/month · {r['InternetService']} internet · {r['PaymentMethod']}</div></div>
            <div class="score"><div class="n">{r['score']:.0%}</div><div class="l">churn risk</div></div></div>
            <div class="bar"><div style="width:{r['score']*100:.0f}%"></div></div>{rs}</div>""", unsafe_allow_html=True)
        shown += 1
        if shown >= n_show:
            break
    st.download_button("Download the full call list (CSV)",
                       flagged.assign(risk_band=[risk_band(s, T) for s in flagged["score"]],
                                      top_reasons=[" | ".join(x) for x in reasons])[
                           ["customerID", "score", "risk_band", "Contract", "tenure", "MonthlyCharges", "top_reasons"]]
                       .to_csv(index=False).encode(), "churn_call_list.csv")

# ------------------------------------------------------------------ explorer
with tabs[1]:
    st.markdown('<div class="h">One customer, in detail <small>change their plan and watch the risk move</small></div>',
                unsafe_allow_html=True)
    cid = st.selectbox("Customer", svc.customers["customerID"].tolist(), index=0)
    base = svc.customers[svc.customers["customerID"] == cid].iloc[0]
    raw_cols = [c for c in load("train").columns if c not in ("churn",) and not c.startswith(("n_services", "auto_pay", "charge_vs_avg", "new_customer", "log_tenure"))]
    left, right = st.columns([1, 1.25])
    with left:
        st.markdown("**What if …**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"],
                                index=["Month-to-month", "One year", "Two year"].index(base["Contract"]))
        tenure = st.slider("Tenure (months)", 0, 72, int(base["tenure"]))
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"],
                                index=["DSL", "Fiber optic", "No"].index(base["InternetService"]))
        support_opts = ["Yes", "No"] if internet != "No" else ["No internet service"]
        support = st.selectbox("Tech support", support_opts,
                               index=support_opts.index(base["TechSupport"]) if base["TechSupport"] in support_opts else 0)
        pay_opts = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        payment = st.selectbox("Payment method", pay_opts, index=pay_opts.index(base["PaymentMethod"]))
        charges = st.slider("Monthly charges ($)", 18.0, 120.0, float(base["MonthlyCharges"]), 0.5)
    new = {c: base[c] for c in raw_cols}
    ratio = base["TotalCharges"] / max(base["tenure"], 1)
    new.update(Contract=contract, tenure=tenure, InternetService=internet, TechSupport=support, PaymentMethod=payment,
               MonthlyCharges=charges, TotalCharges=round(ratio * tenure, 2))
    if internet == "No":
        for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
            new[c] = "No internet service"
    elif base["InternetService"] == "No":
        for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies"]:
            new[c] = "No"
    res = svc.score_customer(new)
    with right:
        delta = res["churn_probability"] - base["score"]
        band = res["risk_band"]
        good = "".join(f'<div class="good">{escape(x)}</div>' for x in res["protective"])
        bad = "".join(f'<div class="reason">{escape(x)}</div>' for x in res["reasons"]) or '<div class="note">No strong risk factors.</div>'
        st.markdown(f"""<div class="card {band}" style="padding:1.1rem 1.2rem"><div class="top">
            <div><div class="cid">{cid} &nbsp;{pill(band)}</div>
            <div class="meta">Actual outcome in the data: <b style="color:{MIST}">{'cancelled' if base['churn'] else 'stayed'}</b></div></div>
            <div class="score"><div class="n">{res['churn_probability']:.0%}</div>
            <div class="l">{'was ' + format(base['score'], '.0%') if abs(delta) > 0.005 else 'churn risk'}</div></div></div>
            <div class="bar"><div style="width:{res['churn_probability']*100:.0f}%"></div></div>
            <div class="note" style="margin-top:.4rem"><b>Pushing risk up</b></div>{bad}
            <div class="note" style="margin-top:.6rem"><b>Keeping them</b></div>{good}
            <div class="note" style="margin-top:.8rem">Recommendation: <b>{'contact with a retention offer' if res['contact'] else 'no action needed'}</b>
            · offer cost ${res['offer_cost']:.0f} · expected value of contacting ${res['expected_value_if_contacted']:.0f}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('<div class="note">Try switching the contract to <b>One year</b> or turning on <b>tech support</b>: '
                    'those are the levers a retention offer can actually pull.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ threshold & value
with tabs[2]:
    st.markdown('<div class="h">Who should we contact? <small>the threshold comes from money, not from 0.5</small></div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    a = assumptions(contact_cost=c1.number_input("Contact cost ($)", 0.0, 50.0, 5.0, 1.0),
                    offer_discount=c2.slider("Discount", 0.0, 0.5, 0.20, 0.05, format="%.2f"),
                    offer_months=c3.slider("Discount months", 1, 6, 3),
                    save_rate=c4.slider("Offer saves this share of churners", 0.05, 0.8, 0.30, 0.05),
                    value_months=c5.slider("Months of revenue a saved customer is worth", 3, 36, 12))
    v, sv = val_scores()
    t_rec, cv = recommended_threshold(v, sv, a)
    t_rec = min(t_rec, 0.95)
    pick = st.slider("Threshold", 0.02, 0.95, float(t_rec), 0.01,
                     help="Customers with a churn score at or above this get a retention offer.")
    m = evaluate_policy(v, sv, pick, a)
    st.markdown(f"""<div class="kpis">
      <div class="kpi hi"><div class="l">Recommended threshold</div><div class="v">{t_rec:.2f}</div><div class="d">highest threshold within 2% of the best value</div></div>
      <div class="kpi"><div class="l">Contacted</div><div class="v">{m['contacted_share']:.0%}</div><div class="d">{m['contacted']} of {len(v)} validation customers</div></div>
      <div class="kpi"><div class="l">Recall</div><div class="v">{m['recall']:.0%}</div><div class="d">of churners get an offer</div></div>
      <div class="kpi"><div class="l">Precision</div><div class="v">{m['precision']:.0%}</div><div class="d">of offers go to real churners</div></div>
      <div class="kpi"><div class="l">Net value</div><div class="v">${m['net_value']/1000:,.1f}K</div><div class="d">saved ${m['value_saved']/1000:,.1f}K − cost ${m['campaign_cost']/1000:,.1f}K</div></div>
    </div>""", unsafe_allow_html=True)
    long = cv.melt(id_vars="threshold", value_vars=["precision", "recall"], var_name="metric", value_name="v")
    rule = alt.Chart(pd.DataFrame({"t": [pick]})).mark_rule(color=ROSE, strokeWidth=2).encode(x="t:Q")
    left, right = st.columns(2)
    with left:
        nv = alt.Chart(cv).mark_area(line={"color": ROSE, "strokeWidth": 2}, color=alt.Gradient(
            gradient="linear", stops=[alt.GradientStop(color=BLACK, offset=0), alt.GradientStop(color=WINE, offset=1)],
            x1=1, x2=1, y1=1, y2=0)).encode(
            x=alt.X("threshold:Q", title="threshold"), y=alt.Y("net_value:Q", title="net value ($)"),
            tooltip=[alt.Tooltip("threshold:Q"), alt.Tooltip("net_value:Q", format="$,.0f"), "contacted"])
        st.markdown("**Net value of the campaign**")
        show_chart((nv + rule).properties(height=280))
    with right:
        pr = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
            x=alt.X("threshold:Q", title="threshold"), y=alt.Y("v:Q", title=None, axis=alt.Axis(format="%")),
            color=alt.Color("metric:N", scale=alt.Scale(domain=["precision", "recall"], range=[MIST, ROSE]), title=None))
        st.markdown("**Precision vs. recall**")
        show_chart((pr + rule).properties(height=280))
    st.markdown("""<div class="note">The value curve is <b>flat near its peak</b>, so we take the highest threshold that still earns
        98% of the best value: almost the same money with fewer customers contacted and fewer wasted discounts.
        <b>Cheaper offers</b> or <b>more valuable customers</b> push the threshold down (contact more people);
        <b>expensive offers</b> push it up.</div>""", unsafe_allow_html=True)
    pol = report("threshold_policies_val.csv")
    st.markdown('<div class="h" style="margin-top:1rem">Policies compared <small>validation data, default assumptions</small></div>',
                unsafe_allow_html=True)
    st.dataframe(pol[["policy", "threshold", "contacted", "precision", "recall", "net_value"]]
                 .style.format({"threshold": "{:.2f}", "precision": "{:.0%}", "recall": "{:.0%}", "net_value": "${:,.0f}"}),
                 hide_index=True, width="stretch")

# ------------------------------------------------------------------ model comparison
with tabs[3]:
    res_t, comp = report("test_results.json"), report("model_comparison.csv")
    lo, hi = res_t["pr_auc_diff"]["ci95"]
    st.markdown(f"""<div class="kpis">
      <div class="kpi hi"><div class="l">Improved model</div><div class="v">{res_t['improved']['pr_auc']:.3f}</div><div class="d">test PR-AUC · ROC-AUC {res_t['improved']['roc_auc']:.3f}</div></div>
      <div class="kpi"><div class="l">Baseline</div><div class="v">{res_t['baseline']['pr_auc']:.3f}</div><div class="d">test PR-AUC · ROC-AUC {res_t['baseline']['roc_auc']:.3f}</div></div>
      <div class="kpi"><div class="l">Improvement</div><div class="v">+{res_t['pr_auc_diff']['diff']:.3f}</div><div class="d">95% CI +{lo:.3f} to +{hi:.3f}</div></div>
      <div class="kpi"><div class="l">No model</div><div class="v">0.265</div><div class="d">PR-AUC of random guessing</div></div>
    </div>""", unsafe_allow_html=True)
    comp = comp.assign(label=comp["model"] + " · " + comp["imbalance"])
    bars = alt.Chart(comp).mark_bar(cornerRadiusEnd=4, height=14).encode(
        x=alt.X("pr_auc:Q", title="validation PR-AUC", scale=alt.Scale(domain=[0, 0.8])),
        y=alt.Y("label:N", title=None, sort="-x", axis=alt.Axis(labelLimit=330)),
        color=alt.Color("family:N", scale=alt.Scale(domain=["no model", "baseline", "improved"], range=["#6E6E84", "#B98AA0", ROSE]),
                        title=None),
        tooltip=["model", "imbalance", alt.Tooltip("pr_auc:Q", format=".3f"), alt.Tooltip("cv_pr_auc:Q", format=".3f")])
    st.markdown('<div class="h">Every model we tried <small>trained on train, compared on validation</small></div>',
                unsafe_allow_html=True)
    show_chart((bars + bars.mark_text(align="left", dx=4, color=MIST).encode(text=alt.Text("pr_auc:Q", format=".3f")))
               .properties(height=alt.Step(24)))
    st.markdown("""<div class="note"><b>Engineered features beat bigger models.</b> Gradient boosting and random forest did not beat the
        logistic regression on this small, mostly categorical dataset. <b>Imbalance handling</b> (class weights, SMOTE,
        undersampling) barely changed the ranking (PR-AUC 0.689–0.693); it only shifts the scores, so imbalance is handled
        where it matters: the business-based threshold.</div>""", unsafe_allow_html=True)

    test = load("test")
    yv = test["churn"].values
    curves = []
    for name, model in [("Improved", svc.model), ("Baseline", svc.baseline)]:
        p = model.predict_proba(test[FEATURES])[:, 1]
        pr_, rc_, _ = precision_recall_curve(yv, p)
        curves.append(pd.DataFrame({"recall": rc_, "precision": pr_, "model": f"{name} · PR-AUC {average_precision_score(yv, p):.3f}"}))
    cdf = pd.concat(curves)
    left, right = st.columns(2)
    with left:
        st.markdown("**Precision–recall curve (test)**")
        show_chart(alt.Chart(cdf).mark_line(strokeWidth=2.5).encode(
            x="recall:Q", y="precision:Q",
            color=alt.Color("model:N", scale=alt.Scale(domain=list(cdf["model"].unique()), range=[ROSE, "#8FA3B8"]),
                            title=None)).properties(height=290))
    with right:
        cm = np.array(res_t["confusion_matrix"])
        m = pd.DataFrame([{"actual": a_, "decision": d_, "n": int(cm[i, j]), "pct": cm[i, j] / cm[i].sum()}
                          for i, a_ in enumerate(["stayed", "churned"]) for j, d_ in enumerate(["not contacted", "contacted"])])
        base_c = alt.Chart(m).encode(x=alt.X("decision:N", sort=["not contacted", "contacted"], title=None, axis=alt.Axis(labelAngle=0)),
                                     y=alt.Y("actual:N", sort=["stayed", "churned"], title=None))
        heat = base_c.mark_rect(stroke=BLACK, strokeWidth=3, cornerRadius=6).encode(
            color=alt.Color("pct:Q", scale=alt.Scale(domain=[0, 1], range=[PLUM, ROSE]), legend=None))
        txt = base_c.mark_text(fontSize=15, fontWeight="bold", color="#fff").encode(
            text=alt.Text("label:N")).transform_calculate(label="datum.n + '  (' + format(datum.pct, '.0%') + ')'")
        st.markdown(f"**Confusion matrix at threshold {T} (test)**")
        show_chart((heat + txt).properties(height=290), grid=False)

# ------------------------------------------------------------------ insights
with tabs[4]:
    eda, imp, fair = report("eda_summary.csv"), report("feature_importance.csv"), report("fairness.csv")
    st.markdown('<div class="h">Who cancels? <small>churn rate in the training data · dotted line = 26.5% average</small></div>',
                unsafe_allow_html=True)
    cols = st.columns(3)
    for col, feat in zip(cols * 2, ["Contract", "Tenure", "Internet service", "Payment method", "Monthly charges", "Tech support"]):
        d = eda[eda["feature"] == feat]
        ch = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=16).encode(
            x=alt.X("churn_rate:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.6])),
            y=alt.Y("value:N", title=None, sort=list(d["value"]), axis=alt.Axis(labelLimit=180)),
            color=alt.condition("datum.churn_rate > 0.265", alt.value(ROSE), alt.value("#6E6E84")),
            tooltip=["value", alt.Tooltip("churn_rate:Q", format=".1%"), "customers"])
        avg = alt.Chart(pd.DataFrame({"x": [0.265]})).mark_rule(color=MIST, strokeDash=[3, 3]).encode(x="x:Q")
        with col:
            st.markdown(f"**{feat}**")
            show_chart((ch + avg).properties(height=alt.Step(26)))
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown('<div class="h">What drives the model</div>', unsafe_allow_html=True)
        imp.columns = ["feature", "impact"]
        top = imp.head(10)
        show_chart(alt.Chart(top).mark_bar(cornerRadiusEnd=4, height=16, color=ROSE).encode(
            x=alt.X("impact:Q", title="average impact on the churn score"), y=alt.Y("feature:N", sort="-x", title=None))
            .properties(height=alt.Step(26)))
    with right:
        st.markdown('<div class="h">Fairness check <small>test set, at the chosen threshold</small></div>', unsafe_allow_html=True)
        st.dataframe(fair.style.format({"churn_rate": "{:.0%}", "flagged_rate": "{:.0%}", "recall": "{:.0%}", "precision": "{:.0%}"}),
                     hide_index=True, width="stretch")
        st.markdown('<div class="note">Gender is <b>not</b> a model input, and recall is the same for women and men (75%). '
                    'Seniors and customers without a partner are flagged more often because they really do churn more.</div>',
                    unsafe_allow_html=True)

# ------------------------------------------------------------------ how it works
with tabs[5]:
    st.markdown('<div class="h">Pipeline <small>python -m src.pipeline reruns all of it</small></div>', unsafe_allow_html=True)
    st.markdown("""<div class="flow">
      <div class="step"><b>1 · Data</b>7,043 Telco customers, 26.5% churn. Blank TotalCharges for brand-new customers set to 0.</div>
      <div class="step"><b>2 · Features</b>Number of services, auto-pay, price change vs. average bill, new customer, log tenure.</div>
      <div class="step"><b>3 · Split</b>60/20/20 stratified. Every choice on validation, test used once.</div>
      <div class="step"><b>4 · Models</b>Baseline logistic regression vs. improved + random forest, gradient boosting, 4 imbalance strategies.</div>
      <div class="step"><b>5 · Threshold</b>Retention business case: offer cost vs. revenue saved, plus sensitivity analysis.</div>
      <div class="step"><b>6 · Explain & serve</b>Per-customer reasons, fairness check, this app and a FastAPI endpoint.</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="h" style="margin-top:1.2rem">How the business should use it</div>', unsafe_allow_html=True)
    st.markdown("""
- **Monthly:** score every customer, export the call list (riskiest first) and hand it to the retention team.
- **Tailor the offer to the reason:** month-to-month → discount for a 1-year contract; no tech support → free support trial; electronic check → incentive to switch to auto-pay.
- **Re-tune the threshold** whenever offer costs or success rates change; the *Threshold & value* tab does it live.
- **Measure:** keep a small random control group without offers, so the real save rate can replace the 30% assumption.
- **Retrain** every quarter and re-check the fairness table.

**Limitations:** single snapshot (no dates), so the split is random rather than time-based; the business numbers are assumptions;
the save rate is not measured yet; the explanations show association, not cause.
""")
