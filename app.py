# =====================================================
# Supply-Chain Strategy Agent (LCE + 5S)
# Authors: Dr. J. Isabel Méndez & Dr. Arturo Molina
# =====================================================

import os, json, time, re
from typing import Dict, List
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px

# =====================================================
#                   SETUP
# =====================================================
st.set_page_config(page_title="Supply-Chain Agent (LCE+5S)", layout="wide")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"]
)
OPENROUTER_HEADERS = {
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "LCE+5S Supply-Chain Agent"
}

LLM_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

# =====================================================
#                CANONICAL VOCAB
# =====================================================
SYSTEMS = ["Product Transfer","Technology Transfer","Facility Design"]
LCE = ["Ideation","Basic Development","Advanced Development","Launch","Operation","End-of-Life"]
FIVE_S = ["Social","Sustainable","Sensing","Smart","Safe"]
PILLARS = ["quality","cost","volume","time","flexibility","environment"]

COMPETITIVE={
    "Product Transfer":"Operational Excellence",
    "Technology Transfer":"Product Innovation",
    "Facility Design":"Customer Focus"
}
VALUE_CHAIN={
    "Product Transfer":"Collaboration Networks",
    "Technology Transfer":"Strategic Business Units",
    "Facility Design":"Vertical Integration"
}
PROD_SERVICE={
    "Product Transfer":"Assemble to Order (ATO) + Catalogue of Services",
    "Technology Transfer":"Make/Build to Order (MTO/BTO) + Configuration of Services",
    "Facility Design":"Hybrid MTS + BTO + Design of Services"
}

# =====================================================
#              BASELINES (simplified)
# =====================================================
BASE_CORE = {
    "Customer Driven Design":{"Product Transfer":1,"Technology Transfer":3,"Facility Design":3},
    "CRM":{"Product Transfer":2,"Technology Transfer":2,"Facility Design":3},
    "Co-Engineering":{"Product Transfer":3,"Technology Transfer":1,"Facility Design":1},
    "SRM":{"Product Transfer":3,"Technology Transfer":3,"Facility Design":1},
    "Order Fulfillment":{"Product Transfer":3,"Technology Transfer":3,"Facility Design":3},
    "Customer Service":{"Product Transfer":1,"Technology Transfer":1,"Facility Design":3},
}
BASE_KPIS = {
    "Supplier on-time delivery":{"Product Transfer":3,"Technology Transfer":0,"Facility Design":0},
    "Ramp-up time":{"Product Transfer":0,"Technology Transfer":3,"Facility Design":0},
    "First-pass yield":{"Product Transfer":0,"Technology Transfer":3,"Facility Design":0},
    "OEE":{"Product Transfer":0,"Technology Transfer":0,"Facility Design":3},
    "ESG index":{"Product Transfer":0,"Technology Transfer":0,"Facility Design":3},
    "Safety incidents":{"Product Transfer":0,"Technology Transfer":0,"Facility Design":3},
}
BASE_DRIVERS = {
    "Inventory/Capacity Buffers":{"Product Transfer":1,"Technology Transfer":3,"Facility Design":3},
    "Network Diversification":{"Product Transfer":3,"Technology Transfer":3,"Facility Design":1},
    "Multisourcing":{"Product Transfer":3,"Technology Transfer":1,"Facility Design":1},
    "Nearshoring":{"Product Transfer":2,"Technology Transfer":2,"Facility Design":2},
    "Ecosystem Partnerships":{"Product Transfer":3,"Technology Transfer":3,"Facility Design":1},
}

# =====================================================
#           SCORING & QUALITATIVE LABELS
# =====================================================
def qualitative_label(v):
    if v < 1.0: return "Low"
    elif v < 2.0: return "Medium"
    return "High"

def normalize_dict(d):
    s = sum(d.values()) or 1
    return {k:v/s for k,v in d.items()}

# =====================================================
#           LLM INTERFACES
# =====================================================
def analyze_to_pillars(payload: Dict) -> Dict:
    msgs = [
        {"role":"system","content":"You are a Supply Chain Director. Output ONLY JSON."},
        {"role":"user","content":json.dumps(payload, ensure_ascii=False)}
    ]
    r = client.chat.completions.create(
        model=LLM_MODEL, messages=msgs,
        extra_headers=OPENROUTER_HEADERS,
        response_format={"type":"json_object"},
        temperature=0.2, max_tokens=700
    )
    return json.loads(r.choices[0].message.content)

def make_single_guidance(ctx: Dict) -> str:
    msgs=[
        {"role":"system","content":"You are a senior supply-chain advisor. Speak directly to the user in ≤160 words."},
        {"role":"user","content":json.dumps(ctx, ensure_ascii=False)}
    ]
    r = client.chat.completions.create(
        model=LLM_MODEL, messages=msgs,
        extra_headers=OPENROUTER_HEADERS,
        temperature=0.3, max_tokens=700
    )
    return r.choices[0].message.content

# =====================================================
#           VISUAL FUNCTIONS
# =====================================================
def radar_plot(pillars_dict):
    df = pd.DataFrame(list(pillars_dict.items()), columns=["Pillar","Weight"])
    fig = px.line_polar(df, r="Weight", theta="Pillar", line_close=True)
    fig.update_traces(fill="toself")
    st.plotly_chart(fig, use_container_width=True)

def sensitivity_scenarios(drivers):
    scenarios = {
        "Baseline": 1.0,
        "Volatility": 1.2,
        "Geo-Risk": 1.3,
        "Carbon Constraint": 1.1,
        "Combined Stress": 1.5
    }
    rows=[]
    for s,f in scenarios.items():
        avg = np.mean([np.mean(list(v.values())) for v in drivers.values()]) * f
        rows.append({"Scenario":s,"Average Score":avg})
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Scenario", y="Average Score",
                 color="Scenario", title="Resilience under Disruption Scenarios",
                 color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each bar shows the average resilience level when systems face different disruption contexts. Higher bars = greater adaptive capacity.")

def extract_keywords(text, topn=8):
    text = re.sub(r"[^A-Za-z\s]", "", text)
    vec = CountVectorizer(stop_words="english")
    X = vec.fit_transform([text.lower()])
    freqs = zip(vec.get_feature_names_out(), X.toarray().flatten())
    return [w for w,_ in sorted(freqs, key=lambda x:-x[1])[:topn]]

# =====================================================
#           SIDEBAR CONFIGURATION
# =====================================================
with st.sidebar:
    st.header("Context Configuration")
    system = st.selectbox("Manufacturing System", SYSTEMS)
    stage = st.selectbox("LCE Stage", LCE)
    industry = st.selectbox("Industry", ["Automotive","Electronics","Medical Devices","Consumer Goods","Other"])
    objective = st.text_input("Objective", "Stabilize outsourced supply performance.")
    role = st.selectbox("Your Role", ["Engineer","Analyst","Manager","Decision Maker"])
    st.header("5S Priorities")
    weights_5s = {s: st.slider(s, 0.0, 1.0, 0.5, 0.05) for s in FIVE_S}
    compare_all = st.toggle("Compare all systems", value=False)

# =====================================================
#              MAIN EXECUTION
# =====================================================
st.title("Supply-Chain Strategy Agent (LCE + 5S)")
st.markdown("Developed by: **Dr. J. Isabel Méndez** & **Dr. Arturo Molina**")

if st.button("Analyze", use_container_width=True):
    payload = {
        "objective": objective,
        "industry": industry,
        "user_role": role,
        "system_type": system,
        "lce_stage": stage,
        "weights_5s": weights_5s
    }

    with st.spinner("Analyzing through LLM…"):
        t0 = time.time()
        try:
            analysis = analyze_to_pillars(payload)
            pillars = normalize_dict(analysis.get("pillars", {p:1/6 for p in PILLARS}))
            reasons = analysis.get("reasons", [])
        except Exception:
            pillars = {p:1/6 for p in PILLARS}
            reasons = ["Default equal weighting applied."]
        elapsed = time.time()-t0

    # deterministic simulated scoring
    scored = {
        "core_processes": BASE_CORE,
        "kpis": BASE_KPIS,
        "drivers": BASE_DRIVERS
    }
    guidance = make_single_guidance(payload)

    st.session_state["results"] = {
        "pillars": pillars,
        "scored": scored,
        "guidance": guidance,
        "elapsed": elapsed
    }

# =====================================================
#           DISPLAY RESULTS
# =====================================================
if "results" in st.session_state:
    res = st.session_state["results"]
    st.success(f"LLM completed in {res['elapsed']:.1f} s")

    # --- Pillars
    st.subheader("Pillar Distribution")
    radar_plot(res["pillars"])
    st.caption("The radar chart illustrates the relative strategic weight of each operational pillar.")

    # --- Core Processes
    st.subheader("Core Processes × System")
    df_core = pd.DataFrame(res["scored"]["core_processes"])
    if not compare_all:
        df_core = df_core[[system]]
    dfq_core = df_core.applymap(qualitative_label)
    st.dataframe(dfq_core)
    st.markdown("_High = Core Process / Medium = Support Process / Low = Peripheral Process_")

    # --- KPIs
    st.subheader("KPIs × System")
    df_kpi = pd.DataFrame(res["scored"]["kpis"])
    if not compare_all:
        df_kpi = df_kpi[[system]]
    dfq_kpi = df_kpi.applymap(qualitative_label)
    st.dataframe(dfq_kpi)
    st.markdown("_High = Critical Performance Metric / Medium = Relevant / Low = Minor Indicator_")

    # --- Drivers
    st.subheader("Resilience Drivers × System")
    df_drv = pd.DataFrame(res["scored"]["drivers"])
    if not compare_all:
        df_drv = df_drv[[system]]
    dfq_drv = df_drv.applymap(qualitative_label)
    st.dataframe(dfq_drv)
    st.markdown("_High = Strategic Resilience Factor / Medium = Supportive / Low = Peripheral_")

    # --- Guidance
    st.subheader("LLM Guidance Summary")
    st.write(res["guidance"])

    # --- Sensitivity
    st.subheader("Sensitivity Simulation")
    sensitivity_scenarios(res["scored"]["drivers"])

    # --- Strategic Emphasis
    st.subheader("Strategic Emphasis Summary")
    top = extract_keywords(res["guidance"])
    st.markdown(" | ".join([f"🟢 **{t}**" for t in top]))
    st.caption("Most emphasized strategic concepts in LLM guidance, reflecting dominant themes from the framework section.")
    # --- Export option
    st.markdown("### Export Results")
    export_df = pd.concat(
        {
            "Core Processes": dfq_core,
            "KPIs": dfq_kpi,
            "Resilience Drivers": dfq_drv
        },
        axis=1
    )
    csv_data = export_df.to_csv(index=True).encode("utf-8")
    st.download_button(
        label="⬇️ Download full qualitative results as CSV",
        data=csv_data,
        file_name="supplychain_results.csv",
        mime="text/csv",
    )

    # --- Notes for interpretation
    st.markdown("### Analytical Interpretation Notes")
    st.info(
        "High scores correspond to *core or strategic* importance under the current LCE stage "
        "and 5S weighting. Medium reflects supportive processes that contribute to performance "
        "stability but are not key differentiators. Low indicates limited short-term impact or "
        "context-specific functions."
    )

# =====================================================
#                CHAT EXTENSION
# =====================================================
st.markdown("---")
st.subheader("Interactive Advisor")

if "chat" not in st.session_state:
    st.session_state["chat"] = []

for msg in st.session_state["chat"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask the Strategy Agent about trade-offs, improvements, or next steps…")

if user_input:
    st.session_state["chat"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if "results" not in st.session_state:
        reply = "Please run **Analyze** first to generate context."
    else:
        ctx = {
            "pillars": res["pillars"],
            "weights_5s": res["scored"],
            "guidance": res["guidance"]
        }
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a supply-chain strategy expert offering concise, evidence-based advice."},
                {"role": "user", "content": json.dumps(ctx, ensure_ascii=False)},
                {"role": "user", "content": user_input},
            ],
            extra_headers=OPENROUTER_HEADERS,
            temperature=0.4,
            max_tokens=600,
        )
        reply = r.choices[0].message.content

    st.session_state["chat"].append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
