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
# DYNAMIC MODIFIERS
# =====================================================
def stage_modifier(stage:str)->float:
    """Return intensity multiplier by LCE stage."""
    mapping={
        "Ideation":0.8,"Basic Development":0.9,"Advanced Development":1.0,
        "Launch":1.1,"Operation":1.2,"End-of-Life":1.0
    }
    return mapping.get(stage,1.0)

def fiveS_modifier(weights:Dict[str,float],matrix:str)->float:
    """Weight influence by 5S profile and matrix type."""
    if matrix=="core_processes": focus=["Social","Smart"]
    elif matrix=="kpis": focus=["Smart","Safe","Sustainable"]
    else: focus=["Sustainable","Safe","Social"]
    return 1+0.4*np.mean([weights.get(s,0.5) for s in focus])-0.2

def dynamic_score(base_map, stage, weights):
    s_mod = stage_modifier(stage)
    scored = {}
    for item, vals in base_map.items():
        scored[item] = {}
        for sys, val in vals.items():
            f_mod = fiveS_modifier(weights, "drivers")
            noise = np.random.uniform(0.9, 1.05)
            scored[item][sys] = np.clip(val * s_mod * f_mod * noise, 0, 3)
    return scored

# =====================================================
# HELPERS
# =====================================================
def qualitative_label(v):
    if v<1: return "Low"
    elif v<2: return "Medium"
    return "High"

def normalize_dict(d):
    s=sum(d.values()) or 1
    return {k:v/s for k,v in d.items()}

def radar_plot(pillars):
    df=pd.DataFrame(list(pillars.items()),columns=["Pillar","Weight"])
    fig=px.line_polar(df,r="Weight",theta="Pillar",line_close=True)
    fig.update_traces(fill="toself")
    st.plotly_chart(fig,use_container_width=True)

def sensitivity_scenarios(drivers):
    scen={"Baseline":1.0,"Volatility":1.2,"Geo-Risk":1.3,
          "Carbon Constraint":1.1,"Combined Stress":1.5}
    rows=[{"Scenario":k,"Avg":np.mean([np.mean(list(v.values())) for v in drivers.values()])*f}
          for k,f in scen.items()]
    df=pd.DataFrame(rows)
    fig=px.bar(df,x="Scenario",y="Avg",color="Scenario",
               title="Resilience under Disruption Scenarios",
               color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Higher bars = greater adaptive capacity under disruption.")

def extract_keywords(text,topn=8):
    text=re.sub(r"[^A-Za-z\s]","",text)
    vec=CountVectorizer(stop_words="english")
    X=vec.fit_transform([text.lower()])
    freqs=zip(vec.get_feature_names_out(),X.toarray().flatten())
    return [w for w,_ in sorted(freqs,key=lambda x:-x[1])[:topn]]

# =====================================================
#           LLM INTERFACES
# =====================================================
def analyze_to_pillars(payload:Dict)->Dict:
    msgs=[{"role":"system","content":"You are a Supply-Chain Director. Output ONLY JSON."},
          {"role":"user","content":json.dumps(payload,ensure_ascii=False)}]
    r=client.chat.completions.create(model=LLM_MODEL,messages=msgs,
        extra_headers=OPENROUTER_HEADERS,response_format={"type":"json_object"},
        temperature=0.2,max_tokens=700)
    return json.loads(r.choices[0].message.content)

def make_single_guidance(ctx:Dict)->str:
    msgs=[{"role":"system","content":"You are a senior supply-chain advisor. ≤160 words."},
          {"role":"user","content":json.dumps(ctx,ensure_ascii=False)}]
    r=client.chat.completions.create(model=LLM_MODEL,messages=msgs,
        extra_headers=OPENROUTER_HEADERS,temperature=0.3,max_tokens=700)
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
    st.header("Configuration")
    system=st.selectbox("Manufacturing System",SYSTEMS)
    stage=st.selectbox("LCE Stage",LCE)
    industry=st.selectbox("Industry",["Automotive","Electronics","Medical Devices","Consumer Goods","Other"])
    objective=st.text_input("Objective","Stabilize outsourced supply performance.")
    role=st.selectbox("Your Role",["Engineer","Analyst","Manager","Decision Maker"])
    st.header("5S Priorities")
    weights_5s={s:st.slider(s,0.0,1.0,0.5,0.05) for s in FIVE_S}
    compare_all=st.toggle("Compare all systems",value=False)


# =====================================================
#              MAIN EXECUTION
# =====================================================
st.title("Supply-Chain Strategy Agent (LCE + 5S)")
st.markdown("Developed by: **Dr. J. Isabel Méndez** & **Dr. Arturo Molina**")

if st.button("Analyze",use_container_width=True):
    payload={"objective":objective,"industry":industry,"user_role":role,
             "system_type":system,"lce_stage":stage,"weights_5s":weights_5s}
    with st.spinner("Analyzing through LLM …"):
        t0=time.time()
        try:
            analysis=analyze_to_pillars(payload)
            pillars=normalize_dict(analysis.get("pillars",{p:1/6 for p in PILLARS}))
        except Exception:
            pillars={p:1/6 for p in PILLARS}
        elapsed=time.time()-t0
    # --- dynamic scoring
    scored={
        "core_processes":dynamic_score(BASE_CORE,stage,weights_5s),
        "kpis":dynamic_score(BASE_KPIS,stage,weights_5s),
        "drivers":dynamic_score(BASE_DRIVERS,stage,weights_5s)
    }
    guidance=make_single_guidance(payload)
    st.session_state["results"]={"pillars":pillars,"scored":scored,
                                 "guidance":guidance,"elapsed":elapsed}

# =====================================================
#           DISPLAY RESULTS (CLEAN + FIXED)
# =====================================================
if "results" in st.session_state:
    res = st.session_state["results"]
    st.success(f"LLM completed in {res['elapsed']:.1f}s")

    # --- Pillars
    st.subheader("Pillar Distribution")
    radar_plot(res["pillars"])
    st.caption("Radar chart shows relative strategic weights of pillars according to the LCE + 5S framework.")

    # --- Core Matrices
    for label, key in [
        ("Core Processes", "core_processes"),
        ("KPIs", "kpis"),
        ("Resilience Drivers", "drivers")
    ]:
        st.subheader(f"{label} × System")
        df = pd.DataFrame(res["scored"][key])
        if not compare_all:
            df = df[[system]]
        elif not compare_all and system not in df.columns:
            st.warning(f"⚠️ System '{system}' not found in results; showing all systems instead.")
        dfq = df.applymap(qualitative_label)
        st.dataframe(dfq, use_container_width=True)

        # Contextual captions per matrix
        if key == "core_processes":
            st.markdown("_High = Core Process / Medium = Support Process / Low = Peripheral Process_")
            df_core = dfq.copy()
        elif key == "kpis":
            st.markdown("_High = Critical Performance Metric / Medium = Relevant / Low = Minor Indicator_")
            df_kpi = dfq.copy()
        else:
            st.markdown("_High = Strategic Resilience Factor / Medium = Supportive / Low = Peripheral_")
            df_drv = dfq.copy()

    # --- LLM Guidance
    st.subheader("LLM Guidance Summary")
    st.write(res["guidance"])

    # --- Sensitivity Analysis
    st.subheader("Sensitivity Simulation")
    sensitivity_scenarios(res["scored"]["drivers"])

    # --- Strategic Emphasis Summary
    st.subheader("Strategic Emphasis Summary")
    top = extract_keywords(res["guidance"])
    st.markdown(" | ".join([f"🟢 **{t}**" for t in top]))
    st.caption("Most emphasized strategic concepts in the LLM guidance, aligned with dominant themes from the framework section.")

    # --- Export Option
    st.markdown("### Export Results")
    export_df = pd.concat(
        {
            "Core Processes": df_core,
            "KPIs": df_kpi,
            "Resilience Drivers": df_drv
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

    # --- Analytical Notes
    st.markdown("### Analytical Interpretation Notes")
    st.info(
        "High scores correspond to *core or strategic* importance under the current LCE stage "
        "and 5S weighting. Medium reflects supportive processes contributing to performance "
        "stability but not primary differentiators. Low indicates limited short-term impact or "
        "context-specific functions."
    )


# =====================================================
#                CHAT EXTENSION
# =====================================================
st.markdown("---")
st.subheader("Interactive Advisor")

if "chat" not in st.session_state: st.session_state["chat"]=[]

for m in st.session_state["chat"]:
    with st.chat_message(m["role"]): st.markdown(m["content"])

q=st.chat_input("Ask about trade-offs or improvements …")
if q:
    st.session_state["chat"].append({"role":"user","content":q})
    with st.chat_message("user"): st.markdown(q)
    if "results" not in st.session_state:
        reply="Please run **Analyze** first."
    else:
        ctx={"pillars":res["pillars"],"weights_5s":weights_5s,"guidance":res["guidance"]}
        r=client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role":"system","content":"You are a supply-chain advisor using LCE + 5S logic."},
                      {"role":"user","content":json.dumps(ctx,ensure_ascii=False)},
                      {"role":"user","content":q}],
            extra_headers=OPENROUTER_HEADERS,temperature=0.4,max_tokens=600)
        reply=r.choices[0].message.content
    st.session_state["chat"].append({"role":"assistant","content":reply})
    with st.chat_message("assistant"): st.markdown(reply)



