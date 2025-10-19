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
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
OPENROUTER_HEADERS = {"HTTP-Referer":"http://localhost:8501","X-Title":"LCE+5S Supply-Chain Agent"}

# --- Single Unified Model for ALL tasks ---
LLM_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

# =====================================================
#                CANONICAL VOCAB
# =====================================================
SYSTEMS = ["Product Transfer","Technology Transfer","Facility Design"]
LCE = ["Ideation","Basic Development","Advanced Development","Launch","Operation","End-of-Life"]
FIVE_S = ["Social","Sustainable","Sensing","Smart","Safe"]
PILLARS = ["quality","cost","volume","time","flexibility","environment"]

COMPETITIVE={"Product Transfer":"Operational Excellence","Technology Transfer":"Product Innovation","Facility Design":"Customer Focus/Intimacy"}
VALUE_CHAIN={"Product Transfer":"Collaboration Networks","Technology Transfer":"Strategic Business Units","Facility Design":"Vertical Integration"}
PROD_SERVICE={"Product Transfer":"Assemble to Order (ATO) + Catalogue of Services",
              "Technology Transfer":"Make/Build to Order (MTO/BTO) + Configuration of Services",
              "Facility Design":"Hybrid MTS + BTO + Design of Services"}

# =====================================================
#               BASELINES & TAGS (full)
# =====================================================
BASE_CORE = {
    "Customer Driven Design":{"Product Transfer":1,"Technology Transfer":3,"Facility Design":3},
    "CRM":{"Product Transfer":2,"Technology Transfer":2,"Facility Design":3},
    "Co-Engineering":{"Product Transfer":3,"Technology Transfer":1,"Facility Design":1},
    "SRM":{"Product Transfer":3,"Technology Transfer":3,"Facility Design":1},
    "NPD":{"Product Transfer":1,"Technology Transfer":2,"Facility Design":2},
    "Obtain Customer Commit.":{"Product Transfer":2,"Technology Transfer":2,"Facility Design":2},
    "Order Fulfillment":{"Product Transfer":3,"Technology Transfer":3,"Facility Design":3},
    "Customer Service":{"Product Transfer":1,"Technology Transfer":1,"Facility Design":3},
}
BASE_KPIS = {
    "Supplier on-time delivery":{"Product Transfer":3,"Technology Transfer":0,"Facility Design":0},
    "Incoming defect rate":{"Product Transfer":3,"Technology Transfer":0,"Facility Design":0},
    "Assembly cost per unit":{"Product Transfer":3,"Technology Transfer":0,"Facility Design":0},
    "Logistics lead time":{"Product Transfer":3,"Technology Transfer":0,"Facility Design":0},
    "Ramp-up time":{"Product Transfer":0,"Technology Transfer":3,"Facility Design":0},
    "First-pass yield":{"Product Transfer":0,"Technology Transfer":3,"Facility Design":0},
    "Learning-curve productivity":{"Product Transfer":0,"Technology Transfer":3,"Facility Design":0},
    "% revenue from new products":{"Product Transfer":0,"Technology Transfer":3,"Facility Design":0},
    "OEE":{"Product Transfer":0,"Technology Transfer":0,"Facility Design":3},
    "OTIF":{"Product Transfer":0,"Technology Transfer":0,"Facility Design":3},
    "Lifecycle cost":{"Product Transfer":0,"Technology Transfer":0,"Facility Design":3},
    "ESG index":{"Product Transfer":0,"Technology Transfer":0,"Facility Design":3},
    "Safety incidents":{"Product Transfer":0,"Technology Transfer":0,"Facility Design":3},
}
BASE_DRIVERS = {
    "Inventory/Capacity Buffers":{"Product Transfer":1,"Technology Transfer":3,"Facility Design":3},
    "Network Diversification":{"Product Transfer":3,"Technology Transfer":3,"Facility Design":1},
    "Multisourcing":{"Product Transfer":3,"Technology Transfer":1,"Facility Design":1},
    "Nearshoring":{"Product Transfer":2,"Technology Transfer":2,"Facility Design":2},
    "Platform/Plant Harmonization":{"Product Transfer":1,"Technology Transfer":1,"Facility Design":3},
    "Ecosystem Partnerships":{"Product Transfer":3,"Technology Transfer":3,"Facility Design":1},
}
KPI_TO_PILLARS = {
    "Supplier on-time delivery":{"time":0.45,"flexibility":0.2,"cost":0.2,"quality":0.05},
    "Incoming defect rate":{"quality":0.6,"cost":0.2,"time":0.1},
    "Assembly cost per unit":{"cost":0.7,"time":0.1},
    "Logistics lead time":{"time":0.6,"cost":0.15,"flexibility":0.15},
    "Ramp-up time":{"time":0.55,"flexibility":0.2,"quality":0.1},
    "First-pass yield":{"quality":0.7,"time":0.1},
    "Learning-curve productivity":{"quality":0.3,"time":0.25,"cost":0.2},
    "% revenue from new products":{"flexibility":0.3,"quality":0.2,"volume":0.2},
    "OEE":{"volume":0.35,"quality":0.25,"cost":0.2},
    "OTIF":{"time":0.35,"quality":0.2,"flexibility":0.2,"volume":0.1},
    "Lifecycle cost":{"environment":0.25,"cost":0.5},
    "ESG index":{"environment":0.8},
    "Safety incidents":{"environment":0.2,"quality":0.2},
}
S_TAGS_KPI = {"ESG index":{"Sustainable":1.0},"Lifecycle cost":{"Sustainable":0.6},
              "Safety incidents":{"Safe":1.0},"OEE":{"Smart":0.4,"Sensing":0.4},
              "First-pass yield":{"Smart":0.4,"Sensing":0.4},"Supplier on-time delivery":{"Social":0.6},
              "Logistics lead time":{"Social":0.6}}
S_TAGS_CORE = {"Co-Engineering":{"Social":0.6,"Smart":0.3},"SRM":{"Social":1.0},
               "Order Fulfillment":{"Social":0.8,"Smart":0.2},"Customer Service":{"Social":1.0},
               "Customer Driven Design":{"Smart":0.4,"Social":0.4},"NPD":{"Smart":0.6}}
S_TAGS_DRIVERS = {"Nearshoring":{"Sustainable":0.6,"Safe":0.2},
                  "Platform/Plant Harmonization":{"Smart":0.5},
                  "Ecosystem Partnerships":{"Social":0.6}}

STAGE_TAGS_KPI = {"Ramp-up time":{"Launch":1.0,"Advanced Development":0.6},
                  "First-pass yield":{"Launch":0.7,"Operation":0.5},
                  "Learning-curve productivity":{"Launch":0.6,"Operation":0.4},
                  "OEE":{"Operation":1.0},"OTIF":{"Operation":0.7},
                  "Lifecycle cost":{"End-of-Life":0.6,"Operation":0.4},
                  "ESG index":{"End-of-Life":0.9,"Operation":0.5}}
STAGE_TAGS_CORE = {"Co-Engineering":{"Ideation":0.8,"Basic Development":0.6},
                   "NPD":{"Advanced Development":0.8,"Launch":0.5},
                   "Order Fulfillment":{"Operation":0.9},
                   "Customer Service":{"Operation":0.7},"SRM":{"Operation":0.6}}
STAGE_TAGS_DRIVERS = {"Inventory/Capacity Buffers":{"Operation":0.7},
                      "Platform/Plant Harmonization":{"Operation":0.7,"Launch":0.4},
                      "Nearshoring":{"Operation":0.4,"End-of-Life":0.4}}

# =====================================================
#                  CORE SCORING FUNCTIONS
# =====================================================
def clamp01(x): return max(0.0,min(1.0,x))
def clamp03(x): return max(0.0,min(3.0,x))

def scenario_memberships(custom_tags: List[str]):
    tags=[t.strip().lower() for t in custom_tags if t.strip()]
    mem={k:0.0 for k in list(BASE_DRIVERS.keys())+list(BASE_KPIS.keys())+list(BASE_CORE.keys())}
    def add(k,v): mem[k]=max(mem[k],v)
    for t in tags:
        if "port" in t or "congestion" in t:
            add("Nearshoring",0.7); add("Network Diversification",0.5)
        if "bankruptcy" in t or "single-source" in t:
            add("Multisourcing",0.8)
        if "digital twin" in t:
            add("Platform/Plant Harmonization",0.6)
    return mem

def s_boost(w,s_tags,name): return sum(w.get(k,0.0)*v for k,v in s_tags.get(name,{}).items())
def stage_boost(stage, tags, name, max_gain=0.8): return clamp01(tags.get(name,{}).get(stage,0.0))*max_gain
def scenario_boost(flags, mem, name, matrix):
    b=0.0
    if flags.get("volatility") and matrix=="drivers" and name in ["Inventory/Capacity Buffers","Platform/Plant Harmonization"]: b+=0.5
    if flags.get("geo_risk") and matrix=="drivers" and name in ["Multisourcing","Network Diversification","Nearshoring"]: b+=0.5
    if flags.get("carbon"):
        if matrix=="drivers" and name in ["Nearshoring","Ecosystem Partnerships"]: b+=0.4
        if matrix=="kpis" and name in ["ESG index","Lifecycle cost"]: b+=0.3
    return b + mem.get(name,0.0)
def pillar_boost(pillars,item_pillars,max_gain=1.2): return max_gain*sum(pillars.get(k,0.0)*v for k,v in item_pillars.items())

def score_matrix(base_map, matrix, w5s, stage, flags, mem, pillars):
    out={}
    for item, cols in base_map.items():
        out[item]={}
        for system, base in cols.items():
            score=float(base)
            if matrix=="kpis":
                score += pillar_boost(pillars, KPI_TO_PILLARS.get(item,{}), 1.2)
                score += stage_boost(stage, STAGE_TAGS_KPI, item, 0.8)
                score += clamp01(s_boost(w5s, S_TAGS_KPI, item))*0.8
                score += scenario_boost(flags, mem, item, matrix)
            elif matrix=="core_processes":
                score += stage_boost(stage, STAGE_TAGS_CORE, item, 0.8)
                score += clamp01(s_boost(w5s, S_TAGS_CORE, item))*0.8
                score += scenario_boost(flags, mem, item, matrix)*0.6
            else:
                score += stage_boost(stage, STAGE_TAGS_DRIVERS, item, 0.6)
                score += clamp01(s_boost(w5s, S_TAGS_DRIVERS, item))*0.6
                score += scenario_boost(flags, mem, item, matrix)
            out[item][system]=clamp03(score)
    return out
def score_all(w5s, stage, flags, custom_tags, pillars):
    mem=scenario_memberships(custom_tags)
    return {"core_processes":score_matrix(BASE_CORE,"core_processes",w5s,stage,flags,mem,pillars),
            "kpis":score_matrix(BASE_KPIS,"kpis",w5s,stage,flags,mem,pillars),
            "drivers":score_matrix(BASE_DRIVERS,"drivers",w5s,stage,flags,mem,pillars)}

# =====================================================
#                 LLM INTERFACES
# =====================================================
ANALYZER_SYSTEM = (
    "You are an experienced Supply Chain Director. Output ONLY JSON.\n"
    "Input: objective, industry, user_role, system_type, lce_stage, 5S weights, scenarios.\n"
    "Return: {pillars:{quality,cost,volume,time,flexibility,environment∈[0,1]},reasons:[…],tags:[…]}."
)
def analyze_to_pillars(payload: Dict) -> Dict:
    msgs=[{"role":"system","content":ANALYZER_SYSTEM},
          {"role":"user","content":json.dumps(payload, ensure_ascii=False)}]
    r=client.chat.completions.create(model=LLM_MODEL, messages=msgs,
        extra_headers=OPENROUTER_HEADERS,
        response_format={"type":"json_object"}, temperature=0.2, max_tokens=700)
    return json.loads(r.choices[0].message.content)

SINGLE_GUIDANCE_SYSTEM = (
    "You are a senior supply-chain advisor. Speak directly to the user in ≤160 words."
)
def make_single_guidance(ctx: Dict) -> str:
    r=client.chat.completions.create(model=LLM_MODEL,
        messages=[{"role":"system","content":SINGLE_GUIDANCE_SYSTEM},
                  {"role":"user","content":json.dumps(ctx, ensure_ascii=False)}],
        extra_headers=OPENROUTER_HEADERS, temperature=0.3, max_tokens=700)
    return r.choices[0].message.content

# =====================================================
#                PERFORMANCE & ANALYTICS
# =====================================================
def extract_keywords(text, topn=10):
    text=re.sub(r'[^A-Za-z\s]', '', text)
    vec=CountVectorizer(stop_words='english')
    X=vec.fit_transform([text.lower()])
    freqs=zip(vec.get_feature_names_out(), X.toarray().flatten())
    return [w for w,_ in sorted(freqs,key=lambda x:-x[1])[:topn]]

def radar_plot(pillars_dict):
    df=pd.DataFrame(list(pillars_dict.items()),columns=["Pillar","Weight"])
    fig=px.line_polar(df,r="Weight",theta="Pillar",line_close=True)
    fig.update_traces(fill='toself')
    st.plotly_chart(fig,use_container_width=True)

def synthetic_stress(weights_5s,lce_stage,custom_tags,pillars):
    data=[]
    combos=[{"volatility":v,"geo_risk":g,"carbon":c}
            for v in [True,False] for g in [True,False] for c in [True,False]]
    for f in combos:
        scores=score_all(weights_5s,lce_stage,f,custom_tags,pillars)
        avg_drv=np.mean([np.mean(list(v.values())) for v in scores["drivers"].values()])
        data.append({"Volatility":f["volatility"],"GeoRisk":f["geo_risk"],
                     "Carbon":f["carbon"],"AvgDriverScore":avg_drv})
    df=pd.DataFrame(data)
    fig=px.bar(df,x=["Volatility","GeoRisk","Carbon"],y="AvgDriverScore",
               title="Resilience Sensitivity under Disruption Scenarios")
    st.plotly_chart(fig,use_container_width=True)
    return df

def llm_variability(payload,runs=3):
    results=[]
    for _ in range(runs):
        try: results.append(analyze_to_pillars(payload)["pillars"])
        except: continue
    if len(results)<2:return None
    return pd.DataFrame(results).std().mean()

# =====================================================
#                SIDEBAR CONFIGURATION
# =====================================================
PRESETS={"IECOS/PRIMA (PT)":{"system":"Product Transfer","preset":["High demand volatility","Geopolitical risk"],
                              "custom":["lean logistics","digital platforms","supplier collaboration"],
                              "objective":"Stabilize outsourced supply and assemble-to-order delivery performance."},
         "Cardanes (TT)":{"system":"Technology Transfer","preset":["High demand volatility"],
                          "custom":["workforce training","process innovation","iot monitoring","cost of adoption"],
                          "objective":"Adopt a new process technology and ramp product variants."},
         "MicroMachines (FD)":{"system":"Facility Design","preset":["Carbon constraints","Geopolitical risk"],
                               "custom":["digital twin","esg-driven operations","vertical integration"],
                               "objective":"Design and ramp a vertically integrated facility with ESG visibility."},
         "Other":{"system":None,"preset":[],"custom":[],"objective":""}}

with st.sidebar:
    st.header("Template & Context")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), key="preset")
    preset = PRESETS[preset_name]
    st.radio("Manufacturing system (view)", SYSTEMS,
             index=SYSTEMS.index(preset["system"]) if preset["system"] in SYSTEMS else 0,
             key="selected_system")
    st.text_input("Objective", value=preset["objective"] or "Design and ramp a flexible small manufacturing cell.", key="objective")
    st.selectbox("Industry", ["Automotive","Electronics","Medical Devices","Consumer Goods","Other"], index=1, key="industry")
    roles=["Design Engineer","Process Engineer","Manufacturing Engineer","Safety Supervisor","Sustainability Manager","Supply Chain Analyst","Manager/Decision Maker","Other"]
    st.selectbox("Your role", roles, index=5, key="user_role")
    if st.session_state.get("user_role")=="Other":
        st.text_input("Specify role", value="Other", key="user_role_other")
    st.header("LCE & 5S Priorities")
    st.selectbox("LCE stage", LCE, key="lce_stage")
    st.caption("5S sliders are priorities: 0 = deprioritize, 0.5 = neutral, 1 = strongly prioritize.")
    for s in FIVE_S:
        st.slider(s, 0.0, 1.0, 0.5, 0.05, key=f"s5_{s}")
    st.header("Scenarios")
    st.multiselect("Preset scenarios",
        ["High demand volatility","Geopolitical risk","Carbon constraints"],
        default=preset["preset"], key="preset_scenarios")
    st.text_input("Additional tags (comma-separated)",
        value=", ".join(preset["custom"]) if preset_name!="Other" else "",
        key="custom_tags_text")
    st.toggle("Compare all systems (view)", value=False, key="compare_all")

# =====================================================
#                MAIN APP LOGIC
# =====================================================
st.title("Supply-Chain Strategy Agent")
st.markdown("Developed by: **Dr. J. Isabel Méndez** & **Dr. Arturo Molina**")

if st.button("Analyze", use_container_width=True):
    role_val = st.session_state.get("user_role_other") if st.session_state.get("user_role")=="Other" else st.session_state.get("user_role")
    weights_5s = {s: st.session_state.get(f"s5_{s}", 0.5) for s in FIVE_S}
    flags={"volatility":"High demand volatility" in st.session_state.get("preset_scenarios",[]),
           "geo_risk":"Geopolitical risk" in st.session_state.get("preset_scenarios",[]),
           "carbon":"Carbon constraints" in st.session_state.get("preset_scenarios",[])}
    custom=[t.strip() for t in st.session_state.get("custom_tags_text","").split(",") if t.strip()]

    payload = {
        "objective": st.session_state.get("objective",""),
        "industry": st.session_state.get("industry",""),
        "user_role": role_val,
        "system_type": st.session_state.get("selected_system","Product Transfer"),
        "lce_stage": st.session_state.get("lce_stage","Operation"),
        "weights_5s": weights_5s,
        "scenarios": {"preset": st.session_state.get("preset_scenarios",[]),"custom": custom}
    }

    with st.spinner("Running LLM analysis..."):
        t0=time.time()
        try:
            analysis=analyze_to_pillars(payload)
            pillars=analysis.get("pillars",{})
            reasons=analysis.get("reasons",[])
            tags=analysis.get("tags",[])
        except Exception as e:
            st.warning(f"Analyzer failed: {e}")
            pillars={k:1/len(PILLARS) for k in PILLARS}; reasons=["Equal weights used"]; tags=[]
        elapsed=time.time()-t0

    ssum=sum(pillars.values()) or 1
    pillars={k:v/ssum for k,v in pillars.items()}

    # run deterministic scoring
    scored=score_all(weights_5s, st.session_state.get("lce_stage","Operation"), flags, custom+tags, pillars)

    # single guidance per system
    guidance_single={}
    for sys in SYSTEMS:
        ctx={
            "objective": st.session_state.get("objective",""),
            "industry": st.session_state.get("industry",""),
            "user_role": role_val,
            "system_type": sys,
            "lce_stage": st.session_state.get("lce_stage","Operation"),
            "pillar_weights": pillars,
            "weights_5s": weights_5s,
            "scenarios": {"preset": st.session_state.get("preset_scenarios",[]),"custom": custom+tags},
            "strategies": {"Competitive":COMPETITIVE[sys],"Value Chain":VALUE_CHAIN[sys],"Product/Service":PROD_SERVICE[sys]},
            "selected_scores": {k:{i:float(v[sys]) for i,v in scored[k].items()} for k in scored}
        }
        guidance_single[sys]=make_single_guidance(ctx)

    st.session_state["results"]={
        "pillars":pillars,"scored":scored,"weights_5s":weights_5s,"flags":flags,
        "reasons":reasons,"guidance_single":guidance_single,"elapsed":elapsed
    }

# =====================================================
#            RESULTS RENDERING SECTION
# =====================================================
if "results" in st.session_state:
    res=st.session_state["results"]
    st.success(f"LLM completed in {res['elapsed']:.1f} s")

    radar_plot(res["pillars"])

    st.markdown("### Core Processes × System")
    st.dataframe(pd.DataFrame(res["scored"]["core_processes"]).style.background_gradient(cmap="YlGn"))

    st.markdown("### KPIs × System")
    st.dataframe(pd.DataFrame(res["scored"]["kpis"]).style.background_gradient(cmap="YlOrBr"))

    st.markdown("### Resilience Drivers × System")
    st.dataframe(pd.DataFrame(res["scored"]["drivers"]).style.background_gradient(cmap="PuBu"))

    if res.get("reasons"):
        st.markdown("**Pillar rationale:**")
        st.write("\n".join(f"- {r}" for r in res["reasons"]))

    st.markdown("### Guidance")
    sel_sys=st.session_state.get("selected_system","Product Transfer")
    st.write(res["guidance_single"].get(sel_sys,"No guidance available."))

    st.markdown("### Sensitivity Simulation")
    df_synth=synthetic_stress(res["weights_5s"], st.session_state.get("lce_stage","Operation"),
                              [], res["pillars"])
    st.caption("Synthetic stress test — average driver scores under combined volatility/geopolitical/carbon constraints.")

    st.markdown("### Keyword Extraction from Guidance")
    all_guidance=" ".join(res["guidance_single"].values())
    top_words=extract_keywords(all_guidance,topn=10)
    st.write("**Top terms:**", ", ".join(top_words))

    # Export CSV option
    if st.download_button("Download Results CSV", df_synth.to_csv(index=False).encode(), "results.csv", "text/csv"):
        st.success("Exported!")

    # Variability check
    st.markdown("### Model Stability Check")
    var_score=llm_variability({
        "objective": st.session_state.get("objective",""),
        "industry": st.session_state.get("industry",""),
        "user_role": st.session_state.get("user_role",""),
        "system_type": st.session_state.get("selected_system","Product Transfer"),
        "lce_stage": st.session_state.get("lce_stage","Operation"),
        "weights_5s": res["weights_5s"],
        "scenarios": {"preset":[], "custom":[]}
    })
    if var_score:
        st.metric("Average Std. Deviation Across Pillars", f"{var_score:.3f}")
    else:
        st.info("Variability check not available (single run only).")

# =====================================================
#                CHAT SECTION
# =====================================================
st.markdown("---")
st.subheader("Chat with the Strategy Agent")

if "chat" not in st.session_state:
    st.session_state["chat"]=[]

for m in st.session_state["chat"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask about trade-offs or recommendations…")
if user_q:
    st.session_state["chat"].append({"role":"user","content":user_q})
    with st.chat_message("user"): st.markdown(user_q)
    if "results" not in st.session_state:
        reply="Please run **Analyze** first."
    else:
        res=st.session_state["results"]
        ctx={"pillars":res["pillars"],"weights_5s":res["weights_5s"],"flags":res["flags"],
             "scores":res["scored"]}
        r=client.chat.completions.create(model=LLM_MODEL,
            messages=[{"role":"system","content":"You are a supply-chain advisor."},
                      {"role":"user","content":json.dumps(ctx)},
                      {"role":"user","content":user_q}],
            extra_headers=OPENROUTER_HEADERS,temperature=0.4,max_tokens=600)
        reply=r.choices[0].message.content
    st.session_state["chat"].append({"role":"assistant","content":reply})
    with st.chat_message("assistant"): st.markdown(reply)
