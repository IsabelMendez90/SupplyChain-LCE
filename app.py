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
LLM_MODEL = "mistralai/mistral-7b-instruct:free"

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



def s_boost(w,s_tags,name): return sum(w.get(k,0.0)*v for k,v in s_tags.get(name,{}).items())
def stage_boost(stage, tags, name, max_gain=0.8): return clamp01(tags.get(name,{}).get(stage,0.0))*max_gain

def pillar_boost(pillars, item_pillars, max_gain=1.2):
    return max_gain * sum(pillars.get(k, 0.0) * v for k, v in item_pillars.items())

def score_matrix(base_map, matrix, w5s, stage, pillars):
    out = {}
    for item, cols in base_map.items():
        out[item] = {}
        for system, base in cols.items():
            score = float(base)

            # --- KPI Matrix ---
            if matrix == "kpis":
                score += pillar_boost(pillars, KPI_TO_PILLARS.get(item, {}), 1.2)
                score += stage_boost(stage, STAGE_TAGS_KPI, item, 0.8)
                score += clamp01(s_boost(w5s, S_TAGS_KPI, item)) * 0.8

            # --- Core Processes Matrix ---
            elif matrix == "core_processes":
                score += stage_boost(stage, STAGE_TAGS_CORE, item, 0.8)
                score += clamp01(s_boost(w5s, S_TAGS_CORE, item)) * 0.8

            # --- Resilience Drivers Matrix ---
            else:
                score += stage_boost(stage, STAGE_TAGS_DRIVERS, item, 0.6)
                score += clamp01(s_boost(w5s, S_TAGS_DRIVERS, item)) * 0.6

            out[item][system] = clamp03(score)
    return out


def score_all(w5s, stage, pillars):
    return {
        "core_processes": score_matrix(BASE_CORE, "core_processes", w5s, stage, pillars),
        "kpis": score_matrix(BASE_KPIS, "kpis", w5s, stage, pillars),
        "drivers": score_matrix(BASE_DRIVERS, "drivers", w5s, stage, pillars),
    }


# =====================================================
#                 LLM INTERFACES
# =====================================================
ANALYZER_SYSTEM = (
    "You are an experienced Supply Chain Director. Output ONLY JSON.\n"
    "Input: objective, industry, user_role, system_type, lce_stage, 5S weights.\n"
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
               title="Resilience Sensitivity")
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
PRESETS={"IECOS (PT)":{"system":"Product Transfer","preset":["High demand volatility","Geopolitical risk"],
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
    st.toggle("Compare all systems (view)", value=False, key="compare_all")

# =====================================================
#                MAIN APP LOGIC
# =====================================================
st.title("Supply-Chain Strategy Agent")
st.markdown("Developed by: **Dr. J. Isabel Méndez** & **Dr. Arturo Molina**")

if st.button("Analyze", use_container_width=True):
    role_val = st.session_state.get("user_role_other") if st.session_state.get("user_role")=="Other" else st.session_state.get("user_role")
    weights_5s = {s: st.session_state.get(f"s5_{s}", 0.5) for s in FIVE_S}

    payload = {
        "objective": st.session_state.get("objective",""),
        "industry": st.session_state.get("industry",""),
        "user_role": role_val,
        "system_type": st.session_state.get("selected_system","Product Transfer"),
        "lce_stage": st.session_state.get("lce_stage","Operation")
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
    scored=score_all(weights_5s, st.session_state.get("lce_stage","Operation"), pillars)

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
            "strategies": {"Competitive":COMPETITIVE[sys],"Value Chain":VALUE_CHAIN[sys],"Product/Service":PROD_SERVICE[sys]},
            "selected_scores": {k:{i:float(v[sys]) for i,v in scored[k].items()} for k in scored}
        }
        guidance_single[sys]=make_single_guidance(ctx)

    st.session_state["results"]={
        "pillars":pillars,"scored":scored,"weights_5s":weights_5s,
        "reasons":reasons,"guidance_single":guidance_single,"elapsed":elapsed
    }


# =====================================================
#     SAFE LLM CALL WRAPPER (handles null responses)
# =====================================================
def safe_llm_call(prompt: str, payload: dict, temp=0.35, max_toks=400):
    """Safer call for OpenRouter free models that occasionally drop outputs."""
    try:
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            extra_headers=OPENROUTER_HEADERS,
            temperature=temp,
            max_tokens=max_toks,
        )
        out = r.choices[0].message.content
        if not out or len(out.strip()) == 0:
            raise ValueError("Empty LLM response")
        return out
    except Exception as e:
        st.warning(f"⚠️ LLM call failed: {e}")
        return ""

# =====================================================
#            RESULTS RENDERING SECTION (IMPROVED)
# =====================================================
if "results" in st.session_state:
    res = st.session_state["results"]
    st.success(f"LLM completed in {res['elapsed']:.1f} s")

    # --- RADAR ---
    radar_plot(res["pillars"])
    st.info("""
    **About this Radar:**  
    This chart shows proportional emphasis (summing to 100%).
    It visualizes how the LLM Agent allocates your strategic focus
    across the pillars Quality, Cost, Volume, Time, Flexibility, and Environment
    based on your role, and industry.
    """)

    # --- Context data ---
    sel_sys = st.session_state.get("selected_system", "Product Transfer")
    role = st.session_state.get("user_role", "")
    industry = st.session_state.get("industry", "")
    objective = st.session_state.get("objective", "")

    context_payload = {
        "system_type": sel_sys,
        "user_role": role,
        "industry": industry,
        "objective": objective,
        "pillars": res["pillars"],
    }

    # =====================================================
    #   RADAR INTERPRETATION (improved contextual prompt)
    # =====================================================
    if "llm_explanations" not in st.session_state:
        st.session_state["llm_explanations"] = {}

    def clean_numbers(text: str) -> str:
        """Remove (3.0)-style numeric parenthesis."""
        return re.sub(r"\s*\(\d+(\.\d+)?\)", "", text)

    prompt_radar = f"""
    You are a senior supply-chain consultant addressing a {role} in the {industry} industry.
    Interpret the radar chart representing the pillar distribution for the {sel_sys} system.
    Each axis (Quality, Cost, Volume, Time, Flexibility, Environment) reflects proportional emphasis that sums to 100%.
    Explain which two or three pillars show the strongest strategic weight and why they align with the user's stated objective:
    "{objective}".
    Also mention which dimensions are comparatively weaker and how that could affect balance, risk, or sustainability.
    Avoid numeric details or parenthetical notation.
    Speak in a professional but concise tone (≤180 words).
    """

    if "radar" not in st.session_state["llm_explanations"]:
        try:
            radar_expl = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": prompt_radar},
                    {"role": "user", "content": json.dumps(context_payload, ensure_ascii=False)}
                ],
                extra_headers=OPENROUTER_HEADERS,
                temperature=0.35,
                max_tokens=400
            ).choices[0].message.content
            st.session_state["llm_explanations"]["radar"] = radar_expl
        except Exception as e:
            st.warning(f"Could not generate radar interpretation: {e}")
            radar_expl = ""
    else:
        radar_expl = st.session_state["llm_explanations"]["radar"]

    if radar_expl:
        st.markdown("### Strategic Interpretation")
        st.write(clean_numbers(radar_expl))

    # --- Helper for labels ---
    def show_matrix(title, df_dict):
        st.markdown(f"### {title}")
        df = pd.DataFrame(df_dict).T
        compare_all = st.session_state.get("compare_all", False)
        selected = st.session_state.get("selected_system", "Product Transfer")
        if isinstance(selected, (tuple, list)): selected = selected[0]
        if isinstance(selected, dict): selected = next(iter(selected.values()))
        if not compare_all:
            if selected in df.columns:
                df = df[[selected]]
            else:
                st.warning(f"⚠️ The selected system '{selected}' is not available; showing all instead.")
        df_label = df.applymap(lambda x: "Low" if x < 1 else "Medium" if x < 2 else "High")
        color_map = {"Low": "#f8d7da", "Medium": "#fff3cd", "High": "#d4edda"}
        styled = df_label.style.applymap(
            lambda v: f"background-color: {color_map[v]}; color:black; text-align:center; font-weight:bold;"
        )
        st.dataframe(styled, use_container_width=True)
        return df_label

    # =====================================================
    #  CORE PROCESSES (Label-Based Interpretation)
    # =====================================================
    df_core = show_matrix("Core Processes × System", res["scored"]["core_processes"])
    st.caption("Core Processes show where structural strengths are most evident. High = critical to maturity level.")
    
    sel_sys = st.session_state.get("selected_system", "Product Transfer")
    core_scores = {k: float(v.get(sel_sys, 0)) for k, v in res["scored"]["core_processes"].items()}
    core_labels = {k: ("High" if v >= 2 else "Medium" if v >= 1 else "Low") for k, v in core_scores.items()}
    
    core_payload = context_payload.copy()
    core_payload.update({"core_labels": core_labels})
    
    prompt_core = f"""
    You are a supply-chain strategist advising a {role} in the {industry} industry.
    Below is the qualitative status of each core process for the {sel_sys} system:
    {json.dumps(core_labels, indent=2)}
    Interpret these labels as *priority levels*, not performance results.
    Focus on which High-priority processes should remain central to achieving "{objective}" ,
    and which Medium ones should be **raised** to High to reinforce system maturity.
    If no Low processes exist, skip that discussion entirely.
    Avoid re-labelling or contradicting the given levels — refer to them exactly as stated.
    End with one prescriptive recommendation on how this role should rebalance focus.
    ≤170 words, directive and analytical tone.
    """
    
    if "core" not in st.session_state["llm_explanations"]:
        try:
            core_expl = safe_llm_call(prompt_core, core_payload)
        except Exception as e:
            st.warning(f"Core analysis failed: {e}")
            core_expl = ""
    else:
        core_expl = safe_llm_call(prompt_core, core_payload)
    
    if core_expl:
        st.markdown("**Interpretation:**")
        st.write(clean_numbers(core_expl))

    # =====================================================
    #  KPIs × System (Label-Based Interpretation)
    # =====================================================
    df_kpi = show_matrix("KPIs × System", res["scored"]["kpis"])
    st.caption("KPIs summarize efficiency, productivity, and cost. High = leverage points; Low = development priorities.")
    
    sel_sys = st.session_state.get("selected_system", "Product Transfer")
    kpi_scores = {k: float(v.get(sel_sys, 0)) for k, v in res["scored"]["kpis"].items()}
    kpi_labels = {k: ("High" if v >= 2 else "Medium" if v >= 1 else "Low") for k, v in kpi_scores.items()}
    
    kpi_payload = context_payload.copy()
    kpi_payload.update({"kpi_labels": kpi_labels})
    
    prompt_kpi = f"""
    You are a performance strategist advising a {role} in the {industry} sector.
    Below is the qualitative status of each KPI for the {sel_sys} system:
    {json.dumps(kpi_labels, indent=2)}
    Interpret these labels as *priority signals*, not results.
    Focus on which High-priority KPIs are essential to achieving "{objective}" ,
    and which Medium or Low KPIs should be reinforced to close operational or sustainability gaps.
    Do not contradict the given labels — refer to them exactly as stated.
    Provide role-oriented guidance on how to leverage High KPIs and develop weaker ones.
    Conclude with one prescriptive insight summarizing how KPI focus can strengthen system performance.
    ≤170 words, professional and directive tone.
    """
    
    if "kpi" not in st.session_state["llm_explanations"]:
        try:
            kpi_expl = safe_llm_call(prompt_kpi, kpi_payload)
        except Exception as e:
            st.warning(f"KPI analysis failed: {e}")
            kpi_expl = ""
    else:
        kpi_expl = safe_llm_call(prompt_kpi, kpi_payload)
    
    if kpi_expl:
        st.markdown("**Interpretation:**")
        st.write(clean_numbers(kpi_expl))
    else:
        st.warning("⚠️ KPI interpretation returned no content.")

    # =====================================================
    #  RESILIENCE DRIVERS × System (Label-Based Interpretation)
    # =====================================================
    df_drv = show_matrix("Resilience Drivers × System", res["scored"]["drivers"])
    st.caption("Resilience Drivers represent adaptability to disruption, sustainability, and ecosystem interdependence.")
    
    sel_sys = st.session_state.get("selected_system", "Product Transfer")
    driver_scores = {k: float(v.get(sel_sys, 0)) for k, v in res["scored"]["drivers"].items()}
    driver_labels = {k: ("High" if v >= 2 else "Medium" if v >= 1 else "Low") for k, v in driver_scores.items()}
    
    driver_payload = context_payload.copy()
    driver_payload.update({"driver_labels": driver_labels})
    
    prompt_drv = f"""
    You are a resilience strategist advising a {role} in the {industry} industry.
    Below is the qualitative status of each resilience driver for the {sel_sys} system:
    {json.dumps(driver_labels, indent=2)}
    Interpret these labels as *priority guidance*—which elements to reinforce or elevate.
    Focus on which High drivers are already central to system stability and which Medium ones should be developed
    to strengthen adaptability.
    If Low drivers appear, discuss how this role can address them through design, sourcing, or collaboration.
    Never contradict or reinterpret the given levels—use them exactly as presented.
    Conclude with one actionable recommendation on how to improve systemic resilience for "{objective}".
    ≤170 words, prescriptive and analytical tone.
    """
    
    if "driver" not in st.session_state["llm_explanations"]:
        try:
            driver_expl = safe_llm_call(prompt_drv, driver_payload)
        except Exception as e:
            st.warning(f"Driver analysis failed: {e}")
            driver_expl = ""
    else:
        driver_expl = safe_llm_call(prompt_drv, driver_payload)
    
    if driver_expl:
        st.markdown("**Interpretation:**")
        st.write(clean_numbers(driver_expl))
    else:
        st.warning("⚠️ Driver interpretation returned no content.")


    # =====================================================
    #  COMPARATIVE INTERPRETATION (when toggle active)
    # =====================================================
    if st.session_state.get("compare_all", False):
        if "compare" not in st.session_state["llm_explanations"]:
            try:
                compare_payload = {
                    "selected_system": sel_sys,
                    "objective": objective,
                    "role": role,
                    "industry": industry,
                    "core": res["scored"]["core_processes"],
                    "kpis": res["scored"]["kpis"],
                    "drivers": res["scored"]["drivers"],
                }
                compare_prompt = f"""
                You are an expert supply-chain strategist. The user focuses on {sel_sys} with objective "{objective}" in {industry}.
                Compare this system against Technology Transfer and Facility Design across Core Processes, KPIs, and Drivers.
                Highlight how strengths and weaknesses differ, and how they align with {role}'s perspective.
                Conclude with one practical insight on when a hybrid or complementary strategy could be beneficial.
                Do not include numeric values, parentheses or the amount of words.
                """
                compare_expl = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": compare_prompt},
                        {"role": "user", "content": json.dumps(compare_payload, ensure_ascii=False)},
                    ],
                    extra_headers=OPENROUTER_HEADERS,
                    temperature=0.35,
                    max_tokens=450
                ).choices[0].message.content
                st.session_state["llm_explanations"]["compare"] = compare_expl
            except Exception as e:
                st.warning(f"Comparative explanation failed: {e}")
                compare_expl = ""
        else:
            compare_expl = st.session_state["llm_explanations"]["compare"]

        if compare_expl:
            st.markdown("### Comparative Interpretation")
            st.write(clean_numbers(compare_expl))


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
        ctx={"pillars":res["pillars"],"weights_5s":res["weights_5s"],
             "scores":res["scored"]}
        r=client.chat.completions.create(model=LLM_MODEL,
            messages=[{"role":"system","content":"You are a supply-chain advisor."},
                      {"role":"user","content":json.dumps(ctx)},
                      {"role":"user","content":user_q}],
            extra_headers=OPENROUTER_HEADERS,temperature=0.4,max_tokens=600)
        reply=r.choices[0].message.content
    st.session_state["chat"].append({"role":"assistant","content":reply})
    with st.chat_message("assistant"): st.markdown(reply)













































