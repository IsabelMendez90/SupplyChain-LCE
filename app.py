# Authors: Dr. J. Isabel Méndez & Dr. Arturo Molina

import os, json, time
from typing import Dict, List
import pandas as pd
import streamlit as st
from openai import OpenAI

# -------------------- Setup --------------------
st.set_page_config(page_title="Supply-Chain Agent (LCE+5S)", layout="wide")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key= st.secrets["OPENROUTER_API_KEY"])
OPENROUTER_HEADERS = {"HTTP-Referer":"http://localhost:8501","X-Title":"LCE+5S Supply-Chain Agent"}

# Hidden models (internal policy)
ANALYZER_CHAIN = [
    "deepseek/deepseek-r1-0528:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1-0528:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]
NARRATIVE_PRIMARY = "openai/gpt-oss-20b:free"
NARRATIVE_FALLBACK = "meta-llama/llama-3.3-70b-instruct:free"
CHAT_DEFAULT = "openai/gpt-oss-20b:free"
CHAT_CHOICES = ["openai/gpt-oss-20b:free","deepseek/deepseek-r1-0528:free","meta-llama/llama-3.3-70b-instruct:free"]

# -------------------- Canonical vocab --------------------
SYSTEMS = ["Product Transfer","Technology Transfer","Facility Design"]
LCE = ["Ideation","Basic Development","Advanced Development","Launch","Operation","End-of-Life"]
FIVE_S = ["Social","Sustainable","Sensing","Smart","Safe"]
PILLARS = ["quality","cost","volume","time","flexibility","environment"]

# -------------------- Strategies by system --------------------
COMPETITIVE={"Product Transfer":"Operational Excellence","Technology Transfer":"Product Innovation","Facility Design":"Customer Focus/Intimacy"}
VALUE_CHAIN={"Product Transfer":"Collaboration Networks","Technology Transfer":"Strategic Business Units","Facility Design":"Vertical Integration"}
PROD_SERVICE={"Product Transfer":"Assemble to Order (ATO) + Catalogue of Services",
              "Technology Transfer":"Make/Build to Order (MTO/BTO) + Configuration of Services",
              "Facility Design":"Hybrid MTS + BTO + Design of Services"}

# -------------------- Baselines (0–3) --------------------
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

# -------------------- KPI → pillar memberships --------------------
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

# -------------------- 5S/Stage tags --------------------
S_TAGS_KPI = {"ESG index":{"Sustainable":1.0},"Lifecycle cost":{"Sustainable":0.6},
              "Safety incidents":{"Safe":1.0},"OEE":{"Smart":0.4,"Sensing":0.4},
              "First-pass yield":{"Smart":0.4,"Sensing":0.4},"Supplier on-time delivery":{"Social":0.6},
              "Logistics lead time":{"Social":0.6}}
S_TAGS_CORE = {"Co-Engineering":{"Social":0.6,"Smart":0.3},"SRM":{"Social":1.0},
               "Order Fulfillment":{"Social":0.8,"Smart":0.2},"Customer Service":{"Social":1.0},
               "Customer Driven Design":{"Smart":0.4,"Social":0.4},"NPD":{"Smart":0.6}}
S_TAGS_DRIVERS = {"Nearshoring":{"Sustainable":0.6,"Safe":0.2},"Platform/Plant Harmonization":{"Smart":0.5},
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

# -------------------- Scenario mapping --------------------
def scenario_memberships(custom_tags: List[str]):
    tags=[t.strip().lower() for t in custom_tags if t.strip()]
    mem={k:0.0 for k in ["OTIF","Logistics lead time","Supplier on-time delivery","OEE","First-pass yield","Lifecycle cost","ESG index",
                          "Nearshoring","Network Diversification","Multisourcing","Inventory/Capacity Buffers",
                          "Platform/Plant Harmonization","Ecosystem Partnerships","SRM","Order Fulfillment","Co-Engineering"]}
    def add(k,v): mem[k] = max(mem[k], v)
    for t in tags:
        if "port" in t or "congestion" in t:
            add("Nearshoring",0.7); add("Network Diversification",0.5)
            add("OTIF",0.6); add("Logistics lead time",0.7); add("Order Fulfillment",0.4)
        if "bankruptcy" in t or "single-source" in t or "supplier failure" in t:
            add("Multisourcing",0.8); add("Ecosystem Partnerships",0.6); add("SRM",0.5)
            add("Supplier on-time delivery",0.6)
        if "tariff" in t or "trade" in t:
            add("Nearshoring",0.6); add("Network Diversification",0.5)
            add("Lifecycle cost",0.5)
        if "raw material" in t or "shortage" in t:
            add("Inventory/Capacity Buffers",0.7); add("Ecosystem Partnerships",0.5)
            add("OEE",0.4); add("Supplier on-time delivery",0.4)
        if "digital twin" in t:
            add("Platform/Plant Harmonization",0.6); add("OEE",0.6); add("First-pass yield",0.5)
        if "lean logistics" in t:
            add("OTIF",0.6); add("Logistics lead time",0.7); add("Order Fulfillment",0.4)
    return mem

clamp01=lambda x:max(0.0,min(1.0,x))
clamp03=lambda x:max(0.0,min(3.0,x))

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
            "kpis":          score_matrix(BASE_KPIS,"kpis",w5s,stage,flags,mem,pillars),
            "drivers":       score_matrix(BASE_DRIVERS,"drivers",w5s,stage,flags,mem,pillars)}

# -------------------- LLMs --------------------
ANALYZER_SYSTEM = (
    "You are an experienced Supply Chain Director. Output ONLY JSON.\n"
    "Input: objective, industry, user_role, system_type, lce_stage, 5S weights, scenarios.\n"
    "Return: { pillars:{quality,cost,volume,time,flexibility,environment ∈ [0,1]}, reasons:[…], tags:[…] }.\n"
    "Sum of pillars should be ~1. Map time-to-market→time; ramp-up/learning→time+quality; scalability→volume+flexibility; ESG→environment; cost-focus→cost.\n"
    "Consider the manufacturing system type."
)
def analyze_to_pillars(payload: Dict) -> Dict:
    msgs=[{"role":"system","content":ANALYZER_SYSTEM},
          {"role":"user","content":json.dumps(payload, ensure_ascii=False)}]
    last=None
    for m in ANALYZER_CHAIN:
        try:
            r=client.chat.completions.create(model=m, messages=msgs, extra_headers=OPENROUTER_HEADERS,
                                             response_format={"type":"json_object"}, temperature=0.2, max_tokens=700)
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            last=e; time.sleep(0.2)
    raise RuntimeError(f"Analyzer failed: {last}")

SINGLE_GUIDANCE_SYSTEM = (
    "You are a senior supply-chain advisor. Speak directly to the user in second person, ≤160 words. "
    "Explain what their selected manufacturing system implies given: objective, role, industry, LCE stage, pillar weights, 5S priorities, scenarios, "
    "and the system’s scores for core processes, KPIs, and drivers (0–3 scale). "
    "Start several sentences with 'You should…', 'Prioritize…', 'Monitor…'. Do not invent targets."
)
def make_single_guidance(ctx: Dict) -> str:
    try:
        r=client.chat.completions.create(model=NARRATIVE_PRIMARY,
            messages=[{"role":"system","content":SINGLE_GUIDANCE_SYSTEM},
                      {"role":"user","content":json.dumps(ctx, ensure_ascii=False)}],
            extra_headers=OPENROUTER_HEADERS, temperature=0.4, max_tokens=700)
        return r.choices[0].message.content
    except Exception:
        r=client.chat.completions.create(model=NARRATIVE_FALLBACK,
            messages=[{"role":"system","content":SINGLE_GUIDANCE_SYSTEM},
                      {"role":"user","content":json.dumps(ctx, ensure_ascii=False)}],
            extra_headers=OPENROUTER_HEADERS, temperature=0.4, max_tokens=700)
        return r.choices[0].message.content

CROSS_GUIDANCE_SYSTEM = (
    "You are a senior supply-chain advisor. Speak directly to the user in second person, ≤200 words. "
    "Provide a cross-system comparison that makes sense to a decision-maker. "
    "Tie your explanation to the objective, role, industry, LCE stage, pillar weights, 5S priorities, and scenarios. "
    "Explain what the side-by-side matrices imply: which system better aligns with the pillar emphasis, "
    "where the selected system is outperformed (by which system and on which items), and what trade-offs to accept. "
    "End with 3 short bullet points (actions) keyed to the user’s role."
)
def make_cross_guidance(ctx: Dict) -> str:
    try:
        r=client.chat.completions.create(model=NARRATIVE_PRIMARY,
            messages=[{"role":"system","content":CROSS_GUIDANCE_SYSTEM},
                      {"role":"user","content":json.dumps(ctx, ensure_ascii=False)}],
            extra_headers=OPENROUTER_HEADERS, temperature=0.4, max_tokens=800)
        return r.choices[0].message.content
    except Exception:
        r=client.chat.completions.create(model=NARRATIVE_FALLBACK,
            messages=[{"role":"system","content":CROSS_GUIDANCE_SYSTEM},
                      {"role":"user","content":json.dumps(ctx, ensure_ascii=False)}],
            extra_headers=OPENROUTER_HEADERS, temperature=0.4, max_tokens=800)
        return r.choices[0].message.content

CHAT_SYSTEM = (
    "You are a helpful supply-chain copilot. Speak in second person. "
    "Answer using ONLY the provided results (pillars, scores, strategies, scenarios). "
    "If asked for metrics, reference intensities (0–3) and practical measurement examples (e.g., OTIF %, FPY %) without inventing targets."
)
def chat_reply(question: str, results_ctx: Dict, history: List[Dict], model: str) -> str:
    msgs=[{"role":"system","content":CHAT_SYSTEM},
          {"role":"user","content":json.dumps(results_ctx, ensure_ascii=False)}]
    for m in history[-8:]: msgs.append(m)
    msgs.append({"role":"user","content":question})
    r=client.chat.completions.create(model=model, messages=msgs, extra_headers=OPENROUTER_HEADERS,
                                     temperature=0.4, max_tokens=700)
    return r.choices[0].message.content

# -------------------- Sidebar (inputs) --------------------
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

    # View selector (LIVE; never triggers LLMs)
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
    st.multiselect("Preset scenarios", ["High demand volatility","Geopolitical risk","Carbon constraints"],
                   default=preset["preset"], key="preset_scenarios")
    st.text_input("Additional tags (comma-separated)", value=", ".join(preset["custom"]) if preset_name!="Other" else "", key="custom_tags_text")
    if preset_name=="Other":
        st.multiselect("Characteristics", ["Outsourced components","In-house novel tech","Vertical integration","Many suppliers","Single-source risk","Regionalization/Nearshoring","High ESG scrutiny","Safety-critical operations"],
                       default=["Many suppliers"], key="other_chars")

    st.header("Options")
    st.toggle("Compare all systems (view)", value=False, key="compare_all")

    with st.expander("Advanced (optional): Chat model", expanded=False):
        st.selectbox("Chat model", CHAT_CHOICES, index=CHAT_CHOICES.index(CHAT_DEFAULT), key="chat_model")

    if st.button("Analyze", use_container_width=True, key="analyze_btn"):
        # Build payload from sidebar controls
        role_val = st.session_state.get("user_role_other") if st.session_state.get("user_role")=="Other" else st.session_state.get("user_role")
        weights_5s = {s: st.session_state.get(f"s5_{s}", 0.5) for s in FIVE_S}
        flags={"volatility":"High demand volatility" in st.session_state.get("preset_scenarios",[]),
               "geo_risk":"Geopolitical risk" in st.session_state.get("preset_scenarios",[]),
               "carbon":"Carbon constraints" in st.session_state.get("preset_scenarios",[])}
        custom = [t.strip() for t in st.session_state.get("custom_tags_text","").split(",") if t.strip()] + \
                 [c.lower() for c in st.session_state.get("other_chars",[])]

        # 1) Analyzer → pillars (ONE CALL CHAIN)
        analysis_payload = {
            "objective": st.session_state.get("objective",""),
            "industry": st.session_state.get("industry",""),
            "user_role": role_val,
            "system_type": st.session_state.get("selected_system","Product Transfer"),  # just for flavor in reasons
            "lce_stage": st.session_state.get("lce_stage","Operation"),
            "weights_5s": weights_5s,
            "scenarios": {"preset": st.session_state.get("preset_scenarios",[]), "custom": custom}
        }
        try:
            analysis = analyze_to_pillars(analysis_payload)
            raw = analysis.get("pillars",{}); ssum = sum(float(raw.get(k,0.0)) for k in PILLARS)
            pillars = {k:(float(raw.get(k,0.0))/ssum if ssum>0 else 0.0) for k in PILLARS}
            reasons = analysis.get("reasons",[])
            tags = analysis.get("tags",[])
        except Exception as e:
            st.warning(f"Analyzer fallback (equal weights) due to error: {e}")
            pillars = {k:1.0/len(PILLARS) for k in PILLARS}; reasons=["Equal weighting used."]; tags=[]

        # 2) Deterministic scoring (for ALL systems) — ONE RUN
        scored = score_all(weights_5s, st.session_state.get("lce_stage","Operation"), flags, custom+tags, pillars)

        # 3) Build guidance contexts for each system (precompute & store texts)
        guidance_single = {}
        for sys in SYSTEMS:
            ctx = {
                "objective": st.session_state.get("objective",""),
                "industry": st.session_state.get("industry",""),
                "user_role": role_val,
                "system_type": sys,
                "lce_stage": st.session_state.get("lce_stage","Operation"),
                "pillar_weights": pillars,
                "weights_5s": weights_5s,
                "scenarios": {"preset": st.session_state.get("preset_scenarios",[]), "custom": custom+tags},
                "strategies": {"Competitive": COMPETITIVE[sys], "Value Chain": VALUE_CHAIN[sys], "Product/Service": PROD_SERVICE[sys]},
                "selected_scores": {
                    "core_processes": {k: float(v[sys]) for k, v in scored["core_processes"].items()},
                    "kpis":           {k: float(v[sys]) for k, v in scored["kpis"].items()},
                    "drivers":        {k: float(v[sys]) for k, v in scored["drivers"].items()},
                }
            }
            guidance_single[sys] = make_single_guidance(ctx)  # stored text (no re-run later)

        # 4) Cross-system narrative (precompute once)
        # Build a compact cross context containing all scores per system
        cross_ctx = {
            "objective": st.session_state.get("objective",""),
            "industry": st.session_state.get("industry",""),
            "user_role": role_val,
            "lce_stage": st.session_state.get("lce_stage","Operation"),
            "pillar_weights": pillars,
            "weights_5s": weights_5s,
            "scenarios": {"preset": st.session_state.get("preset_scenarios",[]), "custom": custom+tags},
            "strategies_by_system": {
                sys: {"Competitive": COMPETITIVE[sys], "Value Chain": VALUE_CHAIN[sys], "Product/Service": PROD_SERVICE[sys]}
                for sys in SYSTEMS
            },
            "scores": scored  # all matrices for all systems
        }
        guidance_cross = make_cross_guidance(cross_ctx)

        # Persist everything for rendering/chat (no more LLMs on toggles)
        st.session_state["results"] = {
            "objective": st.session_state.get("objective",""),
            "industry": st.session_state.get("industry",""),
            "user_role": role_val,
            "lce_stage": st.session_state.get("lce_stage","Operation"),
            "weights_5s": weights_5s,
            "scenario_flags": flags,
            "preset_scenarios": st.session_state.get("preset_scenarios",[]),
            "custom_tags": custom+tags,
            "pillars": pillars,
            "scored": scored,
            "reasons": reasons,
            "guidance_single": guidance_single,  # dict: system -> text
            "guidance_cross": guidance_cross,    # string
        }

# -------------------- Helpers for rendering --------------------
def to_df(mapping: Dict[str, Dict[str, float]], title: str) -> pd.DataFrame:
    df = pd.DataFrame(mapping).T; df.index.name = title; return df

def style_df(df: pd.DataFrame, selected_col: str = None, compare_all: bool = False):
    cmap={0:"#E6E6E6",1:"#FFE6B3",2:"#FFB366",3:"#4CAF50"}
    def color_fn(v):
        b = 3 if v >= 2.6 else 2 if v >= 1.6 else 1 if v >= 0.6 else 0
        return f"background-color: {cmap[b]}"
    sty = df.style.map(color_fn).format("{:.2f}")
    if compare_all and selected_col and selected_col in df.columns:
        other_cols = [c for c in df.columns if c != selected_col]
        sty = sty.set_properties(subset=other_cols, **{"color":"#7a7a7a"})
    return sty

def filter_df_by_system(df: pd.DataFrame, sys_type: str, compare_all: bool):
    return df if compare_all else df[[sys_type]]

# -------------------- Main render (never calls LLMs) --------------------
st.title("Supply-Chain Strategy Agent")
st.markdown("Developed by: **Dr. J. Isabel Méndez** & **Dr. Arturo Molina**")

if "results" in st.session_state:
    res = st.session_state["results"]
    selected_system = st.session_state.get("selected_system","Product Transfer")
    compare_all = st.session_state.get("compare_all", False)

    cp_df = to_df(res["scored"]["core_processes"], "Core Processes")
    kpi_df = to_df(res["scored"]["kpis"], "KPIs")
    drv_df = to_df(res["scored"]["drivers"], "Resilience Drivers")

    cp_view  = filter_df_by_system(cp_df,  selected_system, compare_all)
    kpi_view = filter_df_by_system(kpi_df, selected_system, compare_all)
    drv_view = filter_df_by_system(drv_df, selected_system, compare_all)

    st.markdown(f"**Selected system (view):** `{selected_system}` · **LCE:** `{res['lce_stage']}` · **Compare all:** `{compare_all}`")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Core Processes × System")
        st.write(style_df(cp_view, selected_col=selected_system, compare_all=compare_all), unsafe_allow_html=True)
    with c2:
        st.subheader("KPIs × System (pillar + 5S + LCE aware)")
        st.write(style_df(kpi_view, selected_col=selected_system, compare_all=compare_all), unsafe_allow_html=True)
    st.subheader("Resilience Drivers × System (scenario aware)")
    st.write(style_df(drv_view, selected_col=selected_system, compare_all=compare_all), unsafe_allow_html=True)

    if res.get("reasons"):
        st.markdown("**Analyzer → pillar weights rationale**")
        st.write("\n".join([f"- {r}" for r in res["reasons"]]))

    st.markdown("### Guidance")
    if compare_all:
        # Show the precomputed cross-system explanation
        st.write(res["guidance_cross"])
    else:
        # Show the precomputed single-system guidance for the currently viewed system
        st.write(res["guidance_single"].get(selected_system, "Guidance not available for this system."))

else:
    st.info("Configure inputs on the left and click **Analyze** to compute once. Then use the view controls freely.")

# -------------------- Chat (below; only uses stored results) --------------------
st.markdown("---")
st.subheader("Chat with the Strategy Agent")
if "chat" not in st.session_state: st.session_state["chat"]=[]
for m in st.session_state["chat"]:
    with st.chat_message(m["role"]): st.markdown(m["content"])

user_q = st.chat_input("Ask about the results, trade-offs, or next steps…")
if user_q:
    st.session_state["chat"].append({"role":"user","content":user_q})
    with st.chat_message("user"): st.markdown(user_q)
    if "results" not in st.session_state:
        reply = "Please run **Analyze** first so I can ground my answers in your matrices."
    else:
        res = st.session_state["results"]
        selected_system = st.session_state.get("selected_system","Product Transfer")
        chat_ctx = {
            "objective": st.session_state.get("objective",""),
            "industry": st.session_state.get("industry",""),
            "role": st.session_state.get("user_role_other") if st.session_state.get("user_role")=="Other" else st.session_state.get("user_role"),
            "system_type": selected_system,
            "lce_stage": res["lce_stage"],
            "pillars": res["pillars"],
            "weights_5s": res["weights_5s"],
            "scenarios": {"preset": res["preset_scenarios"], "custom": res["custom_tags"]},
            "strategies": {"Competitive": COMPETITIVE[selected_system],
                           "Value Chain": VALUE_CHAIN[selected_system],
                           "Product/Service": PROD_SERVICE[selected_system]},
            "scores_selected_system": {
                "core_processes": {k: float(v[selected_system]) for k, v in res["scored"]["core_processes"].items()},
                "kpis":           {k: float(v[selected_system]) for k, v in res["scored"]["kpis"].items()},
                "drivers":        {k: float(v[selected_system]) for k, v in res["scored"]["drivers"].items()},
            }
        }
        chat_model = st.session_state.get("chat_model", CHAT_DEFAULT)
        reply = chat_reply(user_q, chat_ctx, st.session_state["chat"], chat_model)
    st.session_state["chat"].append({"role":"assistant","content":reply})
    with st.chat_message("assistant"): st.markdown(reply)

st.caption("Scores are continuous in [0,3]. 3 = Core/Critical/Essential; 2 = Strategic/Important; 1 = Relevant; 0 = Not emphasized.")









