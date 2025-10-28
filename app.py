# =====================================================
# Supply-Chain Strategy Agent (LCE + 5S)
# Complete Research + Interactive Strategy Agent
# Authors: Dr. J. Isabel Méndez & Dr. Arturo Molina
# =====================================================

import os, json, re
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer  # kept as in your code

# =====================================================
#                   SETUP
# =====================================================
st.set_page_config(page_title="Supply-Chain Agent (LCE+5S)", layout="wide")

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"]
)
OPENROUTER_HEADERS = {
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "LCE+5S Supply-Chain Agent"
}

# --- Single Unified Model for ALL tasks ---
LLM_MODEL = "mistralai/mistral-7b-instruct:free"

# =====================================================
#                CANONICAL VOCAB
# =====================================================
SYSTEMS = ["Product Transfer","Technology Transfer","Facility Design"]
LCE = ["Ideation","Basic Development","Advanced Development","Launch","Operation","End-of-Life"]
FIVE_S = ["Social","Sustainable","Sensing","Smart","Safe"]

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

S_TAGS_KPI = {
    "ESG index":{"Sustainable":1.0},
    "Lifecycle cost":{"Sustainable":0.6},
    "Safety incidents":{"Safe":1.0},
    "OEE":{"Smart":0.4,"Sensing":0.4},
    "First-pass yield":{"Smart":0.4,"Sensing":0.4},
    "Supplier on-time delivery":{"Social":0.6},
    "Logistics lead time":{"Social":0.6}
}
S_TAGS_CORE = {
    "Co-Engineering":{"Social":0.6,"Smart":0.3},
    "SRM":{"Social":1.0},
    "Order Fulfillment":{"Social":0.8,"Smart":0.2},
    "Customer Service":{"Social":1.0},
    "Customer Driven Design":{"Smart":0.4,"Social":0.4},
    "NPD":{"Smart":0.6}
}
S_TAGS_DRIVERS = {
    "Nearshoring":{"Sustainable":0.6,"Safe":0.2},
    "Platform/Plant Harmonization":{"Smart":0.5},
    "Ecosystem Partnerships":{"Social":0.6}
}

STAGE_TAGS_KPI = {
    "Ramp-up time":{"Launch":1.0,"Advanced Development":0.6},
    "First-pass yield":{"Launch":0.7,"Operation":0.5},
    "Learning-curve productivity":{"Launch":0.6,"Operation":0.4},
    "OEE":{"Operation":1.0},
    "OTIF":{"Operation":0.7},
    "Lifecycle cost":{"End-of-Life":0.6,"Operation":0.4},
    "ESG index":{"End-of-Life":0.9,"Operation":0.5}
}
STAGE_TAGS_CORE = {
    "Co-Engineering":{"Ideation":0.8,"Basic Development":0.6},
    "NPD":{"Advanced Development":0.8,"Launch":0.5},
    "Order Fulfillment":{"Operation":0.9},
    "Customer Service":{"Operation":0.7},
    "SRM":{"Operation":0.6}
}
STAGE_TAGS_DRIVERS = {
    "Inventory/Capacity Buffers":{"Operation":0.7},
    "Platform/Plant Harmonization":{"Operation":0.7,"Launch":0.4},
    "Nearshoring":{"Operation":0.4,"End-of-Life":0.4}
}

# =====================================================
#                  CORE SCORING FUNCTIONS
# =====================================================
def clamp01(x): return max(0.0,min(1.0,x))
def clamp03(x): return max(0.0,min(3.0,x))

def s_boost(w,s_tags,name): 
    return sum(w.get(k,0.0)*v for k,v in s_tags.get(name,{}).items())

def stage_boost(stage, tags, name, max_gain=0.8): 
    return clamp01(tags.get(name,{}).get(stage,0.0))*max_gain

def score_matrix(base_map, matrix, w5s, stage):
    out = {}
    for item, cols in base_map.items():
        out[item] = {}
        for system, base in cols.items():
            score = float(base)

            # --- KPI Matrix ---
            if matrix == "kpis":
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

def score_all(w5s, stage):
    return {
        "core_processes": score_matrix(BASE_CORE, "core_processes", w5s, stage),
        "kpis": score_matrix(BASE_KPIS, "kpis", w5s, stage),
        "drivers": score_matrix(BASE_DRIVERS, "drivers", w5s, stage),
    }

# =====================================================
#  HELPERS (LLM + formatting)
# =====================================================
def _json_default(o):
    import numpy as _np, pandas as _pd
    if isinstance(o, (_np.integer, _np.floating)): return float(o)
    if isinstance(o, (_pd.Timestamp,)): return o.isoformat()
    return str(o)

def compact_dict(d, max_items=10):
    import numpy as _np
    def _compact(v, depth=0):
        if isinstance(v, dict):
            out = {}
            for i, (kk, vv) in enumerate(v.items()):
                if i >= max_items:
                    break
                out[str(kk)] = _compact(vv, depth + 1)
            return out
        if isinstance(v, (list, tuple)):
            return [_compact(x, depth + 1) for x in v[:max_items]]
        if isinstance(v, (_np.integer, _np.floating)):
            return float(v)
        if isinstance(v, (int, float, bool)) or v is None:
            return v
        if isinstance(v, str):
            return v
        return str(v)
    return {str(k): _compact(v) for k, v in d.items()}

def clean_numbers(text: str) -> str:
    return re.sub(r"\s*\(\d+(\.\d+)?\)", "", text)

def safe_llm_call(prompt: str, payload: dict, temp=0.35, max_toks=400, retries=2):
    for _ in range(retries):
        try:
            r = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=_json_default)},
                ],
                extra_headers=OPENROUTER_HEADERS,
                temperature=temp,
                max_tokens=max_toks,
            )
            out = r.choices[0].message.content.strip()
            if out:
                return out
        except Exception as e:
            st.warning(f"⚠️ LLM call failed: {e}")
    return ""

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
    if st.button("🔄 Reset App"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    preset_name = st.selectbox("Preset", list(PRESETS.keys()), key="preset")
    preset = PRESETS[preset_name]

    st.radio("Manufacturing system (view)", SYSTEMS,
             index=SYSTEMS.index(preset["system"]) if preset["system"] in SYSTEMS else 0,
             key="selected_system")

    st.text_input("Objective", value=preset["objective"] or "Design and ramp a flexible small manufacturing cell.", key="objective")
    st.selectbox("Industry", ["Automotive","Electronics","Medical Devices","Consumer Goods","Other"], index=1, key="industry")

    roles = ["Design Engineer","Process Engineer","Manufacturing Engineer","Safety Supervisor","Sustainability Manager","Supply Chain Analyst","Manager/Decision Maker","Other"]
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
st.title("Supply-Chain Strategy Agent (LCE + 5S)")
st.markdown("Developed by: **Dr. J. Isabel Méndez** & **Dr. Arturo Molina**")

analyze_clicked = st.button("Analyze", use_container_width=True)

if analyze_clicked:
    role_val = st.session_state.get("user_role_other") if st.session_state.get("user_role")=="Other" else st.session_state.get("user_role")
    weights_5s = {s: st.session_state.get(f"s5_{s}", 0.5) for s in FIVE_S}
    lce_stage = st.session_state.get("lce_stage","Operation")
    scored = score_all(weights_5s, lce_stage)

    st.session_state["results"] = {
        "scored": scored,
        "weights_5s": weights_5s,
        "elapsed": 0.0
    }
    st.session_state["analyzed"] = True
    st.session_state["llm_done"] = False

# =====================================================
#  HELPER: DISPLAY MATRIX WITH LABEL COLORS
# =====================================================
def show_matrix(title, df_dict):
    st.markdown(f"### {title}")
    df = pd.DataFrame(df_dict).T
    compare_all = st.session_state.get("compare_all", False)
    selected = st.session_state.get("selected_system", "Product Transfer")

    # Handle cases where system selection comes as list/dict
    if isinstance(selected, (tuple, list)):
        selected = selected[0]
    if isinstance(selected, dict):
        selected = next(iter(selected.values()))

    # Filter or show all systems
    if not compare_all:
        if selected in df.columns:
            df = df[[selected]]
        else:
            st.warning(f"⚠️ The selected system '{selected}' is not available; showing all instead.")

    # Map numeric scores to qualitative labels
    df_label = df.applymap(lambda x: "Low" if x < 1 else "Medium" if x < 2 else "High")

    color_map = {"Low": "#f8d7da", "Medium": "#fff3cd", "High": "#d4edda"}
    styled = df_label.style.applymap(
        lambda v: f"background-color: {color_map[v]}; color:black; text-align:center; font-weight:bold;"
    )

    st.dataframe(styled, use_container_width=True)
    return df_label

# =====================================================
#            RESULTS RENDERING SECTION (DETAILED)
# =====================================================
tabs = st.tabs(["📊 Matrices", "🧠 Interpretations", "⚖️ Comparative", "💬 Chat"])

# ---------- TAB 1: MATRICES ----------
with tabs[0]:
    if "results" in st.session_state:
        res = st.session_state["results"]["scored"]
        show_matrix("Core Processes × System", res["core_processes"])
        show_matrix("KPIs × System", res["kpis"])
        show_matrix("Resilience Drivers × System", res["drivers"])
    else:
        st.info("Run **Analyze** first to see matrices.")

# ---------- TAB 2: INTERPRETATIONS (Role-based prompts preserved) ----------
with tabs[1]:
    if "results" in st.session_state:
        res = st.session_state["results"]

        if not st.session_state.get("llm_done", False):
            st.info("Generating qualitative interpretations with the LLM...")

            sel_sys = st.session_state.get("selected_system", "Product Transfer")
            role = st.session_state.get("user_role", "")
            industry = st.session_state.get("industry", "")
            objective = st.session_state.get("objective", "")

            # ---- CORE ----
            core_scores = {k: float(v.get(sel_sys, 0)) for k, v in res["scored"]["core_processes"].items()}
            core_labels = {k: ("High" if v >= 2 else "Medium" if v >= 1 else "Low") for k, v in core_scores.items()}
            core_payload = {"core_labels": core_labels}
            prompt_core = f"""
            You are a supply-chain strategist advising a {role} in the {industry} industry.
            Below is the qualitative status of each core process for the {sel_sys} system:
            {json.dumps(core_labels, indent=2)}
            Interpret these labels as *priority levels*, not performance metrics.
            Provide a qualitative explanation—no numbers, percentages, or parentheses.
            Describe which High-priority processes should remain central to achieving "{objective}",
            which Medium ones deserve strengthening, and which Low ones can be simplified or delegated.
            Keep tone analytical and directive, under 170 words.
            """
            core_expl = safe_llm_call(prompt_core, core_payload)

            # ---- KPIs ----
            kpi_scores = {k: float(v.get(sel_sys, 0)) for k, v in res["scored"]["kpis"].items()}
            kpi_labels = {k: ("High" if v >= 2 else "Medium" if v >= 1 else "Low") for k, v in kpi_scores.items()}
            kpi_payload = {"kpi_labels": kpi_labels}
            prompt_kpi = f"""
            You are a performance strategist advising a {role} in the {industry} sector.
            Below is the qualitative status of each KPI for the {sel_sys} system:
            {json.dumps(kpi_labels, indent=2)}
            Interpret these labels as *priority signals*, not results or measurements.
            Write only qualitative insights—avoid any numbers, percentages, or parentheses.
            Highlight which High-priority KPIs sustain competitive advantage, which Medium KPIs require optimization,
            and which Low KPIs reflect improvement opportunities aligned with "{objective}".
            Keep tone professional, concise, and prescriptive, under 170 words.
            """
            kpi_expl = safe_llm_call(prompt_kpi, kpi_payload)

            # ---- DRIVERS ----
            driver_scores = {k: float(v.get(sel_sys, 0)) for k, v in res["scored"]["drivers"].items()}
            driver_labels = {k: ("High" if v >= 2 else "Medium" if v >= 1 else "Low") for k, v in driver_scores.items()}
            driver_payload = {"driver_labels": driver_labels}
            prompt_drv = f"""
            You are a resilience strategist advising a {role} in the {industry} industry.
            Below is the qualitative status of each resilience driver for the {sel_sys} system:
            {json.dumps(driver_labels, indent=2)}
            Interpret these labels as *priority guidance* for building robustness and adaptability.
            Do not use numbers, percentages, or parentheses—focus on qualitative reasoning only.
            Explain which High-priority drivers reinforce stability, which Medium drivers enhance flexibility,
            and which Low drivers represent emerging areas of attention to strengthen "{objective}".
            Keep tone prescriptive, analytical, and under 170 words.
            """
            driver_expl = safe_llm_call(prompt_drv, driver_payload)

            st.session_state["llm_interpretations"] = {
                "core": clean_numbers(core_expl),
                "kpi": clean_numbers(kpi_expl),
                "drivers": clean_numbers(driver_expl)
            }
            st.session_state["llm_done"] = True

        inter = st.session_state["llm_interpretations"]
        st.markdown("### Core Processes Interpretation")
        st.write(inter["core"])
        st.markdown("### KPI Interpretation")
        st.write(inter["kpi"])
        st.markdown("### Resilience Drivers Interpretation")
        st.write(inter["drivers"])
    else:
        st.info("Run **Analyze** first to enable interpretations.")

# ---------- TAB 3: COMPARATIVE INTERPRETATION ----------
with tabs[2]:
    if "results" in st.session_state:
        if st.session_state.get("compare_all", False):
            res = st.session_state["results"]["scored"]
            sel_sys = st.session_state.get("selected_system", "Product Transfer")
            objective = st.session_state.get("objective", "")
            role = st.session_state.get("user_role", "")
            industry = st.session_state.get("industry", "")

            other_systems = [s for s in SYSTEMS if s != sel_sys]
            if len(other_systems) == 0:
                st.info("Select a system to enable comparison.")
            else:
                others_str = " and ".join(other_systems)
                compare_payload = compact_dict({
                    "selected_system": sel_sys,
                    "other_systems": other_systems,
                    "objective": objective,
                    "role": role,
                    "industry": industry,
                    "core": res["core_processes"],
                    "kpis": res["kpis"],
                    "drivers": res["drivers"],
                })
                compare_prompt = f"""
                You are an expert supply-chain strategist.
                Compare {sel_sys} with {others_str} across Core Processes, KPIs, and Drivers.
                Explain relative strengths and weaknesses from the lens of a {role} aiming to achieve "{objective}".
                Highlight where {sel_sys} outperforms and where the others offer advantages, and indicate complementarities.
                Conclude with one actionable recommendation to balance synergy, resilience, and innovation.
                Keep tone analytical and concise (≤180 words). Avoid numeric values or parentheses.
                """
                compare_expl = safe_llm_call(compare_prompt, compare_payload, temp=0.35, max_toks=450)
                if compare_expl:
                    st.markdown("### Comparative Interpretation")
                    st.write(clean_numbers(compare_expl))
                else:
                    st.warning("⚠️ Comparative interpretation returned no content or was truncated.")
        else:
            st.info("Activate **Compare all systems (view)** in the sidebar to generate a comparison.")
    else:
        st.info("Run **Analyze** first.")

# ---------- TAB 4: CHAT ----------
with tabs[3]:
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
            st.session_state["chat"].append({"role":"assistant","content":reply})
            with st.chat_message("assistant"): st.markdown(reply)
        else:
            res = st.session_state["results"]
            ctx = {"weights_5s":res["weights_5s"],"scores":res["scored"]}
            try:
                r = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role":"system","content":"You are a supply-chain advisor."},
                        {"role":"user","content":json.dumps(ctx, ensure_ascii=False, default=_json_default)},
                        {"role":"user","content":user_q}
                    ],
                    extra_headers=OPENROUTER_HEADERS,
                    temperature=0.4,
                    max_tokens=600
                )
                reply = r.choices[0].message.content
            except Exception as e:
                reply = f"LLM error: {e}"

            st.session_state["chat"].append({"role":"assistant","content":reply})
            with st.chat_message("assistant"): st.markdown(reply)
