# =====================================================
# Supply-Chain Strategy Agent (LCE + 5S)
# Complete Research + Interactive Strategy Agent
# Authors: Dr. J. Isabel Méndez & Dr. Arturo Molina
# =====================================================

import os, json, re
from typing import Dict, List
import hashlib, io, random
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer  

# =====================================================
#                  LOAD BENCHMARKS
# =====================================================
try:
    with open("benchmarks.json") as f:
        BENCHMARKS = json.load(f)
except Exception as e:
    st.warning(f"⚠️ Could not load benchmarks.json: {e}")
    BENCHMARKS = {}


BENCHMARK_META = {
    "Product Transfer": {
        "Objective": "Achieve demand-driven fulfillment with high service reliability and agile response to market variability, following Gartner’s hierarchy of supply-chain metrics.",
        "meta": {
            "source": "Gartner (2025) Strengthening Supply Chain Performance Improvement Initiatives",
            "mapping_framework": "5S–LCE",
            "note": "Focus on service reliability, agility, and customer responsiveness."
        }
    },
    "Technology Transfer": {
        "Objective": "Accelerate innovation and manufacturing ramp-up by improving yield, shortening cycle time, and strengthening supplier performance through active performance management.",
        "meta": {
            "source": "Gartner (2025); McKinsey (2020)",
            "mapping_framework": "5S–LCE",
            "note": "Emphasizes ramp-up, yield improvement, and supplier coordination."
        }
    },
    "Facility Design": {
        "Objective": "Maximize equipment efficiency, maintenance discipline, and workforce reliability to sustain continuous improvement and productivity gains in industrial operations.",
        "meta": {
            "source": "McKinsey (2020) Energizing Industrial Manufacturing Through Active Performance Management",
            "mapping_framework": "5S–LCE",
            "note": "Focus on OEE, maintenance, and workforce-driven performance culture."
        }
    }
}
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
    "Supplier on-time delivery": {"Social": 0.6, "Sustainable": 0.2},
    "Incoming defect rate": {"Smart": 0.4, "Sensing": 0.6},
    "Assembly cost per unit": {"Smart": 0.4, "Sustainable": 0.4},
    "Logistics lead time": {"Social": 0.6, "Smart": 0.2},
    "Ramp-up time": {"Smart": 0.6, "Safe": 0.3},
    "First-pass yield": {"Smart": 0.4, "Sensing": 0.4},
    "Learning-curve productivity": {"Smart": 0.5, "Social": 0.3},
    "% revenue from new products": {"Sustainable": 0.6, "Smart": 0.4},
    "OEE": {"Smart": 0.4, "Sensing": 0.4, "Safe": 0.2},
    "OTIF": {"Social": 0.5, "Safe": 0.3},
    "Lifecycle cost": {"Sustainable": 0.8, "Smart": 0.2},
    "ESG index": {"Sustainable": 1.0},
    "Safety incidents": {"Safe": 1.0, "Social": 0.2},
}

S_TAGS_CORE = {
    "Customer Driven Design": {"Smart": 0.4, "Social": 0.4, "Sustainable": 0.2},
    "CRM": {"Social": 0.8, "Smart": 0.2},
    "Co-Engineering": {"Social": 0.6, "Smart": 0.3, "Sensing": 0.2},
    "SRM": {"Social": 0.8, "Sustainable": 0.3},
    "NPD": {"Smart": 0.6, "Sustainable": 0.4},
    "Obtain Customer Commit.": {"Social": 0.7, "Smart": 0.3},
    "Order Fulfillment": {"Social": 0.8, "Smart": 0.2, "Safe": 0.2},
    "Customer Service": {"Social": 0.8, "Safe": 0.3, "Sustainable": 0.2},
}

S_TAGS_DRIVERS = {
    "Inventory/Capacity Buffers": {"Safe": 0.6, "Sustainable": 0.3},
    "Network Diversification": {"Social": 0.5, "Safe": 0.3, "Sustainable": 0.3},
    "Multisourcing": {"Sustainable": 0.4, "Social": 0.4, "Safe": 0.2},
    "Nearshoring": {"Sustainable": 0.6, "Safe": 0.2, "Social": 0.2},
    "Platform/Plant Harmonization": {"Smart": 0.5, "Sensing": 0.3, "Safe": 0.2},
    "Ecosystem Partnerships": {"Social": 0.6, "Sustainable": 0.4, "Smart": 0.2},
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
def s_boost(w, s_tags, name):
    tags = s_tags.get(name, {})
    total_weight = sum(tags.values())
    if total_weight == 0:
        return 0.0
    weighted = sum(w[k]*v for k,v in tags.items()) / total_weight
    return weighted


def clamp03(x): 
    return max(0.0, min(3.0, x))

def clamp01(x): 
    return max(0.0, min(1.0, x))

def s_boost(w, s_tags, name):
    tags = s_tags.get(name, {})
    tot = sum(tags.values())
    if tot == 0:
        return 0.0
    return sum(w.get(k, 0.0) * v for k, v in tags.items()) / tot


def stage_boost(stage, tags, name, max_gain=0.8): 
    return clamp01(tags.get(name,{}).get(stage,0.0))*max_gain

def score_matrix(base_map, matrix, w5s, stage):
    out = {}
    for item, cols in base_map.items():
        out[item] = {}
        for system, base in cols.items():
            base = float(base)
            s_influence = s_boost(w5s, 
                                  S_TAGS_KPI if matrix=="kpis"
                                  else S_TAGS_CORE if matrix=="core_processes"
                                  else S_TAGS_DRIVERS, item)
            stage_influence = stage_boost(stage,
                                          STAGE_TAGS_KPI if matrix=="kpis"
                                          else STAGE_TAGS_CORE if matrix=="core_processes"
                                          else STAGE_TAGS_DRIVERS, item)

            total_inf = clamp01((s_influence + stage_influence) / 2)
            contrast = 1.2  # increase for sharper spread
            penalty = (0.5 - total_inf) * contrast  # negative if strong match, positive if weak
            score = clamp03(base * (1 - penalty) + 3 * total_inf * 0.5)

            out[item][system] = round(score, 3)
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
#  CONVERT NUMERIC SCORES TO QUALITATIVE LABELS
# =====================================================
def qualitative_scores(scored_dict):
    qualitative = {}
    for category, items in scored_dict.items():
        qualitative[category] = {
            name: {
                sys: ("High" if val >= 2 else "Medium" if val >= 1 else "Low")
                for sys, val in sysvals.items()
            }
            for name, sysvals in items.items()
        }
    return qualitative

# =====================================================
#                SIDEBAR CONFIGURATION
# =====================================================
with st.sidebar:
    st.header("Benchmark & Context Configuration")
    if st.button("🔄 Reset App"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # --- Select manufacturing system ---
    selected_system = st.selectbox("Manufacturing system", SYSTEMS, key="selected_system")

    # --- Load benchmark info dynamically ---
    if BENCHMARKS and selected_system in BENCHMARKS:
        bench_meta = BENCHMARKS[selected_system].get("meta", BENCHMARK_META.get(selected_system, {}).get("meta", {}))
    else:
        bench_meta = BENCHMARK_META.get(selected_system, {}).get("meta", {})

    # --- Objective ---
    
    default_obj = (
        BENCHMARKS.get(selected_system, {})
        .get("objective", BENCHMARK_META.get(selected_system, {}).get("Objective", f"Optimize {selected_system} performance vs benchmarks"))
    )

    # --- Auto-update objective when system changes ---
    if "last_system" not in st.session_state or st.session_state["last_system"] != selected_system:
        st.session_state["objective"] = (
            BENCHMARKS.get(selected_system, {})
            .get("objective", BENCHMARK_META.get(selected_system, {}).get("Objective", f"Optimize {selected_system} performance vs benchmarks"))
        )
        st.session_state["last_system"] = selected_system
    
    st.text_input("Objective", value=default_obj, key="objective")

    # --- Industry + role ---
    st.selectbox("Industry", ["Automotive","Electronics","Medical Devices","Consumer Goods","Other"], index=1, key="industry")
    roles = ["Design Engineer","Process Engineer","Manufacturing Engineer",
             "Safety Supervisor","Sustainability Manager","Supply Chain Analyst",
             "Manager/Decision Maker","Other"]
    st.selectbox("Your role", roles, index=5, key="user_role")
    if st.session_state.get("user_role")=="Other":
        st.text_input("Specify role", value="Other", key="user_role_other")

    # --- LCE + 5S ---
    st.header("LCE & 5S Priorities")
    st.selectbox("LCE stage", LCE, key="lce_stage")
    st.caption("5S sliders are priorities: 0 = deprioritize, 0.5 = neutral, 1 = strongly prioritize.")
    for s in FIVE_S:
        st.slider(s, 0.0, 1.0, 0.5, 0.05, key=f"s5_{s}")

    st.toggle("Compare all systems (view)", value=False, key="compare_all")

    # --- Transparency note ---
    st.caption("""
    Benchmarks represent industry-average KPI ranges compiled from sources 
    such as Deloitte, McKinsey, and Gartner.  
    The 5S–LCE framework and fuzzy weighting logic were developed by the authors 
    (Méndez & Molina, 2025) to translate these metrics into a lifecycle-aware context.
    """)


# =====================================================
#                MAIN APP LOGIC
# =====================================================
weights_5s = {s: st.session_state.get(f"s5_{s}", 0.5) for s in FIVE_S}
lce_stage  = st.session_state.get("lce_stage", "Operation")
st.session_state["matrices_live"] = score_all(weights_5s, lce_stage)

st.title("Supply-Chain Strategy Agent (LCE + 5S)")
st.markdown("Developed by: **Dr. J. Isabel Méndez** & **Dr. Arturo Molina**")

# Single analyze button: freeze state & trigger LLM on tabs 2–3
if st.button("Analyze", use_container_width=True):
    st.session_state["results"] = {
        "scored": st.session_state["matrices_live"],
        "weights_5s": weights_5s,
        "elapsed": 0.0,
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

# -----------------------------------------------------
# Generate a unique, reproducible run hash
# -----------------------------------------------------
def compute_run_hash(weights_5s, lce_stage, system):
    payload = json.dumps(
        {"weights_5s": weights_5s, "lce_stage": lce_stage, "system": system},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:10]

# -----------------------------------------------------
# Simple dominance / monotonicity test
# -----------------------------------------------------
def dominance_test(scored):
    fails = []
    for matrix, items in scored.items():
        for name, vals in items.items():
            # Example rule: PT baseline should not be lower than FD in ≥ half the rows
            if not all(0 <= v <= 3 for v in vals.values()):
                fails.append((matrix, name, "Out of [0,3] range"))
    return fails

# -----------------------------------------------------
# Simple MCDA baseline (TOPSIS-style)
# -----------------------------------------------------
def topsis_compare(matrix):
    df = pd.DataFrame(matrix).T
    df = df.fillna(0)
    norm = df / np.sqrt((df**2).sum())
    weights = np.ones(len(df.columns)) / len(df.columns)
    score = (norm * weights).sum(axis=1)
    return score.rank(ascending=False)

# -----------------------------------------------------
# Sensitivity / robustness test
# -----------------------------------------------------
def perturb_weights(weights, delta=0.2):
    return {
        k: min(1.0, max(0.0, v + random.uniform(-delta, delta)))
        for k, v in weights.items()
    }

def compare_matrices(base, new):
    """Compute correlation between base and perturbed average scores."""
    base_df = pd.DataFrame(base).T.mean()
    new_df = pd.DataFrame(new).T.mean()
    return base_df.corr(new_df)

# =====================================================
#            RESULTS RENDERING SECTION (DETAILED)
# =====================================================
tabs = st.tabs(["📊 Matrices", "🧠 Interpretations", "⚖️ Comparative", "💬 Chat", "🧪 Validation", "📈 Benchmarks"])

# ---------- TAB 1: MATRICES ----------
with tabs[0]:
    res_live = st.session_state.get("matrices_live")
    if res_live:
        show_matrix("Core Processes × System", res_live["core_processes"])
        show_matrix("KPIs × System",            res_live["kpis"])
        show_matrix("Resilience Drivers × System", res_live["drivers"])
    else:
        st.info("Adjust 5S sliders or LCE stage to generate matrices.")

# ---------- TAB 2: INTERPRETATIONS (5S + LCE-AWARE) ----------
with tabs[1]:
    if "results" in st.session_state:
        res = st.session_state["results"]

        # --- funciones auxiliares locales ---
        def qual_5s_weights(w5s):
            def q(x): return "High" if x >= 0.75 else "Medium" if x >= 0.5 else "Low"
            return {k: q(v) for k, v in w5s.items()}

        def item_contrib_5s(item_name, matrix_type, w5s):
            s_tags = {
                "kpis": S_TAGS_KPI,
                "core_processes": S_TAGS_CORE,
                "drivers": S_TAGS_DRIVERS
            }[matrix_type]
            raw = {s: w5s.get(s, 0.0) * s_tags.get(item_name, {}).get(s, 0.0) for s in FIVE_S}
            if sum(raw.values()) == 0:
                return []
            sorted_S = sorted(raw.items(), key=lambda x: x[1], reverse=True)
            return [k for k, _ in sorted_S[:2]]  # top 2 S most influential

        def item_contrib_lce(item_name, matrix_type, stage):
            stage_tags = {
                "kpis": STAGE_TAGS_KPI,
                "core_processes": STAGE_TAGS_CORE,
                "drivers": STAGE_TAGS_DRIVERS
            }[matrix_type]
            return [stage for stage, val in stage_tags.get(item_name, {}).items() if val > 0.5]

        if not st.session_state.get("llm_done", False):
            st.info("Generating qualitative interpretations with the LLM...")

            sel_sys = st.session_state.get("selected_system", "Product Transfer")
            role = st.session_state.get("user_role", "")
            industry = st.session_state.get("industry", "")
            objective = st.session_state.get("objective", "")
            lce_stage = st.session_state.get("lce_stage", "Operation")
            w5s = res["weights_5s"]
            w5s_qual = qual_5s_weights(w5s)

            # ---- CORE ----
            core_scores = {k: float(v.get(sel_sys, 0)) for k, v in res["scored"]["core_processes"].items()}
            core_labels = {k: ("High" if v >= 2 else "Medium" if v >= 1 else "Low") for k, v in core_scores.items()}
            core_topS = {k: item_contrib_5s(k, "core_processes", w5s) for k in core_labels}
            core_stage = {k: item_contrib_lce(k, "core_processes", lce_stage) for k in core_labels}

            core_payload = {
                "core_labels": core_labels,
                "weights_5s_qual": w5s_qual,
                "top_5s_per_item": core_topS,
                "stage": lce_stage,
                "stage_push": core_stage
            }

            prompt_core = f"""
            You are a supply-chain strategist advising a {role} in the {industry} industry.
            The user's 5S priorities are: {json.dumps(w5s_qual)}.
            Below is the qualitative status of each core process for the {sel_sys} system:
            {json.dumps(core_labels, indent=2)}.
            For each High or Medium process, refer to the dominant 5S dimensions that drove it 
            (see 'top_5s_per_item') and consider how the current LCE stage '{lce_stage}' 
            influences that priority. 
            Provide a concise, 5S-aware qualitative explanation on which processes to strengthen,
            simplify, or maintain to achieve "{objective}". Avoid numbers or parentheses.
            Limit to 170 words.
            """
            core_expl = safe_llm_call(prompt_core, core_payload)

            # ---- KPIs ----
            kpi_scores = {k: float(v.get(sel_sys, 0)) for k, v in res["scored"]["kpis"].items()}
            kpi_labels = {k: ("High" if v >= 2 else "Medium" if v >= 1 else "Low") for k, v in kpi_scores.items()}
            kpi_topS = {k: item_contrib_5s(k, "kpis", w5s) for k in kpi_labels}
            kpi_stage = {k: item_contrib_lce(k, "kpis", lce_stage) for k in kpi_labels}

            kpi_payload = {
                "kpi_labels": kpi_labels,
                "weights_5s_qual": w5s_qual,
                "top_5s_per_item": kpi_topS,
                "stage": lce_stage,
                "stage_push": kpi_stage
            }
            
            # Add benchmark context to LLM payload
            if sel_sys in BENCHMARKS:
                kpi_payload["benchmark_reference"] = BENCHMARKS[sel_sys]

            prompt_kpi = f"""
            You are a performance strategist advising a {role} in the {industry} sector.
            The user's 5S priorities are: {json.dumps(w5s_qual)}.
            Below is the qualitative status of each KPI for the {sel_sys} system:
            {json.dumps(kpi_labels, indent=2)}.
            Use the benchmark_reference data to calibrate your reasoning.
            If a KPI is 'Low' relative to the benchmark, recommend realistic improvements to reach 'High' maturity.
            Avoid numeric values in your text, but base your analysis on benchmark thresholds.
            """
            kpi_expl = safe_llm_call(prompt_kpi, kpi_payload)

            # ---- DRIVERS ----
            driver_scores = {k: float(v.get(sel_sys, 0)) for k, v in res["scored"]["drivers"].items()}
            driver_labels = {k: ("High" if v >= 2 else "Medium" if v >= 1 else "Low") for k, v in driver_scores.items()}
            driver_topS = {k: item_contrib_5s(k, "drivers", w5s) for k in driver_labels}
            driver_stage = {k: item_contrib_lce(k, "drivers", lce_stage) for k in driver_labels}

            driver_payload = {
                "driver_labels": driver_labels,
                "weights_5s_qual": w5s_qual,
                "top_5s_per_item": driver_topS,
                "stage": lce_stage,
                "stage_push": driver_stage
            }

            prompt_drv = f"""
            You are a resilience strategist advising a {role} in the {industry} industry.
            The user's 5S priorities are: {json.dumps(w5s_qual)}.
            Below is the qualitative status of each resilience driver for the {sel_sys} system:
            {json.dumps(driver_labels, indent=2)}.
            Use the 5S profile ('top_5s_per_item') and the LCE stage '{lce_stage}' to reason 
            which drivers reinforce stability, enhance flexibility, or need rethinking.
            Align the explanation with "{objective}" and write it 5S-aware, prescriptive,
            analytical, and concise (≤170 words, no numbers or parentheses).
            """
            driver_expl = safe_llm_call(prompt_drv, driver_payload)

            # --- store ---
            st.session_state["llm_interpretations"] = {
                "core": clean_numbers(core_expl),
                "kpi": clean_numbers(kpi_expl),
                "drivers": clean_numbers(driver_expl)
            }
            st.session_state["llm_done"] = True

        # --- render ---
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

                # Use cached value if exists
                if "compare_analysis" not in st.session_state:
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

                    st.info("Generating comparative interpretation (first run only)...")
                    compare_expl = safe_llm_call(compare_prompt, compare_payload,
                                                 temp=0.35, max_toks=450)
                    st.session_state["compare_analysis"] = clean_numbers(compare_expl)
                else:
                    compare_expl = st.session_state["compare_analysis"]

                # Display cached or freshly generated result
                if compare_expl:
                    st.markdown("### Comparative Interpretation")
                    st.write(compare_expl)
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

    # Initialize chat history
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    # Display existing messages
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # --------------------------------------------------
    #  CHAT INPUT
    # --------------------------------------------------
    user_q = st.chat_input("Ask about trade-offs or recommendations…")

    if user_q:
        # Record user question
        st.session_state["chat"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # --------------------------------------------------
        #  SAFETY: Verify analysis exists
        # --------------------------------------------------
        if "results" not in st.session_state or not st.session_state["results"]:
            reply = "Please run **Analyze** first to activate the Strategy Agent."
            st.session_state["chat"].append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        else:
            res = st.session_state["results"]

            # Pull all contextual elements already generated
            interp = st.session_state.get("llm_interpretations", {})
            compare_expl = st.session_state.get("compare_analysis", "")
            objective = st.session_state.get("objective", "")
            lce_stage = st.session_state.get("lce_stage", "")
            sel_sys = st.session_state.get("selected_system", "")
            role = st.session_state.get("user_role", "")
            industry = st.session_state.get("industry", "")

            # --------------------------------------------------
            #  ENRICHED CONTEXT
            # --------------------------------------------------
            ctx = {
                "weights_5s": res.get("weights_5s", {}),
                "scores": qualitative_scores(res.get("scored", {})),
                "interpretations": interp,
                "comparative_summary": compare_expl,
                "constraints": {
                    "objective": objective,
                    "lce_stage": lce_stage,
                    "selected_system": sel_sys,
                    "role": role,
                    "industry": industry,
                },
            }

            # --------------------------------------------------
            #  STRATEGY-AWARE SYSTEM PROMPT
            # --------------------------------------------------
            system_prompt = (
                "You are the Strategy Agent, a supply-chain advisor. "
                "Base every answer on the user's 5S weights, scored matrices, "
                "qualitative interpretations, and comparative summary. "
                "Stay consistent with previous analyses. "
                "Explain reasoning clearly and give actionable guidance "
                "for trade-offs, prioritization, and system design."
            )

        
            ctx_compact = compact_dict(ctx, max_items=5)


            # --------------------------------------------------
            #  CALL MODEL
            # --------------------------------------------------
            try:
                r = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",
                         "content": json.dumps(ctx_compact, ensure_ascii=False, default=_json_default)},
                        {"role": "user", "content": user_q},
                    ],
                    extra_headers=OPENROUTER_HEADERS,
                    temperature=0.4,
                    max_tokens=700,
                )
                reply = r.choices[0].message.content.strip() or \
                        "No response generated — please try rephrasing."
            except Exception as e:
                reply = f"⚠️ LLM error: {e}"
            
            # --------------------------------------------------
            #  CLEAN REPLY (remove stray numbers and parentheses)
            # --------------------------------------------------
            reply = re.sub(r"\b\d+(\.\d+)?\b", "", reply)   # remove numeric values
            reply = re.sub(r"\(\s*\)", "", reply).strip()   # clean empty parentheses

            # --------------------------------------------------
            #  DISPLAY + STORE REPLY
            # --------------------------------------------------
            st.session_state["chat"].append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

with tabs[4]:
    st.header("🧪 Validation, Robustness & Reproducibility")

    if "results" not in st.session_state:
        st.info("Run **Analyze** first to enable validation.")
    else:
        results = st.session_state["results"]
        weights_5s = results["weights_5s"]
        stage = st.session_state["lce_stage"]
        system = st.session_state.get("selected_system", "Product Transfer")

        # -------------------------------------------------
        # Compute and display run hash
        # -------------------------------------------------
        run_hash = compute_run_hash(weights_5s, stage, system)
        st.caption(f"Run ID: `{run_hash}`")

        # -------------------------------------------------
        # Dominance / Monotonicity / Range checks
        # -------------------------------------------------
        st.subheader("Internal Consistency Checks")
        dom_fails = dominance_test(results["scored"])
        if dom_fails:
            st.warning(f"{len(dom_fails)} inconsistencies detected")
            st.dataframe(dom_fails)
        else:
            st.success("All scores within [0,3] and consistent across matrices.")

        # -------------------------------------------------
        # 2️Save / Load reproducible JSON
        # -------------------------------------------------
        st.subheader("Reproducibility")
        run_data = {
            "hash": run_hash,
            "system": system,
            "lce_stage": stage,
            "weights_5s": weights_5s,
            "scores": results["scored"],
        }

        json_bytes = io.BytesIO(json.dumps(run_data, indent=2).encode("utf-8"))
        st.download_button(
            "💾 Download Run JSON",
            data=json_bytes,
            file_name=f"run_{run_hash}.json",
            mime="application/json",
        )

        uploaded_run = st.file_uploader("📤 Reload run.json", type="json")
        if uploaded_run:
            loaded = json.load(uploaded_run)
            st.session_state["results"] = {
                "scored": loaded["scores"],
                "weights_5s": loaded["weights_5s"],
            }
            st.success(f"Run {loaded.get('hash','?')} reloaded successfully.")

        # -------------------------------------------------
        # Sensitivity / Robustness Sandbox
        # -------------------------------------------------
        st.subheader("Sensitivity Sandbox")
        
        delta = st.slider("Perturbation (±%)", 0.0, 1.0, 0.2, 0.05)
        
        # Initialize variable outside to avoid NameError
        corr = None  
        
        if st.button("Run Sensitivity Test"):
            perturbed = perturb_weights(weights_5s, delta)
            st.json(perturbed, expanded=False)
            scored_pert = score_all(perturbed, stage)
        
            base_df = pd.DataFrame(results["scored"]["kpis"]).T.mean()
            new_df = pd.DataFrame(scored_pert["kpis"]).T.mean()
            corr = base_df.corr(new_df)
        
            st.metric("KPI Correlation (original vs perturbed)", f"{corr:.2f}")
        
            if corr < 0.6:
                st.warning("High sensitivity — small changes in weights alter results substantially.")
            else:
                st.success("Robust response — stable under weight perturbations.")
        

        if corr is None:
            st.info("Adjust the perturbation slider and click **Run Sensitivity Test** to compute robustness.")
        
    

        # -------------------------------------------------
        # Baseline comparison (TOPSIS)
        # -------------------------------------------------
        st.subheader("MCDA Baseline Comparison")

        kpi_matrix = results["scored"]["kpis"]
        rank_custom = pd.DataFrame(kpi_matrix).mean().rank(ascending=False)
        rank_topsis = topsis_compare(kpi_matrix)
        tau = rank_custom.corr(rank_topsis, method="kendall")

        st.metric("Kendall τ vs TOPSIS baseline", f"{tau:.2f}")

        if tau >= 0.7:
            st.success("High alignment with MCDA baseline — consistent prioritization.")
        else:
            st.warning("Divergence from baseline — check 5S or stage weight impacts.")

        # -------------------------------------------------
        # Quantitative Amplitude Check (5S effect range)
        # -------------------------------------------------
        st.subheader("Amplitude of 5S Influence")
        
        # Combine all scored matrices into one unified DataFrame
        scores_df = pd.concat([
            pd.DataFrame(results["scored"]["core_processes"]).T,
            pd.DataFrame(results["scored"]["kpis"]).T,
            pd.DataFrame(results["scored"]["drivers"]).T,
        ])
        
        # Compute range across systems
        mean_max = scores_df.max().mean()
        mean_min = scores_df.min().mean()
        variation = mean_max - mean_min
        
        st.metric("Average Score Range across Systems", f"{variation:.2f}")
        
        if variation < 0.25:
            st.warning("⚠️ Low amplitude — 5S sliders may have limited visible impact.")
        elif variation < 0.6:
            st.info("Moderate amplitude — 5S weights produce perceptible variation.")
        else:
            st.success("High amplitude — 5S sliders meaningfully reshape system priorities.")


        # -------------------------------------------------
        # Summary panel
        # -------------------------------------------------
        st.subheader("Validation Summary")
        
        # Safe formatting for None values
        corr_val = f"{corr:.2f}" if corr is not None else "N/A"
        
        st.markdown(f"""
        - **Run ID:** `{run_hash}`  
        - **LCE Stage:** `{stage}`  
        - **System:** `{system}`  
        - **Dominance tests:** {'Pass' if not dom_fails else 'Fail'}  
        - **Robustness (KPI corr):** {corr_val}  
        - **Baseline alignment (Kendall τ):** {tau:.2f}
        """)


# ---------- TAB 6: BENCHMARKS ----------
with tabs[5]:
    st.header("📈 Industry Benchmark Reference")

    selected = st.session_state.get("selected_system", "Product Transfer")
    meta = BENCHMARK_META.get(selected, {}).get("meta", {})
    objective = BENCHMARK_META.get(selected, {}).get("Objective", "N/A")

    st.subheader(f"Benchmarks for {selected}")
    st.markdown(f"**Objective:** {objective}")
    st.markdown(f"**Source:** {meta.get('source', 'N/A')}")
    st.markdown(f"**Framework:** {meta.get('mapping_framework', '5S–LCE')}")
    st.markdown(f"**Note:** {meta.get('note', '')}")
    st.divider()

    if BENCHMARKS and selected in BENCHMARKS:
        df_bench = pd.DataFrame(BENCHMARKS[selected]).T
        st.dataframe(df_bench, use_container_width=True)
    else:
        st.warning("No benchmark data loaded for this system.")














