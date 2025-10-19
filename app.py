# Authors: Dr. J. Isabel Méndez & Dr. Arturo Molina

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

# Models
ANALYZER_CHAIN = ["deepseek/deepseek-r1-0528:free","meta-llama/llama-3.3-70b-instruct:free"]
NARRATIVE_PRIMARY = "openai/gpt-oss-20b:free"
NARRATIVE_FALLBACK = "meta-llama/llama-3.3-70b-instruct:free"
CHAT_DEFAULT = "openai/gpt-oss-20b:free"
CHAT_CHOICES = ["openai/gpt-oss-20b:free","deepseek/deepseek-r1-0528:free","meta-llama/llama-3.3-70b-instruct:free"]

# =====================================================
#                CANONICAL VOCAB
# =====================================================
SYSTEMS = ["Product Transfer","Technology Transfer","Facility Design"]
LCE = ["Ideation","Basic Development","Advanced Development","Launch","Operation","End-of-Life"]
FIVE_S = ["Social","Sustainable","Sensing","Smart","Safe"]
PILLARS = ["quality","cost","volume","time","flexibility","environment"]

# Strategies
COMPETITIVE={"Product Transfer":"Operational Excellence","Technology Transfer":"Product Innovation","Facility Design":"Customer Focus/Intimacy"}
VALUE_CHAIN={"Product Transfer":"Collaboration Networks","Technology Transfer":"Strategic Business Units","Facility Design":"Vertical Integration"}
PROD_SERVICE={"Product Transfer":"Assemble to Order (ATO) + Catalogue of Services",
              "Technology Transfer":"Make/Build to Order (MTO/BTO) + Configuration of Services",
              "Facility Design":"Hybrid MTS + BTO + Design of Services"}

# =====================================================
#               BASELINES AND TAGS
# =====================================================
# (same BASE_CORE, BASE_KPIS, BASE_DRIVERS, KPI_TO_PILLARS, S_TAGS_KPI, etc. — keep your definitions intact)
# [PASTE ALL BASELINES AND TAG DICTS HERE — unchanged from your version]

# =====================================================
#                 ANALYSIS CORE
# =====================================================
def scenario_memberships(custom_tags: List[str]):
    tags=[t.strip().lower() for t in custom_tags if t.strip()]
    mem={k:0.0 for k in ["OTIF","Logistics lead time","Supplier on-time delivery","OEE","First-pass yield",
                          "Lifecycle cost","ESG index","Nearshoring","Network Diversification",
                          "Multisourcing","Inventory/Capacity Buffers","Platform/Plant Harmonization",
                          "Ecosystem Partnerships","SRM","Order Fulfillment","Co-Engineering"]}
    def add(k,v): mem[k]=max(mem[k],v)
    for t in tags:
        if "port" in t or "congestion" in t:
            add("Nearshoring",0.7); add("Network Diversification",0.5); add("OTIF",0.6)
        if "bankruptcy" in t or "single-source" in t:
            add("Multisourcing",0.8); add("Ecosystem Partnerships",0.6)
        if "tariff" in t: add("Nearshoring",0.6)
        if "raw material" in t: add("Inventory/Capacity Buffers",0.7)
        if "digital twin" in t: add("Platform/Plant Harmonization",0.6)
        if "lean logistics" in t: add("Logistics lead time",0.7)
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
    last=None
    for m in ANALYZER_CHAIN:
        try:
            r=client.chat.completions.create(model=m, messages=msgs, extra_headers=OPENROUTER_HEADERS,
                                             response_format={"type":"json_object"}, temperature=0.2, max_tokens=700)
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            last=e; time.sleep(0.2)
    raise RuntimeError(f"Analyzer failed: {last}")

# Guidance functions same as before (make_single_guidance, make_cross_guidance)
# -------------------------------------------------------------

# =====================================================
#                PERFORMANCE & METRICS
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
        try:
            results.append(analyze_to_pillars(payload)["pillars"])
        except:
            continue
    if len(results)<2:return None
    return pd.DataFrame(results).std().mean()

# =====================================================
#                SIDEBAR + MAIN LOGIC
# =====================================================
# (keep your full sidebar definitions + PRESETS exactly)
# After clicking "Analyze", replace that section with:

if st.button("Analyze", use_container_width=True, key="analyze_btn"):
    role_val = st.session_state.get("user_role_other") if st.session_state.get("user_role")=="Other" else st.session_state.get("user_role")
    weights_5s={s:st.session_state.get(f"s5_{s}",0.5) for s in FIVE_S}
    flags={"volatility":"High demand volatility" in st.session_state.get("preset_scenarios",[]),
           "geo_risk":"Geopolitical risk" in st.session_state.get("preset_scenarios",[]),
           "carbon":"Carbon constraints" in st.session_state.get("preset_scenarios",[])}
    custom=[t.strip() for t in st.session_state.get("custom_tags_text","").split(",") if t.strip()] + [c.lower() for c in st.session_state.get("other_chars",[])]

    payload={"objective":st.session_state.get("objective",""),"industry":st.session_state.get("industry",""),
             "user_role":role_val,"system_type":st.session_state.get("selected_system","Product Transfer"),
             "lce_stage":st.session_state.get("lce_stage","Operation"),"weights_5s":weights_5s,
             "scenarios":{"preset":st.session_state.get("preset_scenarios",[]),"custom":custom}}

    t0=time.time()
    analysis=analyze_to_pillars(payload)
    elapsed=round(time.time()-t0,2)

    raw=analysis.get("pillars",{}); ssum=sum(float(raw.get(k,0.0)) for k in PILLARS)
    pillars={k:(float(raw.get(k,0.0))/ssum if ssum>0 else 0.0) for k in PILLARS}
    reasons=analysis.get("reasons",[]); tags=analysis.get("tags",[])
    variability=llm_variability(payload)
    scored=score_all(weights_5s,st.session_state.get("lce_stage","Operation"),flags,custom+tags,pillars)
    df_stress=synthetic_stress(weights_5s,st.session_state.get("lce_stage","Operation"),custom+tags,pillars)
    radar_plot(pillars)

    # Guidance + keywords
    guidance_text=make_single_guidance({
        "objective":st.session_state.get("objective",""),"industry":st.session_state.get("industry",""),
        "user_role":role_val,"system_type":st.session_state.get("selected_system","Product Transfer"),
        "lce_stage":st.session_state.get("lce_stage","Operation"),"pillar_weights":pillars,
        "weights_5s":weights_5s,"scenarios":{"preset":st.session_state.get("preset_scenarios",[]),"custom":custom+tags},
        "strategies":{"Competitive":COMPETITIVE[st.session_state.get("selected_system","Product Transfer")],
                      "Value Chain":VALUE_CHAIN[st.session_state.get("selected_system","Product Transfer")],
                      "Product/Service":PROD_SERVICE[st.session_state.get("selected_system","Product Transfer")]},
        "selected_scores":{"core_processes":{k:float(v[st.session_state.get("selected_system","Product Transfer")]) for k,v in scored["core_processes"].items()},
                           "kpis":{k:float(v[st.session_state.get("selected_system","Product Transfer")]) for k,v in scored["kpis"].items()},
                           "drivers":{k:float(v[st.session_state.get("selected_system","Product Transfer")]) for k,v in scored["drivers"].items()}}
    })
    keywords=extract_keywords(guidance_text)

    st.session_state["metrics"]={"analysis_time_s":elapsed,"pillar_variability":variability,
                                 "top_keywords":keywords,"stress_results":df_stress.to_dict("records")}
    st.success(f"LLM completed in {elapsed}s | Pillar variability: {variability:.3f}")
    st.write("**Top guidance keywords:**",", ".join(keywords))
    st.write(guidance_text)

# =====================================================
#                FOOTER / EXPORT
# =====================================================
if "metrics" in st.session_state:
    if st.download_button("📊 Export metrics CSV",
        data=pd.DataFrame(st.session_state["metrics"]["stress_results"]).to_csv(index=False).encode(),
        file_name="stress_metrics.csv"): pass
st.caption("Scores ∈ [0,3]. 3=Core, 2=Strategic, 1=Relevant.")
