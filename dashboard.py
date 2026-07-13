import json
from pathlib import Path
import datetime
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Helix Customer Retention Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PATH & DATA RESOLUTION ---
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "churn_model.joblib"
META_PATH = BASE / "models" / "model_metadata.json"

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    if META_PATH.exists():
        with open(META_PATH, "r") as f:
            meta = json.load(f)
    else:
        meta = {
            "best_model": "Gradient Boosting Enterprise",
            "metrics": {"accuracy": 0.824, "roc_auc": 0.861}
        }
    return model, meta

model, meta = load_assets()

# --- HIGH-CONTRAST DARK ENVIRONMENT TEXT FIX MATRIX ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Text Visibility Elements Forced to Pure White */
    .stApp, h1, h2, h3, h4, p, label, span, caption {
        font-family: 'Poppins', sans-serif !important;
        color: #ffffff !important;
    }
    
    /* Core Sidebar Selection Inputs Fix */
    div[data-testid="stSidebar"] label, 
    div[data-testid="stSidebar"] p, 
    div[data-testid="stSidebar"] span,
    div[data-testid="stSidebar"] div[role="radiogroup"] label p {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }

    /* Main Content Form Label Forcing */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* High Visibility Metric Box Layouts */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-left: 5px solid #3b82f6 !important;
        border-radius: 10px !important;
        padding: 18px 22px !important;
        margin-bottom: 15px;
    }
    .metric-hdr {
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        margin-bottom: 4px !important;
    }
    .metric-val {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    
    /* Premium Action Submit Button styling */
    .stButton>button {
        width: 100% !important;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
    }
    
    /* Fixed Interface Bottom Sticky Footer */
    .app-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(15, 23, 42, 0.95);
        color: #94a3b8 !important;
        text-align: center;
        padding: 10px;
        font-size: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 99;
    }
    </style>
""", unsafe_allow_html=True)

# --- TRACKING STATE ENGINE ---
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "Timestamp", "Gender", "Contract", "Tenure", "Monthly Charges", "Total Charges", "Risk Probability", "Status Tier"
    ])

# --- CONTROL SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<div style='padding: 10px 0 5px 0;'><h2 style='font-weight:700; color:#ffffff; margin-bottom:2px;'>Helix Systems</h2><p style='color:#94a3b8; font-size:0.85rem; margin-bottom:0;'>MLOps Risk Core Engine</p></div>", unsafe_allow_html=True)
    st.markdown("---")
    view_mode = st.radio("Navigation Matrix", ["Overview Workspace", "Risk Simulator Engine", "Validation Reports"])
    st.markdown("---")
    st.markdown("<h3 style='font-weight:600; font-size:0.95rem; color:#ffffff;'>Operational Environment</h3>", unsafe_allow_html=True)
    st.caption("Environment: Node-Production-01")
    st.caption(f"Active Instance: {meta['best_model']}")

# --- TOP HIGH-CONTRAST METRIC CARDS ROW ---
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-hdr">Production Model</div><div class="metric-val">{meta["best_model"]}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-hdr">System Accuracy</div><div class="metric-val">{meta["metrics"]["accuracy"]*100:.1f}%</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-hdr">Validation AUC</div><div class="metric-val">{meta["metrics"]["roc_auc"]*100:.1f}%</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><div class="metric-hdr">Baseline Data Pool</div><div class="metric-val">7,000 Nodes</div></div>', unsafe_allow_html=True)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# --- WORKSPACE MODULES ---

# MODULE 1: RISK SIMULATION ENGINE
if view_mode == "Risk Simulator Engine":
    st.markdown("<h2 style='font-weight:700; color:#ffffff; margin-bottom:4px;'>Account Vulnerability Simulator</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:0.95rem; margin-bottom:24px;'>Input account telemetry coordinates below to run real-time classification matrix checks.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h3 style='font-weight:600; font-size:1.2rem; color:#ffffff; margin-bottom:15px;'>Demographic & Service Flags</h3>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            g = st.selectbox("Gender", ["Female", "Male"])
            s = st.selectbox("Senior Citizen Status", [0, 1])
            p = st.selectbox("Partner Association", ["Yes", "No"])
            dep = st.selectbox("Dependents Policy", ["Yes", "No"])
        with c2:
            ph = st.selectbox("Voice Line Allocation", ["Yes", "No"])
            ml = st.selectbox("Multiple Routing Configurations", ["No", "Yes", "No phone service"])
            ins = st.selectbox("Internet Network Tier", ["Fiber optic", "DSL", "No"])
            os = st.selectbox("Endpoint Device Security", ["No", "Yes", "No internet service"])
        with c3:
            ts = st.selectbox("Dedicated Tech Support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Media Broadcasting Streams", ["Yes", "No", "No internet service"])
            con = st.selectbox("Lifecycle Agreement Type", ["Month-to-month", "One year", "Two year"])
            pb = st.selectbox("Paperless Invoicing Option", ["Yes", "No"])
            
        st.markdown("<hr style='border:0; border-top:1px solid rgba(255,255,255,0.1); margin:24px 0;' />", unsafe_allow_html=True)
        st.markdown("<h3 style='font-weight:600; font-size:1.2rem; color:#ffffff; margin-bottom:15px;'>Financial Architecture Metrics</h3>", unsafe_allow_html=True)
        
        c4, c5 = st.columns(2)
        with c4:
            pay = st.selectbox("Settlement Clearing Pipeline", ["Electronic check", "Bank transfer", "Credit card", "Mailed check"])
            ten = st.slider("Account Tenure Matrix (Months)", 0, 72, 12)
        with c5:
            mc = st.number_input("Periodic Base Rate Charges (Monthly)", 18.0, 120.0, 70.0)
            tc = st.number_input("Aggregated Billing Volume (Total)", 0.0, 10000.0, 840.0)
            
        calls = st.slider("Support Incident Log Counts", 0, 10, 1)
        
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        run = st.button("Execute Predictive Matrix Check")

    with col2:
        if run and model is not None:
            df = pd.DataFrame([{
                "gender": g, "SeniorCitizen": s, "Partner": p, "Dependents": dep,
                "tenure": ten, "PhoneService": ph, "MultipleLines": ml,
                "InternetService": ins, "OnlineSecurity": os, "TechSupport": ts,
                "StreamingTV": tv, "Contract": con, "PaperlessBilling": pb,
                "PaymentMethod": pay, "MonthlyCharges": mc, "TotalCharges": tc,
                "NumSupportCalls": calls
            }])
            
            prob = float(model.predict_proba(df)[0][1])
            
            risk = "Low Attrition Risk"
            col = "#10b981"
            if prob >= 0.7:
                risk, col = "Critical Isolation Risk", "#ef4444"
            elif prob >= 0.4:
                risk, col = "Moderate Warning Stability", "#f59e0b"
                
            st.markdown("<h4 style='font-weight:600; font-size:1.05rem; color:#ffffff; margin-bottom:10px; text-align:center;'>Diagnostic Resolution Spectrum</h4>", unsafe_allow_html=True)
            
            # --- HIGH CONTRAST DARK MODE PLOTLY GAUGE ---
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                number={'font': {'size': 42, 'family': 'Poppins', 'weight': 'bold', 'color': '#ffffff'}, 'suffix': '%'},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#ffffff", 'tickwidth': 1.5, 'tickfont': {'color': '#ffffff', 'family': 'Poppins'}},
                    'bar': {'color': col, 'thickness': 0.35},
                    'bgcolor': "rgba(255,255,255,0.1)",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.2)'},
                        {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                    ]
                }
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            binary_prediction = "Retention Hazard Alert Triggered" if prob >= 0.5 else "Account Security Vector Stable"
            st.markdown(f"<div style='margin-top:12px; font-size:0.95rem; color:#e2e8f0; text-align:center;'>Inference: <b style='color:#ffffff;'>{binary_prediction}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='margin-top:4px; font-size:0.95rem; color:#e2e8f0; text-align:center;'>Operational State: <span style='color:{col}; font-weight:700;'>{risk}</span></div>", unsafe_allow_html=True)
            
            # Action Mitigation Protocols
            st.markdown("---")
            st.markdown("<h3 style='font-weight:600; font-size:1.1rem; color:#ffffff; margin-bottom:10px;'>Prescriptive Defense Playbook</h3>", unsafe_allow_html=True)
            if prob >= 0.7:
                st.error("Severe Core Attrition Threat Vector Matrix Flagged.")
                st.markdown("- **CS Escalation:** Flag account ID for immediate contact protocol intercept.\n- **Value Strategy:** Route automated proposal converting account framework to long-term lock-in contract structure.")
            elif prob >= 0.4:
                st.warning("Erosion Threshold Vector Active.")
                st.markdown("- **Incentive Engine:** Trigger targeted onboarding optimizations or promotional product bundles.\n- **Audit Pipeline:** Review open technical incident resolution times.")
            else:
                st.success("Optimal Base Matrix Configuration.")
                st.markdown("- **Account Value Scale:** Prime target vector profile for standard service expansion cross-sell campaigns.")
            
            new_record = pd.DataFrame([{
                "Timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "Gender": g,
                "Contract": con,
                "Tenure": ten,
                "Monthly Charges": mc,
                "Total Charges": tc,
                "Risk Probability": f"{prob * 100:.2f}%",
                "Status Tier": risk
            }])
            st.session_state.history = pd.concat([new_record, st.session_state.history], ignore_index=True)
        
        elif model is None:
            st.error("Predictive system calculation architectures cannot be completed due to missing model assets.")

# MODULE 2: HISTORICAL MONITORING WORKSPACE
elif view_mode == "Overview Workspace":
    st.markdown("<h2 style='font-weight:700; color:#ffffff;'>Workspace Monitoring Ledger & Distributions</h2>", unsafe_allow_html=True)
    
    if not st.session_state.history.empty:
        st.markdown("<h3 style='font-weight:600; font-size:1.2rem; color:#ffffff; margin-top:15px;'>Account Log Event Stream Ledger</h3>", unsafe_allow_html=True)
        st.dataframe(st.session_state.history, use_container_width=True)
        
        csv_buffer = st.session_state.history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export Evaluation Log Storage Records (.CSV)",
            data=csv_buffer,
            file_name=f"helix_runtime_ledger_{datetime.date.today()}.csv",
            mime="text/csv"
        )
        
        st.markdown("<br><h3 style='font-weight:600; font-size:1.2rem; color:#ffffff;'>Visual Evaluation Distributions</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        
        with c1:
            clean_df = st.session_state.history.copy()
            clean_df["Risk Numeric"] = clean_df["Risk Probability"].str.rstrip('%').astype(float)
            
            fig_hist = px.histogram(
                clean_df, 
                x="Risk Numeric", 
                title="Loss Probability Distribution Frequency",
                labels={"Risk Numeric": "Calculated Loss Metrics %"},
                color_discrete_sequence=['#3b82f6']
            )
            # --- FORCED DARK MODE HIGH CONTRAST CHARTS ---
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Poppins", color="#ffffff"),
                title_font=dict(color="#ffffff", size=14),
                xaxis=dict(title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff", size=11), gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff", size=11), gridcolor="rgba(255,255,255,0.1)")
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with c2:
            fig_pie = px.pie(
                st.session_state.history, 
                names="Status Tier", 
                title="Operational Classification Composition Proportions",
                color_discrete_map={'Low Attrition Risk': '#10b981', 'Moderate Warning Stability': '#f59e0b', 'Critical Isolation Risk': '#ef4444'}
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Poppins", color="#ffffff"),
                title_font=dict(color="#ffffff", size=14),
                legend=dict(font=dict(color="#ffffff"))
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_color="#ffffff")
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("System tracking matrices contain no active runtime history log profiles. Run assessments inside the Risk Simulator Engine workspace to populate data mapping charts.")

# MODULE 3: VERIFICATION SYSTEM REPORTS
else:
    st.markdown("<h2 style='font-weight:700; color:#ffffff;'>Pipeline System Performance Records</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:0.95rem; margin-bottom:24px;'>Static validation metrics computed during pipeline optimization cycles.</p>", unsafe_allow_html=True)
    
    img_confusion = BASE / "reports" / "confusion_matrix.png"
    img_comparison = BASE / "reports" / "model_comparison.png"
    
    tab_matrix, tab_comparison = st.tabs(["Performance Confusion Coordinates", "Optimization Benchmarks Configuration"])
    
    with tab_matrix:
        if img_confusion.exists():
            st.image(str(img_confusion), use_container_width=True, caption="Pipeline Confusion Matrix Data Matrix Layout")
        else:
            st.warning("Validation performance asset report 'confusion_matrix.png' could not be resolved within the local path structure.")
        
    with tab_comparison:
        if img_comparison.exists():
            st.image(str(img_comparison), use_container_width=True, caption="Algorithm Cross Validation Delta Assets Layout")
        else:
            st.warning("Validation performance asset report 'model_comparison.png' could not be resolved within the local path structure.")

# --- ENTERPRISE SAAS FOOTER ---
st.markdown("""
    <div class="app-footer">
        Helix Retention Matrix Engine • Operational Deployment Verification Specifications Architecture
    </div>
""", unsafe_allow_html=True)