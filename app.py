import os
import joblib
import pandas as pd
import streamlit as st
from train import train_and_save

# Page configuration
st.set_page_config(
    page_title="Blood Pressure Predictor | ML Health App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .bp-card {
        padding: 1.2rem;
        border-radius: 12px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .bp-value {
        font-size: 2.4rem;
        font-weight: 800;
    }
    .bp-label {
        font-size: 0.95rem;
        color: #475569;
        font-weight: 600;
    }
    .category-banner {
        padding: 1rem 1.25rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.15rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    if not os.path.exists(model_path):
        with st.spinner("Training model for initial deployment..."):
            train_and_save()
    artifact = joblib.load(model_path)
    return artifact

def classify_blood_pressure(sbp, dbp):
    """Classifies blood pressure based on American Heart Association (AHA) guidelines."""
    if sbp > 180 or dbp > 120:
        return {
            "category": "Hypertensive Crisis",
            "color": "#DC2626",
            "bg": "#FEE2E2",
            "desc": "Emergency medical attention required! Systolic > 180 or Diastolic > 120 mmHg."
        }
    elif sbp >= 140 or dbp >= 90:
        return {
            "category": "Hypertension Stage 2",
            "color": "#EA580C",
            "bg": "#FFEDD5",
            "desc": "High blood pressure stage 2. Medical treatment and lifestyle modification recommended."
        }
    elif (130 <= sbp <= 139) or (80 <= dbp <= 89):
        return {
            "category": "Hypertension Stage 1",
            "color": "#D97706",
            "bg": "#FEF3C7",
            "desc": "High blood pressure stage 1. Lifestyle changes and regular monitoring advised."
        }
    elif (120 <= sbp <= 129) and (dbp < 80):
        return {
            "category": "Elevated Blood Pressure",
            "color": "#CA8A04",
            "bg": "#FEF9C3",
            "desc": "Elevated blood pressure. Likely to develop into hypertension without lifestyle changes."
        }
    else:
        return {
            "category": "Normal Blood Pressure",
            "color": "#16A34A",
            "bg": "#DCFCE7",
            "desc": "Blood pressure is in optimal range (Systolic < 120 and Diastolic < 80 mmHg)."
        }

def get_recommendations(user_inputs, sbp, dbp):
    recs = []
    if user_inputs["BMI"] >= 25.0:
        recs.append("🥗 **Weight Management**: Your BMI is elevated. Aiming for a healthy weight can reduce SBP by 5-10 mmHg.")
    if user_inputs["Smoking_Status"] == "Smoker":
        recs.append("🚭 **Smoking Cessation**: Quitting smoking immediately lowers cardiovascular strain and stabilizes vascular elasticity.")
    if user_inputs["Salt_Intake"] == "High":
        recs.append("🧂 **Reduce Sodium**: Limit daily sodium to under 2,000 mg (DASH diet) to significantly lower systolic blood pressure.")
    if user_inputs["Daily_Steps"] < 5000:
        recs.append("🚶 **Physical Activity**: Increase walking to 7,000-10,000 steps/day for sustained aerobic benefit.")
    if user_inputs["Sleep_Hours"] < 6.5:
        recs.append("😴 **Sleep Routine**: Chronically sleeping under 7 hours increases sympathetic nerve activity and blood pressure.")
    if user_inputs["Stress_Level"] >= 7:
        recs.append("🧘 **Stress Reduction**: Practice mindfulness, deep breathing, or yoga to mitigate chronic cortisol spikes.")
    if not recs:
        recs.append("🌟 **Maintain Habits**: Your lifestyle habits are well-aligned with cardiovascular health guidelines.")
    return recs

def main():
    artifact = get_model()
    pipeline = artifact["pipeline"]
    test_metrics = artifact.get("test_metrics", {})

    st.markdown('<div class="main-header">🩺 Blood Pressure AI Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Estimate Systolic and Diastolic Blood Pressure using clinical biometrics and lifestyle indicators.</div>',
        unsafe_allow_html=True
    )

    # Sidebar: About & Model Metrics
    with st.sidebar:
        st.subheader("📊 Model Performance")
        if test_metrics:
            sbp_mae = test_metrics.get("Systolic_BP", {}).get("MAE", 0)
            dbp_mae = test_metrics.get("Diastolic_BP", {}).get("MAE", 0)
            st.metric(label="Systolic MAE", value=f"±{sbp_mae:.2f} mmHg")
            st.metric(label="Diastolic MAE", value=f"±{dbp_mae:.2f} mmHg")
            st.caption(f"Systolic R²: {test_metrics.get('Systolic_BP', {}).get('R2', 0):.2f} | Diastolic R²: {test_metrics.get('Diastolic_BP', {}).get('R2', 0):.2f}")
        
        st.markdown("---")
        st.subheader("💡 Cloud Ready")
        st.markdown("""
        - **Model**: Multi-target Random Forest
        - **Pipeline**: Strict One-Hot & Scaling Transformer
        - **Guidelines**: American Heart Association (AHA)
        - **Deployable**: Streamlit Cloud, Render, Hugging Face, GCP
        """)

    # Main Input Form
    with st.form("bp_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 👤 Patient Biometrics")
            age = st.slider("Age (years)", min_value=18, max_value=85, value=35)
            gender = st.selectbox("Biological Sex", options=["Male", "Female"])
            bmi = st.number_input("Body Mass Index (BMI)", min_value=15.0, max_value=50.0, value=25.2, step=0.1)
            heart_rate = st.slider("Resting Heart Rate (bpm)", min_value=45, max_value=120, value=72)
            stress_level = st.select_slider("Daily Stress Level (1-10)", options=list(range(1, 11)), value=5)

        with col2:
            st.markdown("### 🏃 Lifestyle & Habits")
            daily_steps = st.number_input("Average Daily Steps", min_value=500, max_value=25000, value=6500, step=500)
            sleep_hours = st.slider("Sleep Duration (hours/night)", min_value=4.0, max_value=11.0, value=7.0, step=0.5)
            smoking_status = st.selectbox("Smoking Status", options=["Non-Smoker", "Smoker"])
            alcohol_intake = st.selectbox("Alcohol Consumption", options=["None", "Moderate", "Heavy"])
            salt_intake = st.selectbox("Daily Dietary Salt / Sodium", options=["Low", "Medium", "High"])

        submitted = st.form_submit_button("⚡ Predict Blood Pressure", use_container_width=True)

    if submitted:
        input_data = {
            "Age": age,
            "Gender": gender,
            "BMI": bmi,
            "Heart_Rate": heart_rate,
            "Daily_Steps": daily_steps,
            "Sleep_Hours": sleep_hours,
            "Stress_Level": stress_level,
            "Smoking_Status": smoking_status,
            "Alcohol_Intake": alcohol_intake,
            "Salt_Intake": salt_intake
        }

        input_df = pd.DataFrame([input_data])
        predictions = pipeline.predict(input_df)[0]
        sbp_pred = round(float(predictions[0]), 1)
        dbp_pred = round(float(predictions[1]), 1)

        classification = classify_blood_pressure(sbp_pred, dbp_pred)

        st.markdown("---")
        st.subheader("🎯 Prediction Results")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown(f"""
            <div class="bp-card">
                <div class="bp-label">SYSTOLIC (SBP)</div>
                <div class="bp-value" style="color: {classification['color']};">{sbp_pred} <span style="font-size: 1.1rem; font-weight: 500;">mmHg</span></div>
                <div style="font-size: 0.85rem; color: #64748B;">Pressure during heartbeat</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div class="bp-card">
                <div class="bp-label">DIASTOLIC (DBP)</div>
                <div class="bp-value" style="color: {classification['color']};">{dbp_pred} <span style="font-size: 1.1rem; font-weight: 500;">mmHg</span></div>
                <div style="font-size: 0.85rem; color: #64748B;">Pressure between heartbeats</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="category-banner" style="background-color: {classification['bg']}; color: {classification['color']};">
            🩺 Classification: <strong>{classification['category']}</strong><br>
            <span style="font-size: 0.95rem; font-weight: 400;">{classification['desc']}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🌿 Personalized Lifestyle Guidance")
        recs = get_recommendations(input_data, sbp_pred, dbp_pred)
        for r in recs:
            st.markdown(f"- {r}")

if __name__ == "__main__":
    main()
