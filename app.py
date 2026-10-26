import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="No-Show Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f1f5f9 !important;
    }

    .block-container {
        max-width: 960px !important;
        padding-top: 1.25rem !important;
        padding-bottom: 3rem !important;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stToolbar"] {display: none;}

    /* ---- Top bar ---- */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .brand-icon {
        width: 40px;
        height: 40px;
        background: #14b8a6;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.15rem;
    }
    .brand-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
        margin: 0;
    }
    .brand-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin: 0;
    }
    .mode-pills {
        display: flex;
        gap: 0.5rem;
    }
    .pill {
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid #e2e8f0;
        background: #ffffff;
        color: #64748b;
    }
    .pill.active {
        background: #ccfbf1;
        border-color: #5eead4;
        color: #0f766e;
    }

    /* ---- Hero ---- */
    .hero {
        text-align: center;
        margin-bottom: 1.75rem;
    }
    .hero-eyebrow {
        color: #14b8a6;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }
    .hero h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        margin: 0 0 0.5rem 0 !important;
        letter-spacing: -0.02em;
    }
    .hero-desc {
        color: #64748b;
        font-size: 0.95rem;
        margin: 0;
    }

    /* ---- Form card ---- */
    div[data-testid="stForm"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 1.75rem 1.75rem 1rem 1.75rem !important;
        box-shadow: 0 4px 24px rgba(15, 23, 42, 0.06);
    }

    /* Force dark readable text everywhere in form */
    div[data-testid="stForm"] label,
    div[data-testid="stForm"] p,
    div[data-testid="stForm"] span,
    div[data-testid="stForm"] .stMarkdown,
    div[data-testid="stForm"] [data-testid="stWidgetLabel"] p,
    div[data-testid="stForm"] [data-testid="stWidgetLabel"] span,
    div[data-testid="stForm"] [data-testid="stWidgetLabel"] label {
        color: #1e293b !important;
        font-weight: 500 !important;
    }

    /* Slider value in teal */
    div[data-testid="stForm"] [data-testid="stTickBarMin"],
    div[data-testid="stForm"] [data-testid="stTickBarMax"] {
        color: #94a3b8 !important;
    }
    div[data-testid="stForm"] [data-testid="stSliderTickBar"] ~ div,
    div[data-testid="stForm"] .stSlider label span[data-testid="stMarkdownContainer"] p {
        color: #1e293b !important;
    }

    /* Select / input fields — white with soft border */
    div[data-testid="stForm"] [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        min-height: 42px;
        color: #0f172a !important;
    }
    div[data-testid="stForm"] [data-baseweb="select"] span,
    div[data-testid="stForm"] [data-baseweb="select"] div {
        color: #0f172a !important;
    }
    div[data-testid="stForm"] input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 10px !important;
    }
    div[data-testid="stForm"] [data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stForm"] [data-baseweb="input"] input {
        color: #0f172a !important;
    }

    /* Teal slider track */
    div[data-testid="stForm"] div[data-baseweb="slider"] div[role="slider"] {
        background-color: #ffffff !important;
        border: 2px solid #14b8a6 !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background-color: #14b8a6 !important;
    }

    /* Predict button */
    .stFormSubmitButton > button {
        background: #14b8a6 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.7rem 1rem !important;
        margin-top: 0.5rem;
    }
    .stFormSubmitButton > button:hover {
        background: #0d9488 !important;
        color: #ffffff !important;
    }
    .stFormSubmitButton > button p,
    .stFormSubmitButton > button span {
        color: #ffffff !important;
    }

    /* Result card */
    .result-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.25rem;
        box-shadow: 0 4px 24px rgba(15, 23, 42, 0.06);
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/no_show_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# ---------------- TOP BAR ----------------
st.markdown("""
<div class="top-bar">
  <div class="brand">
    <div class="brand-icon">📁</div>
    <div>
      <p class="brand-title">No-Show Risk Predictor</p>
      <p class="brand-sub">Healthcare Appointment Analytics</p>
    </div>
  </div>
  <div class="mode-pills">
    <span class="pill">Heuristic Mode</span>
    <span class="pill active">✨ AI-Powered</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">🩺 Healthcare Appointment No-Show Risk Predictor</div>
  <h1>Predict Patient No-Show Risk</h1>
  <p class="hero-desc">Predict whether a patient is likely to miss a scheduled medical appointment.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT FORM ----------------
with st.form("prediction_form"):
    c1, c2 = st.columns(2, gap="large")

    with c1:
        age = st.slider("Age", 0, 100, 30)
        gender = st.selectbox("Gender", ["Female", "Male"])
        scholarship = st.selectbox(
            "Scholarship", [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )
        hypertension = st.selectbox(
            "Hypertension", [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )
        diabetes = st.selectbox(
            "Diabetes", [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )
        appointment_month = st.selectbox(
            "Appointment Month",
            list(range(1, 13)),
            index=4
        )

    with c2:
        alcoholism = st.selectbox(
            "Alcoholism", [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )
        handicap = st.selectbox("Handicap Level", [0, 1, 2, 3])
        sms = st.selectbox(
            "SMS Reminder Received", [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )
        waiting_days = st.slider("Waiting Days", 0, 30, 5)
        appointment_day = st.selectbox(
            "Appointment Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        )
        appointment_hour = st.slider("Appointment Booking Hour", 0, 23, 10)

    neighbourhood = st.text_input(
        "Neighbourhood",
        "JARDIM CAMBURI",
        help="Enter the neighbourhood name exactly as in the dataset."
    )

    submitted = st.form_submit_button("Predict No-Show Risk", use_container_width=True)

# ---------------- DAY MAPPING ----------------
day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5
}

# ---------------- PREDICT ----------------
if submitted:
    appointment_day_num = day_map[appointment_day]

    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0

    input_df["Age"] = age
    input_df["Scholarship"] = scholarship
    input_df["Hypertension"] = hypertension
    input_df["Diabetes"] = diabetes
    input_df["Alcoholism"] = alcoholism
    input_df["Handicap"] = handicap
    input_df["SMS_received"] = sms
    input_df["WaitingDays"] = waiting_days
    input_df["Appointment_DayOfWeek"] = appointment_day_num
    input_df["Appointment_Month"] = appointment_month
    input_df["Appointment_Hour"] = appointment_hour

    if "Gender_M" in input_df.columns and gender == "Male":
        input_df["Gender_M"] = 1

    neigh_col = f"Neighbourhood_{neighbourhood.upper()}"
    if neigh_col in input_df.columns:
        input_df[neigh_col] = 1

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    if probability < 30:
        risk_label = "Low Risk"
        risk_color = "#059669"
        tip = "Patient is likely to attend. A normal reminder is sufficient."
    elif probability < 60:
        risk_label = "Medium Risk"
        risk_color = "#d97706"
        tip = "Send an SMS reminder 24 hours before the appointment."
    else:
        risk_label = "High Risk"
        risk_color = "#dc2626"
        tip = "Send SMS + phone call reminder and consider follow-up scheduling."

    st.markdown(f"""
    <div class="result-card">
      <p style="margin:0 0 0.35rem 0; font-size:0.8rem; color:#64748b; font-weight:600; letter-spacing:0.04em; text-transform:uppercase;">Prediction Result</p>
      <p style="margin:0 0 0.5rem 0; font-size:1.35rem; font-weight:700; color:{risk_color};">{risk_label} — {probability:.1f}%</p>
      <p style="margin:0 0 1rem 0; color:#475569; font-size:0.92rem;">{tip}</p>
    </div>
    """, unsafe_allow_html=True)

    st.progress(min(int(probability), 100))

    # ---------------- PREDICTION SUMMARY ----------------
    st.subheader("Prediction Summary — Why this result?")

    if probability < 30:
        st.info(
            f"The model estimates a {probability:.1f}% probability of a no-show, "
            "which falls in this app's Low Risk band. Based on this model score, "
            "the appointment is less likely to be missed than appointments receiving higher scores."
        )
    elif probability < 60:
        st.warning(
            f"The model estimates a {probability:.1f}% probability of a no-show, "
            "which falls in this app's Medium Risk band. A reminder and attendance confirmation may be useful."
        )
    else:
        st.error(
            f"The model estimates a {probability:.1f}% probability of a no-show, "
            "which falls in this app's High Risk band. Consider a reminder and a brief confirmation call."
        )

    st.caption(
        "This is a model estimate, not a certainty. Risk bands are app-defined thresholds, "
        "and the score should support—not replace—staff judgement."
    )

    # ---------------- EXPLAINABLE AI (SHAP) ----------------
    st.subheader("Explainable AI — Main factors behind the prediction")
    st.write(
        "The explanation below uses this patient's input and the model's SHAP values. "
        "Positive contributions push the model output toward the no-show class; "
        "negative contributions push it away. These are associations learned by the model, not proof of cause."
    )

    def readable_feature_name(name):
        known_names = {
            "Age": "Patient age",
            "Gender_M": "Gender: Male",
            "Scholarship": "Scholarship status",
            "Hypertension": "Hypertension",
            "Diabetes": "Diabetes",
            "Alcoholism": "Alcoholism",
            "Handicap": "Handicap level",
            "SMS_received": "SMS reminder received",
            "WaitingDays": "Days between scheduling and appointment",
            "Appointment_DayOfWeek": "Appointment day of week",
            "Appointment_Month": "Appointment month",
            "Appointment_Hour": "Booking hour",
        }
        if name in known_names:
            return known_names[name]
        if name.startswith("Neighbourhood_"):
            return "Neighbourhood: " + name.replace("Neighbourhood_", "").replace("_", " ").title()
        return name.replace("_", " ").title()

    def readable_feature_value(name, row):
        if name == "Gender_M":
            return "Male" if gender == "Male" else "Female"
        if name == "Scholarship":
            return "Yes" if scholarship else "No"
        if name == "Hypertension":
            return "Yes" if hypertension else "No"
        if name == "Diabetes":
            return "Yes" if diabetes else "No"
        if name == "Alcoholism":
            return "Yes" if alcoholism else "No"
        if name == "SMS_received":
            return "Yes" if sms else "No"
        if name == "WaitingDays":
            return f"{waiting_days} day(s)"
        if name == "Appointment_DayOfWeek":
            return appointment_day
        if name == "Appointment_Month":
            return str(appointment_month)
        if name == "Appointment_Hour":
            return f"{appointment_hour}:00"
        if name == "Age":
            return str(age)
        if name == "Handicap":
            return str(handicap)
        if name.startswith("Neighbourhood_"):
            return "Selected" if row[name] == 1 else "Not selected"
        return str(row[name])

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        # Support SHAP's list output and newer 2-D/3-D output formats.
        if isinstance(shap_values, list):
            class_index = list(getattr(model, "classes_", [0, 1])).index(1)
            contributions = np.asarray(shap_values[class_index])[0].reshape(-1)
        else:
            values = np.asarray(shap_values)
            if values.ndim == 3:
                # Newer SHAP versions usually return (rows, features, classes).
                if values.shape[0] == 1 and values.shape[-1] > 1:
                    class_index = list(getattr(model, "classes_", [0, 1])).index(1)
                    contributions = values[0, :, class_index]
                # Some versions return (classes, rows, features).
                elif values.shape[0] > 1 and values.shape[1] == 1:
                    class_index = list(getattr(model, "classes_", [0, 1])).index(1)
                    contributions = values[class_index, 0, :]
                else:
                    raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
            elif values.ndim == 2:
                contributions = values[0]
            else:
                raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

        contributions = np.asarray(contributions).reshape(-1)
        if len(contributions) != len(input_df.columns):
            raise ValueError(
                f"SHAP returned {len(contributions)} contributions for {len(input_df.columns)} features."
            )

        explanation_df = pd.DataFrame({
            "Feature": input_df.columns,
            "Feature name": [readable_feature_name(col) for col in input_df.columns],
            "Patient value": [readable_feature_value(col, input_df.iloc[0]) for col in input_df.columns],
            "SHAP contribution": contributions.astype(float),
        })
        explanation_df["Absolute impact"] = explanation_df["SHAP contribution"].abs()
        explanation_df = explanation_df.sort_values("Absolute impact", ascending=False)
        meaningful_df = explanation_df[explanation_df["Absolute impact"] > 1e-8]
        top_df = (meaningful_df if not meaningful_df.empty else explanation_df).head(8).copy()

        toward = top_df[top_df["SHAP contribution"] > 0].sort_values(
            "SHAP contribution", ascending=False
        )
        away = top_df[top_df["SHAP contribution"] < 0].sort_values(
            "SHAP contribution", ascending=True
        )

        st.markdown("#### What pushed the prediction toward a no-show?")
        if not toward.empty:
            for _, item in toward.head(3).iterrows():
                st.markdown(
                    f"- **{item['Feature name']}** (patient value: {item['Patient value']}) "
                    "pushed the model output toward the no-show class."
                )
        else:
            st.write("No strong positive SHAP contributions were identified among the leading features.")

        st.markdown("#### What pushed the prediction away from a no-show?")
        if not away.empty:
            for _, item in away.head(3).iterrows():
                st.markdown(
                    f"- **{item['Feature name']}** (patient value: {item['Patient value']}) "
                    "pushed the model output away from the no-show class."
                )
        else:
            st.write("No strong negative SHAP contributions were identified among the leading features.")

        st.markdown("#### Feature contribution chart")
        chart_df = top_df.sort_values("SHAP contribution")
        st.bar_chart(chart_df.set_index("Feature name")["SHAP contribution"])
        st.dataframe(
            top_df[["Feature name", "Patient value", "SHAP contribution"]].rename(columns={
                "Feature name": "Feature",
                "Patient value": "Patient's input",
            }),
            hide_index=True,
            use_container_width=True
        )
    except Exception as e:
        st.warning(
            "The prediction was generated, but the feature-level explanation could not be displayed. "
            f"Technical detail: {e}"
        )
        st.info(
            "You can still use the risk score and the general recommendation above. "
            "The app does not guess specific reasons when SHAP is unavailable."
        )

    with st.expander("Patient summary"):
        st.dataframe(
            pd.DataFrame({
                "Feature": [
                    "Age", "Gender", "Scholarship", "Hypertension",
                    "Diabetes", "Alcoholism", "Handicap",
                    "SMS Received", "Waiting Days", "Neighbourhood"
                ],
                "Value": [
                    age, gender, scholarship, hypertension,
                    diabetes, alcoholism, handicap,
                    sms, waiting_days, neighbourhood.upper()
                ]
            }),
            hide_index=True,
            use_container_width=True
        )
