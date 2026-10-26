
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "medical_appointments_2017.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "carepulse_model.pkl"
METRICS_PATH = MODEL_DIR / "carepulse_metrics.txt"

NUMERIC_FEATURES = [
    "Age", "Scholarship", "Hypertension", "Diabetes",
    "Alcoholism", "Handicap", "SMS_received", "WaitingDays",
    "Appointment_DayOfWeek", "Appointment_Month", "Appointment_Hour"
]

CATEGORICAL_FEATURES = ["Gender", "Neighbourhood"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

st.set_page_config(
    page_title="CarePulse | Appointment Intelligence",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# MODERN FRONTEND
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background: #f3f6fb;
    color: #182d45;
}

.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #142b45;
}

[data-testid="stSidebar"] * {
    color: #f3f7fb;
}

.brand-card {
    background: white;
    padding: 18px 22px;
    border: 1px solid #e1e8f0;
    border-radius: 16px;
    margin-bottom: 20px;
}

.brand-name {
    color: #142b45;
    font-size: 1.45rem;
    font-weight: 800;
    margin: 0;
}

.brand-description {
    color: #718198;
    margin: 3px 0 0;
    font-size: .88rem;
}

.hero {
    background: linear-gradient(115deg, #142b45, #176b70);
    padding: 32px;
    border-radius: 22px;
    color: white;
    margin-bottom: 24px;
}

.hero-label {
    color: #78e4d4;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 2px;
    font-size: .75rem;
}

.hero h1 {
    color: white;
    font-size: 2.1rem;
    font-weight: 800;
    margin: 10px 0;
}

.hero p {
    color: #e0edf2;
    margin-bottom: 0;
}

.metric-card {
    background: white;
    padding: 20px;
    border: 1px solid #e1e8f0;
    border-radius: 16px;
    min-height: 125px;
}

.metric-label {
    color: #718198;
    font-size: .85rem;
    font-weight: 600;
}

.metric-value {
    color: #142b45;
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 8px;
}

.metric-note {
    color: #8391a5;
    font-size: .76rem;
    margin-top: 3px;
}

.panel {
    background: white;
    padding: 22px;
    border: 1px solid #e1e8f0;
    border-radius: 18px;
    margin: 12px 0;
}

.stButton > button,
.stFormSubmitButton > button {
    background: #0c988d;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 44px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #08786f;
    color: white;
}

[data-testid="stForm"] {
    background: white;
    border: 1px solid #e1e8f0;
    border-radius: 18px;
    padding: 22px;
}

footer {
    visibility: hidden;
}

@media (max-width: 700px) {
    .hero h1 {
        font-size: 1.55rem;
    }
    .hero {
        padding: 22px;
    }
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATA LOADING AND VALIDATION
# --------------------------------------------------

if not DATA_PATH.exists():
    st.error("Your dataset could not be found.")
    st.write(f"Expected location: `{DATA_PATH}`")
    st.info(
        "Create a folder named data beside app.py and put "
        "medical_appointments_2017.csv inside it."
    )
    st.stop()

try:
    raw = pd.read_csv(DATA_PATH)
except Exception as exc:
    st.error(f"Could not read your CSV: {exc}")
    st.stop()

REQUIRED_COLUMNS = [
    "Gender", "ScheduledDay", "AppointmentDay", "Age",
    "Neighbourhood", "Scholarship", "Hipertension",
    "Diabetes", "Alcoholism", "Handcap", "SMS_received",
    "No-show"
]

missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]

if missing:
    st.error(f"Required columns missing from CSV: {missing}")
    st.stop()

if raw.empty:
    st.error("Your CSV contains no appointment records.")
    st.stop()


# --------------------------------------------------
# FEATURE ENGINEERING
# The same function is used for training and prediction.
# --------------------------------------------------

def prepare_features(df):
    result = pd.DataFrame(index=df.index)

    scheduled = pd.to_datetime(
        df["ScheduledDay"], errors="coerce", utc=True
    )
    appointment = pd.to_datetime(
        df["AppointmentDay"], errors="coerce", utc=True
    )

    result["Age"] = pd.to_numeric(
        df["Age"], errors="coerce"
    ).clip(0, 120)

    result["Scholarship"] = pd.to_numeric(
        df["Scholarship"], errors="coerce"
    )

    result["Hypertension"] = pd.to_numeric(
        df["Hipertension"], errors="coerce"
    )

    result["Diabetes"] = pd.to_numeric(
        df["Diabetes"], errors="coerce"
    )

    result["Alcoholism"] = pd.to_numeric(
        df["Alcoholism"], errors="coerce"
    )

    result["Handicap"] = pd.to_numeric(
        df["Handcap"], errors="coerce"
    )

    result["SMS_received"] = pd.to_numeric(
        df["SMS_received"], errors="coerce"
    )

    result["WaitingDays"] = (
        (appointment - scheduled).dt.total_seconds() / 86400
    ).clip(0, 365)

    result["Appointment_DayOfWeek"] = appointment.dt.dayofweek
    result["Appointment_Month"] = appointment.dt.month
    result["Appointment_Hour"] = scheduled.dt.hour

    result["Gender"] = (
        df["Gender"].fillna("Unknown")
        .astype(str).str.strip().str.upper()
    )

    result["Neighbourhood"] = (
        df["Neighbourhood"].fillna("Unknown")
        .astype(str).str.strip().str.upper()
    )

    return result[FEATURES]


# --------------------------------------------------
# MODEL TRAINING
# Train only when our separate CarePulse model is absent.
# --------------------------------------------------

@st.cache_resource
def load_or_train_model(data_path, modified_time):
    df = pd.read_csv(data_path)

    features = prepare_features(df)

    target = (
        df["No-show"].astype(str).str.strip().str.lower()
        .map({"yes": 1, "no": 0})
    )

    scheduled = pd.to_datetime(
        df["ScheduledDay"], errors="coerce", utc=True
    )

    appointment = pd.to_datetime(
        df["AppointmentDay"], errors="coerce", utc=True
    )

    valid = (
        target.notna()
        & scheduled.notna()
        & appointment.notna()
    )

    X = features.loc[valid].reset_index(drop=True)
    y = target.loc[valid].astype(int).reset_index(drop=True)

    if len(X) < 20 or y.nunique() != 2:
        raise ValueError(
            "Not enough valid records or the target does not "
            "contain both No-show classes."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        try:
            saved = joblib.load(MODEL_PATH)
            if (
                isinstance(saved, dict)
                and saved.get("dataset_mtime") == modified_time
            ):
                return saved["model"], saved.get("metrics", {})
        except Exception:
            pass

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions))
    }

    try:
        metrics["roc_auc"] = float(
            roc_auc_score(y_test, probabilities)
        )
    except ValueError:
        metrics["roc_auc"] = None

    joblib.dump({
        "model": model,
        "dataset_mtime": modified_time,
        "metrics": metrics
    }, MODEL_PATH)

    report = (
        "CarePulse model evaluation\n"
        "Dataset: artificially generated; not clinical validation\n"
        f"Valid records: {metrics['rows']}\n"
        f"Training records: {metrics['train_rows']}\n"
        f"Test records: {metrics['test_rows']}\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"ROC-AUC: {metrics['roc_auc']}\n"
    )

    METRICS_PATH.write_text(report, encoding="utf-8")

    return model, metrics


# --------------------------------------------------
# APP HEADER AND NAVIGATION
# --------------------------------------------------

st.markdown("""
<div class="brand-card">
    <p class="brand-name">✚ CarePulse</p>
    <p class="brand-description">
        Appointment Intelligence · Healthcare Analytics
    </p>
</div>

<div class="hero">
    <div class="hero-label">Healthcare analytics platform</div>
    <h1>Understand attendance. Plan follow-ups.</h1>
    <p>
        Explore appointment patterns and estimate no-show risk
        using a model trained on your selected dataset.
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("# ✚ CarePulse")
st.sidebar.caption("Appointment Intelligence")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Predict Risk", "Model Evaluation", "About Data"]
)

st.sidebar.divider()
st.sidebar.caption("Development and testing application")

# --------------------------------------------------
# LOAD OUR OWN MODEL
# --------------------------------------------------

try:
    with st.spinner("Checking dataset and preparing the CarePulse model..."):
        model, metrics = load_or_train_model(
            str(DATA_PATH), DATA_PATH.stat().st_mtime
        )
except Exception as exc:
    st.error(f"Model preparation failed: {exc}")
    st.info(
        "Check that your CSV is valid and that requirements.txt "
        "has been installed."
    )
    st.stop()

# --------------------------------------------------
# DASHBOARD DATA
# --------------------------------------------------

outcomes = raw["No-show"].astype(str).str.strip().str.lower()

total = len(raw)
no_shows = int((outcomes == "yes").sum())
attended = int((outcomes == "no").sum())

valid_outcomes = no_shows + attended
no_show_rate = (
    no_shows / valid_outcomes * 100 if valid_outcomes else 0
)

dates = pd.to_datetime(
    raw["AppointmentDay"], errors="coerce", utc=True
)

# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":
    st.subheader("Performance overview")
    st.caption(
        "Statistics below are calculated from your own CSV file."
    )

    columns = st.columns(4)

    cards = [
        ("Total appointments", f"{total:,}", "Dataset records"),
        ("Attended", f"{attended:,}", "Recorded attendance"),
        ("No-shows", f"{no_shows:,}", "Recorded absences"),
        ("No-show rate", f"{no_show_rate:.1f}%", "Among known outcomes")
    ]

    for col, (label, value, note) in zip(columns, cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Attendance distribution")

        st.bar_chart(pd.DataFrame({
            "Appointments": [attended, no_shows]
        }, index=["Attended", "No-show"]))

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Appointment volume by month")

        monthly = dates.dropna().dt.month.value_counts().sort_index()

        if not monthly.empty:
            monthly.index = [
                pd.Timestamp(2000, int(m), 1).strftime("%b")
                for m in monthly.index
            ]
            st.bar_chart(monthly.rename("Appointments"))
        else:
            st.info("No valid appointment dates are available.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Attendance by age group")

    age_data = pd.DataFrame({
        "Age": pd.to_numeric(raw["Age"], errors="coerce"),
        "Outcome": outcomes
    })

    age_data["Age group"] = pd.cut(
        age_data["Age"],
        bins=[-1, 17, 29, 44, 59, 74, 120],
        labels=["0–17", "18–29", "30–44", "45–59", "60–74", "75+"]
    )

    grouped = (
        age_data[age_data["Outcome"].isin(["yes", "no"])]
        .groupby(["Age group", "Outcome"], observed=False)
        .size()
        .unstack(fill_value=0)
    )

    st.bar_chart(grouped)

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

elif page == "Predict Risk":
    st.subheader("Patient risk assessment")
    st.write(
        "Enter appointment information to estimate the probability "
        "of a recorded no-show."
    )

    neighbourhoods = sorted(
        raw["Neighbourhood"].dropna()
        .astype(str).str.strip().str.upper().unique().tolist()
    )

    if not neighbourhoods:
        neighbourhoods = ["UNKNOWN"]

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            age = st.slider("Patient age", 0, 100, 30)

            gender = st.selectbox(
                "Gender",
                ["F", "M"],
                format_func=lambda x: "Female" if x == "F" else "Male"
            )

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

            alcoholism = st.selectbox(
                "Alcoholism", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

        with right:
            handicap = st.selectbox(
                "Handicap level", [0, 1, 2, 3, 4]
            )

            sms = st.selectbox(
                "SMS reminder received", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

            waiting_days = st.slider(
                "Waiting days", 0, 60, 5
            )

            weekday = st.selectbox(
                "Appointment weekday",
                [
                    "Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday"
                ]
            )

            month = st.selectbox(
                "Appointment month", list(range(1, 13)), index=4
            )

            hour = st.slider("Scheduling hour", 0, 23, 10)

        neighbourhood = st.selectbox(
            "Neighbourhood", neighbourhoods
        )

        submitted = st.form_submit_button(
            "Calculate no-show risk",
            use_container_width=True
        )

    if submitted:
        weekday_number = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ].index(weekday)

        patient = pd.DataFrame([{
            "Age": age,
            "Scholarship": scholarship,
            "Hypertension": hypertension,
            "Diabetes": diabetes,
            "Alcoholism": alcoholism,
            "Handicap": handicap,
            "SMS_received": sms,
            "WaitingDays": waiting_days,
            "Appointment_DayOfWeek": weekday_number,
            "Appointment_Month": month,
            "Appointment_Hour": hour,
            "Gender": gender,
            "Neighbourhood": neighbourhood
        }])[FEATURES]

        try:
            probability = float(
                model.predict_proba(patient)[0][1]
            ) * 100

            if probability < 30:
                label = "Low risk"
                color = "#13866f"
                advice = "A standard appointment reminder may be appropriate."
            elif probability < 60:
                label = "Moderate risk"
                color = "#bd7b12"
                advice = "Consider sending a reminder and requesting confirmation."
            else:
                label = "Higher risk"
                color = "#c44242"
                advice = "Consider a reminder and follow-up confirmation."

            st.markdown(f"""
            <div class="panel">
                <div style="color:#718198;font-weight:700;
                text-transform:uppercase;font-size:.8rem">
                    Model estimate
                </div>
                <h2 style="color:{color};margin:.5rem 0">
                    {label} · {probability:.1f}%
                </h2>
                <p>{advice}</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(
                min(max(int(round(probability)), 0), 100)
            )

            st.caption(
                "Risk bands are application-defined. This is an "
                "educational model estimate, not a clinical diagnosis."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# --------------------------------------------------
# MODEL EVALUATION PAGE
# --------------------------------------------------

elif page == "Model Evaluation":
    st.subheader("Model evaluation")

    st.metric(
        "Test accuracy",
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    st.write(f"Valid records: {metrics['rows']:,}")
    st.write(f"Training records: {metrics['train_rows']:,}")
    st.write(f"Test records: {metrics['test_rows']:,}")

    if metrics.get("roc_auc") is not None:
        st.metric("Test ROC-AUC", f"{metrics['roc_auc']:.4f}")
    else:
        st.info("ROC-AUC could not be calculated.")

    st.warning(
        "These results are measured on a held-out split of artificially "
        "generated data. They do not demonstrate clinical performance."
    )

# --------------------------------------------------
# ABOUT DATA PAGE
# --------------------------------------------------

else:
    st.subheader("About this dataset")

    st.warning(
        "This dataset is artificially generated for development and "
        "testing. It is not real clinical data."
    )

    st.write(f"**Dataset file:** `{DATA_PATH}`")
    st.write(f"**Rows:** {len(raw):,}")
    st.write(f"**Columns:** {len(raw.columns)}")

    st.markdown("### Dataset preview")
    st.dataframe(
        raw.head(25),
        use_container_width=True,
        hide_index=True
    )

st.divider()

st.caption(
    "CarePulse · Educational demonstration · "
    "Synthetic data must not be used for clinical decisions."
)
