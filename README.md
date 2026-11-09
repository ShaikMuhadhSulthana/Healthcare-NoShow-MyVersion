# MediPredict AI – Healthcare Appointment No-Show Prediction

## 1. Project Overview

MediPredict AI is a machine learning-based healthcare application developed to predict the likelihood of patients missing their scheduled medical appointments.

Missed medical appointments are a common challenge in healthcare organizations. They can result in wasted appointment slots, inefficient use of medical staff, longer waiting periods for other patients, and difficulties in hospital resource management.

This project aims to address this problem by analyzing historical medical appointment records and identifying patterns associated with patient attendance.

Using machine learning techniques, the application estimates the probability of a patient not attending an appointment and categorizes the result into different risk levels. The interactive Streamlit dashboard allows users to explore appointment statistics, understand attendance patterns, and generate individual no-show risk estimates.

The main objective is to support better appointment planning and improve the utilization of healthcare resources.

## 2. Project Objectives

- Analyze historical healthcare appointment records.
- Identify factors associated with missed medical appointments.
- Develop a machine learning model to estimate no-show probability.
- Categorize appointments into low, moderate, and higher risk levels.
- Provide an interactive dashboard for exploring healthcare appointment data.
- Present meaningful visualizations to understand attendance trends.
- Support healthcare professionals in making more informed appointment-management decisions.

## 3. Key Features

### Interactive Dashboard
- Displays the total number of appointments.
- Shows attended and missed appointment counts.
- Calculates the overall no-show percentage.
- Presents appointment attendance statistics through visualizations.

### Appointment Risk Prediction
- Accepts patient and appointment-related information.
- Analyzes factors such as age, gender, medical conditions, SMS reminders, and waiting days.
- Estimates the probability of a patient missing an appointment.
- Classifies the prediction into low, moderate, and higher risk categories.
- Provides basic recommendations based on the estimated risk.

### Data Visualization
- Appointment attendance distribution.
- Monthly appointment trends.
- Attendance patterns across different age groups.
- Graphical representation of healthcare appointment statistics.

### Machine Learning
- Uses a Random Forest classification algorithm.
- Performs data preprocessing and feature engineering.
- Handles missing values through appropriate imputation.
- Uses numerical and categorical features for model training.
- Evaluates model performance using a separate test dataset.

### Model Evaluation
- Displays model accuracy.
- Reports ROC-AUC when available.
- Shows training and testing record counts.
- Provides an overview of model performance.

## 4. Dataset Description

The project uses a medical appointment no-show dataset containing historical patient appointment information.

The dataset includes information about patients, appointment schedules, medical conditions, reminders, and attendance outcomes.

The important attributes include:

| Attribute | Description |
|---|---|
| PatientId | Unique patient identifier |
| AppointmentID | Unique appointment identifier |
| Gender | Patient gender |
| ScheduledDay | Date and time when the appointment was scheduled |
| AppointmentDay | Date of the actual appointment |
| Age | Patient age |
| Neighbourhood | Location associated with the appointment |
| Scholarship | Indicates participation in a welfare scholarship program |
| Hipertension | Indicates hypertension status |
| Diabetes | Indicates diabetes status |
| Alcoholism | Indicates alcoholism status |
| Handcap | Indicates handicap information |
| SMS_received | Indicates whether an SMS reminder was received |
| No-show | Indicates whether the patient missed the appointment |

The target variable is **No-show**, which is used to train the classification model.

- Yes – The patient did not attend the appointment.
- No – The patient attended the appointment.

The dataset is stored separately from the application code in the `data` directory.

## 5. Technology Stack

**Programming Language**
- Python

**Frontend and Dashboard**
- Streamlit
- HTML
- CSS

**Data Processing**
- Pandas
- NumPy

**Machine Learning**
- Scikit-learn
- Random Forest Classifier

**Model Storage**
- Joblib

**Data Visualization**
- Matplotlib
- Streamlit Charts

**Development Tools**
- Visual Studio Code
- Git
- GitHub

## 6. Machine Learning Workflow

The application follows a structured machine learning workflow.

**Step 1 – Data Collection**

Historical medical appointment records are loaded from the CSV dataset.

**Step 2 – Data Preprocessing**

The dataset is checked and prepared for model training. Date columns are converted into appropriate datetime formats, and missing or invalid values are handled.

**Step 3 – Feature Engineering**

Additional appointment-related features are derived, including:

- Waiting days between scheduling and appointment.
- Appointment weekday.
- Appointment month.
- Scheduling hour.

These features help the model identify patterns associated with appointment attendance.

**Step 4 – Feature Selection**

Relevant numerical and categorical attributes are selected for training the model.

**Step 5 – Model Training**

A Random Forest Classifier is trained using the prepared dataset. The data is divided into training and testing subsets using a stratified split.

**Step 6 – Model Evaluation**

The trained model is evaluated using test data. Accuracy and ROC-AUC are calculated where applicable.

**Step 7 – Risk Prediction**

The trained model receives patient and appointment information and estimates the probability of a no-show.

**Step 8 – Dashboard Visualization**

The results are displayed through the MediPredict AI dashboard using charts, statistics, and risk indicators.

## 7. Project Structure

```text
Healthcare-NoShow-MyVersion/
│
├── app.py
│
├── data/
│   └── Medical_Appointment_No_Shows_2017.csv
│
├── models/
│   ├── feature_columns.pkl
│   └── no_show_model.pkl
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

*The actual model filenames may differ depending on the saved model version in your project.*

## 8. Installation and Setup

Follow these steps to run the application on your local machine.

### Prerequisites

- Python 3.10 or later.
- Visual Studio Code.
- Git.
- Required project dataset.

### Step 1 – Clone the Repository

```bash
git clone https://github.com/ShaikMuhadhSulthana/Healthcare-NoShow-MyVersion.git
```

### Step 2 – Navigate to the Project

```bash
cd Healthcare-NoShow-MyVersion
```

### Step 3 – Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Step 4 – Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5 – Verify the Dataset

Make sure the CSV file is available inside the `data` folder.

### Step 6 – Run the Application

```bash
streamlit run app.py
```

The application will start locally and can be accessed through:

```text
http://localhost:8501
```

## 9. Application Interface

MediPredict AI uses a Burgundy and Rose Gold dark-mode theme to provide a clean and consistent dashboard experience.

The application includes the following sections:

- **Overview:** Provides a summary of appointment records and attendance patterns.
- **Predict Risk:** Allows users to enter patient details and estimate no-show risk.
- **Model Evaluation:** Displays the machine learning model's evaluation results.
- **About Data:** Provides information about the dataset and displays a sample of its records.

## 10. Expected Benefits

- Helps identify appointments with a higher estimated probability of being missed.
- Supports proactive appointment reminders.
- Helps healthcare organizations understand patient attendance patterns.
- Provides useful insights into appointment scheduling.
- Encourages more efficient utilization of available appointment slots.
- Offers a data-driven approach to healthcare appointment management.

## 11. Future Enhancements

The project can be extended with additional capabilities, including:

- Secure login and role-based access for administrators, doctors, and staff.
- Patient and appointment management modules.
- Advanced machine learning models and hyperparameter optimization.
- Explainable AI techniques for understanding prediction factors.
- More detailed healthcare analytics and reports.
- Automated appointment reminders.
- Integration with hospital management systems.
- Cloud deployment for wider accessibility.

## 12. Disclaimer

MediPredict AI is an academic and educational machine learning project. Its predictions are estimates generated from historical appointment data and should not be treated as guaranteed outcomes or clinical recommendations.

The system is intended to support appointment planning and resource management. Healthcare professionals should use their judgment and other relevant information when making operational decisions.

## 13. Author

**Shaik Muhadh Sulthana**

B.Tech – Artificial Intelligence and Machine Learning

GitHub: https://github.com/ShaikMuhadhSulthana

---

**MediPredict AI – Smarter Insights. Better Appointment Planning.**