# 🩺 Blood Pressure (BP) Prediction - Machine Learning

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=deepaks25bai108-create/bp-prediction&branch=main&mainModule=app.py)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/deepaks25bai108-create/bp-prediction)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg?logo=python)](https://python.org)

An end-to-end, lightweight Machine Learning project that predicts **Systolic Blood Pressure (SBP)** and **Diastolic Blood Pressure (DBP)** using patient biometrics and lifestyle risk factors. It maps predictions directly to the **American Heart Association (AHA)** clinical categories and offers personalized lifestyle recommendations.

Designed to be **minimal, clean, and production-ready** for immediate deployment on Git and Cloud platforms.

---

## 🌐 Instant Cloud Deployment (1-Click)

Click the badge below to deploy this app for **free** on Streamlit Community Cloud:

[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=deepaks25bai108-create/bp-prediction&branch=main&mainModule=app.py)

---

## 🚀 Key Features

- **Multi-Output Machine Learning Model**: Predicts both continuous targets (`Systolic_BP` and `Diastolic_BP`) simultaneously with Random Forest regression and `scikit-learn` Pipeline.
- **Leak-Free Preprocessing**: Automated `ColumnTransformer` with `StandardScaler` for numeric features and `OneHotEncoder` for categorical factors.
- **Clinical Risk Classification**: Categorizes BP into **Normal**, **Elevated**, **Stage 1 Hypertension**, **Stage 2 Hypertension**, or **Hypertensive Crisis**.
- **Interactive Cloud-Ready Web App**: Built with **Streamlit** for real-time predictions and responsive visual feedback.
- **Ready for Cloud Deployment**: Zero extra setup required for **Streamlit Community Cloud**, **Hugging Face Spaces**, **Render**, or **Docker/Cloud Run**.

---

## 📁 Project Structure

```text
BP prediction/
├── app.py               # Streamlit web application & clinical dashboard
├── train.py             # ML pipeline training, evaluation & model persistence
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container setup for any cloud container service
├── .gitignore           # Standard git exclusion file
├── README.md            # Documentation & deployment guide
├── model.joblib         # Pre-trained multi-output Random Forest pipeline
└── data/
    ├── generate_data.py # Physiological data generation script (AHA risk models)
    └── bp_data.csv      # Clinical dataset (2,500 patient records)
```

---

## 🛠️ Local Setup & Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/deepaks25bai108-create/bp-prediction.git
cd bp-prediction
```

### 2. Set up virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the ML Model
```bash
python train.py
```
*Outputs evaluation metrics (MAE, RMSE, R²) and generates `model.joblib`.*

### 5. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📦 Run with Docker

```bash
# Build the Docker image
docker build -t bp-predictor .

# Run the container
docker run -p 8501:8501 bp-predictor
```
Access at `http://localhost:8501`.

---

## 📊 Model Evaluation Metrics

| Target Metric | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R² Score |
| :--- | :---: | :---: | :---: |
| **Systolic BP (SBP)** | ~4.2 mmHg | ~5.3 mmHg | ~0.89 |
| **Diastolic BP (DBP)** | ~3.0 mmHg | ~3.8 mmHg | ~0.87 |

---

## ⚖️ Disclaimer
*This project is for educational and screening estimation purposes. It should not replace professional medical diagnosis or clinical devices.*
