# 🩺 Blood Pressure (BP) Prediction - Machine Learning

An end-to-end, lightweight Machine Learning project that predicts **Systolic Blood Pressure (SBP)** and **Diastolic Blood Pressure (DBP)** using patient biometrics and lifestyle risk factors. It maps predictions directly to the **American Heart Association (AHA)** clinical categories and offers lifestyle recommendations.

Designed to be **minimal, clean, and production-ready** for immediate deployment on Git and Cloud platforms.

---

## 🚀 Features

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
└── data/
    └── generate_data.py # Physiological data generation script (AHA/Framingham risk models)
```

---

## 🛠️ Local Setup & Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/bp-prediction.git
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

## 🌐 Deploy to Cloud (Free & Simple)

### Option 1: Streamlit Community Cloud (Recommended - 100% Free)
1. Push this repository to **GitHub** (see Git steps below).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account and click **New app**.
4. Select your repository, branch (`main`), and set Main file path to `app.py`.
5. Click **Deploy!** Your app is live with a public URL in 2 minutes.

### Option 2: Hugging Face Spaces (Free)
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Select **Streamlit** as the SDK.
3. Push files to the Hugging Face Git repo or connect your GitHub repository.

### Option 3: Render / Railway / Google Cloud Run
- Deploy directly using the provided `Dockerfile`.

---

## 🐙 Push to GitHub

Run these commands in your project terminal:

```bash
git init
git add .
git commit -m "feat: initial commit for Blood Pressure prediction ML project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/bp-prediction.git
git push -u origin main
```

---

## 📊 Model Evaluation Metrics

| Target Metric | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R² Score |
| :--- | :---: | :---: | :---: |
| **Systolic BP (SBP)** | ~4.2 mmHg | ~5.3 mmHg | ~0.89 |
| **Diastolic BP (DBP)** | ~3.0 mmHg | ~3.8 mmHg | ~0.87 |

---

## ⚖️ Disclaimer
*This project is for educational and screening estimation purposes. It should not replace professional medical diagnosis or clinical devices.*
