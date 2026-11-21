import os
import numpy as np
import pandas as pd

def generate_blood_pressure_dataset(num_samples=2500, random_seed=42):
    """
    Generates a realistic clinical dataset for blood pressure prediction
    based on empirical medical risk factors (AHA / Framingham studies).
    """
    np.random.seed(random_seed)

    # 1. Demographics & Physiology
    age = np.random.randint(18, 80, size=num_samples)
    gender = np.random.choice(["Male", "Female"], size=num_samples, p=[0.5, 0.5])
    
    # BMI: typically normal distribution centered at ~26.5
    bmi = np.random.normal(loc=26.5, scale=5.0, size=num_samples)
    bmi = np.clip(bmi, 16.0, 45.0)

    # Resting Heart Rate (bpm): normal ~72
    heart_rate = np.random.normal(loc=72, scale=9.5, size=num_samples)
    heart_rate = np.clip(heart_rate, 50, 115)

    # 2. Lifestyle & Environmental Factors
    daily_steps = np.random.normal(loc=6500, scale=2500, size=num_samples)
    daily_steps = np.clip(daily_steps, 1000, 18000)

    sleep_hours = np.random.normal(loc=7.0, scale=1.2, size=num_samples)
    sleep_hours = np.clip(sleep_hours, 4.0, 10.0)

    stress_level = np.random.randint(1, 11, size=num_samples)  # 1 (low) to 10 (high)
    smoking_status = np.random.choice(["Non-Smoker", "Smoker"], size=num_samples, p=[0.75, 0.25])
    alcohol_intake = np.random.choice(["None", "Moderate", "Heavy"], size=num_samples, p=[0.50, 0.35, 0.15])
    salt_intake = np.random.choice(["Low", "Medium", "High"], size=num_samples, p=[0.25, 0.50, 0.25])

    # 3. Cardiovascular Relationship Models (Physiological synthesis)
    # Systolic BP baseline ~ 100 mmHg
    sbp = (
        98.0
        + 0.52 * (age - 20)
        + 1.15 * (bmi - 22)
        + 0.14 * (heart_rate - 70)
        - 0.0012 * (daily_steps - 5000)
        - 1.2 * (sleep_hours - 7)
        + 1.1 * (stress_level - 5)
        + (smoking_status == "Smoker") * 6.5
        + (alcohol_intake == "Heavy") * 5.0 + (alcohol_intake == "Moderate") * 1.5
        + (salt_intake == "High") * 5.5 + (salt_intake == "Low") * -2.5
        + (gender == "Male") * 2.0
        + np.random.normal(0, 5.5, size=num_samples)  # Physiological individual variance
    )

    # Diastolic BP baseline ~ 66 mmHg
    dbp = (
        65.0
        + 0.24 * (age - 20)
        + 0.68 * (bmi - 22)
        + 0.11 * (heart_rate - 70)
        - 0.0007 * (daily_steps - 5000)
        - 0.8 * (sleep_hours - 7)
        + 0.75 * (stress_level - 5)
        + (smoking_status == "Smoker") * 4.0
        + (alcohol_intake == "Heavy") * 3.5 + (alcohol_intake == "Moderate") * 1.0
        + (salt_intake == "High") * 3.8 + (salt_intake == "Low") * -1.8
        + (gender == "Male") * 1.5
        + np.random.normal(0, 4.0, size=num_samples)  # Individual variance
    )

    # Realistic rounding & bounding
    sbp = np.round(np.clip(sbp, 85, 205), 1)
    dbp = np.round(np.clip(dbp, 50, 130), 1)

    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "BMI": np.round(bmi, 1),
        "Heart_Rate": np.round(heart_rate).astype(int),
        "Daily_Steps": np.round(daily_steps).astype(int),
        "Sleep_Hours": np.round(sleep_hours, 1),
        "Stress_Level": stress_level,
        "Smoking_Status": smoking_status,
        "Alcohol_Intake": alcohol_intake,
        "Salt_Intake": salt_intake,
        "Systolic_BP": sbp,
        "Diastolic_BP": dbp
    })

    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "bp_data.csv")
    df = generate_blood_pressure_dataset()
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} samples and saved to {output_path}")
    print(df.head())
