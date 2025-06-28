import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

# Load dataset
df = pd.read_csv("dataset/student_stress.csv")

# Drop kolom tidak dipakai
# df = df.drop(columns=["Extracurricular_Hours_Per_Day"], errors="ignore")
df = df.drop(columns=["Student_ID"], errors="ignore")

# ubah stress_level jadi angka
label_mapping = {"Low": 0, "Moderate": 1, "High": 2}
df["Stress_Level_Encoded"] = df["Stress_Level"].map(label_mapping)

# Fitur dan label
X = df.drop(columns=["Stress_Level", "Stress_Level_Encoded"])
y = df["Stress_Level_Encoded"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Pipeline model
pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Simpan model ke folder model/
with open("model/model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model berhasil disimpan ke model/model.pkl")
