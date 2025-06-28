import streamlit as st
import numpy as np
import pickle

# Load model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Prediksi Tingkat Stress Mahasiswa 🎓")

study = st.slider("Jam Belajar per Hari", 0, 10, 5, step=1)
sleep = st.slider("Jam Tidur per Hari", 0, 12, 5, step=1)
phys = st.slider("Jam Aktivitas Fisik per Hari", 0, 5, 1, step=1)
social = st.slider("Jam Bersosialisasi per Hari", 0, 5, 1, step=1)
gpa = st.number_input("IPK", min_value=0.0, max_value=4.0, step=0.01, value=2.5)

if st.button("Prediksi"):
    input_data = np.array([[study, sleep, phys, social, gpa]])
    pred = model.predict(input_data)[0]
    
    label_map = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
    st.success(f"Tingkat Stress: **{label_map.get(pred, 'Tidak diketahui')}**")
