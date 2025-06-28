# 🧠 Prediksi Tingkat Stres Mahasiswa dari Hasil Survei

Aplikasi web sederhana berbasis machine learning yang digunakan untuk mengklasifikasikan tingkat stres mahasiswa menjadi tiga level: **Rendah, Sedang,** atau **Tinggi**. Klasifikasi ini didasarkan pada berbagai faktor seperti gaya hidup, IPK, durasi tidur, dan jam belajar.

Aplikasi ini dibuat menggunakan **Python** dan **Streamlit**, model yang telah dilatih disimpan kedalam `model.pkl`.

---

## 🔧 Tools yang Digunakan

- **Python 3.10+**
- **Streamlit** — untuk membuat tampilan pada website.
- **scikit-learn** — untuk load dan menggunakan model yang telah dilatih.
- **pandas, numpy** — untuk data processing.

---

## ▶️ Cara Kerja
- Install semua tools yang dibutuhkan.
- Ketik `python train_model.py` di dalam Terminal untuk train model terlebih dahulu.
- Ketik `streamlit run app.py` untuk menjalankan aplikasi di portal localhost.