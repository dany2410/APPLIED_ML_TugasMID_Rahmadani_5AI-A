# Deploy_Gradio.py
# Aplikasi Gradio untuk Prediksi Jenis Wilayah Sekolah (Urban / Rural)

import os
import pickle
import pandas as pd
import numpy as np
import gradio as gr

# ==============================
# Load Model
# ==============================
MODEL_PATH = "../Models/rf_pipeline_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model belum ditemukan. Jalankan notebook terlebih dahulu!")

with open(MODEL_PATH, "rb") as f:
    obj = pickle.load(f)

pipeline = obj["pipeline"]
le = obj["label_encoder"]
numeric_cols = obj["numeric_cols"]
categorical_cols = obj["categorical_cols"]

# ==============================
# UI Inputs (dipilih lebih ringkas)
# ==============================
ui_numeric = [
    c for c in [
        "jumlah_peserta_didik",
        "jumlah_pendidik",
        "proporsi_pendidik_min_s1",
        "proporsi_pendidik_sertifikasi"
    ] if c in numeric_cols
]

ui_categorical = [
    c for c in [
        "kurikulum",
        "jenis_sek",
        "sts_sek",
        "daerah_khusus",
        "wilayah_bagian"
    ] if c in categorical_cols
]

inputs = []

for c in ui_numeric:
    inputs.append(gr.Number(label=c))

for c in ui_categorical:
    inputs.append(gr.Textbox(label=c))

# ==============================
# Mapping Admin → Urban/Rural
# ==============================
mapping = {
    "KOTA": "URBAN",
    "KABUPATEN": "RURAL"
}

# ==============================
# Prediction Function
# ==============================
def predict(*vals):
    data = {}

    # isi kolom numerik
    i = 0
    for col in ui_numeric:
        data[col] = [vals[i]]
        i += 1

    # isi kolom kategorikal
    for col in ui_categorical:
        data[col] = [vals[i]]
        i += 1

    # pastikan semua kolom terisi
    for col in numeric_cols:
        if col not in data:
            data[col] = [0]

    for col in categorical_cols:
        if col not in data:
            data[col] = [""]

    # buat DataFrame untuk prediksi
    X = pd.DataFrame(data)[numeric_cols + categorical_cols]

    # prediksi
    pred = pipeline.predict(X)[0]
    proba = pipeline.predict_proba(X)[0]

    # label asli dataset → “KOTA” / “KABUPATEN”
    label_admin = le.inverse_transform([pred])[0]

    # hasil mapping → “URBAN / RURAL”
    final_label = mapping.get(label_admin, "UNKNOWN")

    # probabilitas dimapping juga
    proba_named = {}
    for idx, p in enumerate(proba):
        original_label = le.inverse_transform([idx])[0]
        mapped_label = mapping.get(original_label, original_label)
        proba_named[mapped_label] = float(p)

    # output final
    return f"{final_label} ({label_admin})", proba_named

# ==============================
# Gradio App
# ==============================
outputs = [
    gr.Label(label="Prediksi Jenis Wilayah"),
    gr.Label(label="Probabilitas (Urban/Rural)")
]

app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title="Prediksi Jenis Wilayah Sekolah",
    description="Aplikasi Machine Learning dengan Random Forest untuk menentukan jenis wilayah sekolah (URBAN/RURAL)."
)

# ==============================
# Jalankan Aplikasi
# ==============================
if __name__ == "__main__":
    app.launch()
