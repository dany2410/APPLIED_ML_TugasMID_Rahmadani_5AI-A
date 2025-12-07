# APPLIED_ML_TugasMID_Rahmadani_5AI-A
# Prediksi Jenis Wilayah Sekolah (Perkotaan vs Perdesaan)

**Applied Machine Learning – Tugas MID**  
**Nama:** RAHMADANI | **Kelas:** 5AI-A | **Tanggal:** 7 Desember 2025

Proyek ini mengikuti metodologi **CRISP-DM** secara lengkap untuk memprediksi apakah sekolah berada di wilayah **Perkotaan** atau **Perdesaan** hanya dari indikator pendidikan (literasi, numerasi, fasilitas, rasio siswa-guru, dll.) menggunakan **Random Forest**.

### Dataset
Rapor Publik Asesmen Nasional 2024 (Kemdikbud)  
Link: https://data.kemendikdasmen.go.id/dataset/p/asesmen-nasional/rapor-publik-asesmen-nasional-2024-kepala-satuan-pendidikan-2024-indonesia

### Isi Repository
- `LAPORAN/` → LK Perancangan Proyek + Laporan lengkap  
- `Notebook/` → Notebook .ipynb  
- `Models/` → rf_pipeline_model.pkl  
- `PROGRAM/` → deploy_gradio.py  
- `Reports/` → Feature importance & evaluasi

### Cara Menjalankan Demo
```bash
pip install gradio pandas scikit-learn joblib
python PROGRAM/deploy_gradio.py
