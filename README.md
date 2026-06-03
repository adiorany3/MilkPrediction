# Prediksi Produksi Susu Sapi Perah

Aplikasi Streamlit ini menggunakan model `milk_yield_model.keras` untuk memprediksi produksi susu sapi perah dari gambar.

## Struktur file

```text
.
├── app.py
├── milk_yield_model.keras
├── preprocess_config.json
├── requirements.txt
└── README.md
```

## Menjalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Buat repository GitHub baru.
2. Upload semua file dalam folder ini.
3. Buka Streamlit Community Cloud.
4. Pilih repository.
5. Main file path: `app.py`.
6. Deploy.

## Catatan output

Konfigurasi model menyatakan target bernama `milk_yield_305d`. Karena itu, output utama aplikasi mengikuti target training tersebut. Jika target training adalah 305-day milk yield, maka hasil utama juga merupakan estimasi 305-day milk yield. Aplikasi juga menyediakan estimasi rata-rata per hari dengan rumus sederhana:

```text
estimasi_per_hari = prediksi_305_hari / 305
```
