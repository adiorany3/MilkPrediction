# Prediksi Produksi Susu Sapi Perah

Aplikasi Streamlit ini menggunakan model `milk_yield_model.keras` untuk memprediksi produksi susu sapi perah dari gambar.

## Training data sheet

Dataset/training notebook rujukan:

- Kaggle Notebook: [Cow Milk Prediction by Galuh Adi Insani](https://www.kaggle.com/code/adioranye/cow-milk-prediction-by-galuh-adi-insani/notebook)

Model dalam paket ini menggunakan konfigurasi:

```json
{
  "img_size": 224,
  "model_type": "EfficientNetB0 image regression",
  "target_name": "milk_yield_305d"
}
```

## Struktur file

```text
.
├── app.py
├── milk_yield_model.keras
├── preprocess_config.json
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
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

## Tampilan light/dark

Tema aplikasi tidak dikunci ke `light` atau `dark` di `config.toml`. Elemen kustom seperti footer dan catatan menggunakan CSS adaptif agar tetap terbaca pada lingkungan light maupun dark.

Branding/menu bawaan Streamlit disembunyikan melalui CSS di `app.py`.

## Footer aplikasi

Footer aplikasi menampilkan:

```text
Developed by Galuh Adi Insani
```

## Catatan output

Konfigurasi model menyatakan target bernama `milk_yield_305d`. Karena itu, output utama aplikasi mengikuti target training tersebut. Jika target training adalah 305-day milk yield, maka hasil utama juga merupakan estimasi 305-day milk yield. Aplikasi juga menyediakan estimasi rata-rata per hari dengan rumus sederhana:

```text
estimasi_per_hari = prediksi_305_hari / 305
```

## Catatan penggunaan

Hasil prediksi adalah estimasi berbasis model, bukan pengukuran produksi aktual. Akurasi sangat bergantung pada kualitas dataset training, sudut foto, pencahayaan, dan kemiripan gambar input dengan data training.
