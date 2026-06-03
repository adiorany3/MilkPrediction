# Prediksi Produksi Susu Sapi Perah

Aplikasi Streamlit ini menggunakan pipeline model terbaru:

```text
Gambar sapi perah
    → EfficientNetB0 feature extractor
    → SVR RBF regressor
    → prediksi milk_yield_305d
    → konversi kg ke liter
```

## Training data sheet

Dataset/training notebook rujukan:

- Kaggle Notebook: [Cow Milk Prediction by Galuh Adi Insani](https://www.kaggle.com/code/adioranye/cow-milk-prediction-by-galuh-adi-insani/notebook)

## Model yang digunakan

File model yang dipakai aplikasi:

```text
feature_extractor.keras
milk_yield_regressor.pkl
preprocess_config.json
```

Konfigurasi model:

```json
{
  "img_size": 224,
  "model_type": "EfficientNetB0 feature extractor + classical regression",
  "best_regressor": "SVR_rbf_C10",
  "target_name": "milk_yield_305d",
  "output_unit": "kg",
  "conversion_to_liter": "liter = kg / 1.03"
}
```

Output mentah model adalah estimasi **kg/305 hari**, lalu aplikasi mengonversinya menjadi **liter/305 hari** dengan rumus:

```text
liter = kg / 1.03
```

Aplikasi juga menampilkan estimasi rata-rata harian:

```text
liter_per_day = liter_305_day / 305
```

## Struktur file

```text
.
├── app.py
├── feature_extractor.keras
├── milk_yield_regressor.pkl
├── preprocess_config.json
├── requirements.txt
├── runtime.txt
├── .python-version
├── README.md
└── .streamlit/
    └── config.toml
```

## Menjalankan lokal

Disarankan memakai Python 3.11.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Upload semua file ke repository GitHub.
2. Buka Streamlit Community Cloud.
3. Deploy repository tersebut.
4. Main file path: `app.py`.
5. Pada **Advanced settings**, pilih **Python 3.11**.
6. Deploy.

## Catatan dependency penting

File `milk_yield_regressor.pkl` dibuat menggunakan `scikit-learn==1.2.2`, sehingga `requirements.txt` mem-pin versi tersebut:

```text
scikit-learn==1.2.2
joblib==1.2.0
```

Jika versi `scikit-learn` berbeda terlalu jauh, file `.pkl` dapat gagal dimuat atau menghasilkan warning kompatibilitas.

## Tampilan light/dark

Tema aplikasi tidak dikunci ke `light` atau `dark`. Elemen kustom seperti footer dan catatan menggunakan CSS adaptif agar tetap terbaca pada lingkungan light maupun dark.

Branding/menu bawaan Streamlit disembunyikan melalui CSS di `app.py`.

## Footer aplikasi

Footer aplikasi menampilkan informasi pengembang dan tautan proses training di Kaggle:

```text
Developed by Galuh Adi Insani with training process at Kaggle
```

Training process: [Cow Milk Prediction by Galuh Adi Insani](https://www.kaggle.com/code/adioranye/cow-milk-prediction-by-galuh-adi-insani/notebook)

## Catatan penggunaan

Hasil prediksi adalah estimasi berbasis model, bukan pengukuran produksi aktual. Model ini bersifat eksperimental karena dataset training kecil dan hasil cross-validation masih terbatas.

Gunakan input gambar yang mendekati data training: gambar sapi perah, pencahayaan jelas, dan sudut foto yang serupa dengan data training.
