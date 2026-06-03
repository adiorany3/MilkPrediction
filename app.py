import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageOps


APP_TITLE = "Prediksi Produksi Susu Sapi Perah"
FEATURE_EXTRACTOR_PATH = Path("feature_extractor.keras")
REGRESSOR_PATH = Path("milk_yield_regressor.pkl")
CONFIG_PATH = Path("preprocess_config.json")
KG_PER_LITER = 1.03
DAYS_IN_TARGET = 305


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🐄",
    layout="centered",
)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            #MainMenu {
                visibility: hidden;
            }

            footer {
                visibility: hidden;
            }

            header {
                visibility: hidden;
            }

            .stDeployButton {
                display: none !important;
            }

            [data-testid="stToolbar"] {
                display: none !important;
            }

            [data-testid="stDecoration"] {
                display: none !important;
            }

            [data-testid="stStatusWidget"] {
                display: none !important;
            }

            [data-testid="stAppDeployButton"] {
                display: none !important;
            }

            :root {
                color-scheme: light dark;
                --milk-footer-border: rgba(49, 51, 63, 0.18);
                --milk-footer-text: rgba(49, 51, 63, 0.72);
                --milk-soft-card: rgba(255, 255, 255, 0.72);
            }

            @media (prefers-color-scheme: dark) {
                :root {
                    --milk-footer-border: rgba(250, 250, 250, 0.18);
                    --milk-footer-text: rgba(250, 250, 250, 0.72);
                    --milk-soft-card: rgba(18, 18, 18, 0.72);
                }
            }

            .milk-footer {
                margin-top: 2.5rem;
                padding-top: 1rem;
                border-top: 1px solid var(--milk-footer-border);
                color: var(--milk-footer-text);
                text-align: center;
                font-size: 0.88rem;
                line-height: 1.5;
            }

            .milk-footer a {
                color: inherit;
                font-weight: 600;
                text-decoration: underline;
                text-underline-offset: 0.18rem;
            }

            .milk-note {
                padding: 0.85rem 1rem;
                border-radius: 0.75rem;
                background: var(--milk-soft-card);
                border: 1px solid var(--milk-footer-border);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()


@st.cache_data
def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    return {
        "img_size": 224,
        "model_type": "EfficientNetB0 feature extractor + classical regression",
        "best_regressor": "SVR_rbf_C10",
        "target_name": "milk_yield_305d",
        "output_unit": "kg",
        "conversion_to_liter": "liter = kg / 1.03",
        "prediction_note": "Prediction output is 305-day milk yield in kg before conversion to liter.",
    }


@st.cache_resource
def load_models() -> Tuple[Any, Any]:
    try:
        import joblib
        import tensorflow as tf
    except ModuleNotFoundError as error:
        st.error(
            "Dependency model belum lengkap. Pastikan `requirements.txt` berisi TensorFlow, "
            "scikit-learn, dan joblib. Untuk Streamlit Community Cloud, gunakan Python 3.11."
        )
        raise error

    if not FEATURE_EXTRACTOR_PATH.exists():
        st.error("File `feature_extractor.keras` tidak ditemukan.")
        st.stop()

    if not REGRESSOR_PATH.exists():
        st.error("File `milk_yield_regressor.pkl` tidak ditemukan.")
        st.stop()

    try:
        feature_extractor = tf.keras.models.load_model(
            FEATURE_EXTRACTOR_PATH,
            compile=False,
            safe_mode=False,
        )
    except TypeError:
        feature_extractor = tf.keras.models.load_model(
            FEATURE_EXTRACTOR_PATH,
            compile=False,
        )

    regressor = joblib.load(
        REGRESSOR_PATH
    )

    return feature_extractor, regressor


def prepare_image(uploaded_file, img_size: int) -> np.ndarray:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize((img_size, img_size))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_milk_yield(image_array: np.ndarray) -> Tuple[float, float, float]:
    feature_extractor, regressor = load_models()

    features = feature_extractor.predict(
        image_array,
        verbose=0,
    )

    prediction_kg_305d = float(
        np.ravel(
            regressor.predict(features)
        )[0]
    )

    prediction_liter_305d = prediction_kg_305d / KG_PER_LITER
    prediction_liter_per_day = prediction_liter_305d / DAYS_IN_TARGET

    return prediction_kg_305d, prediction_liter_305d, prediction_liter_per_day


def get_tolerance_values(config_data: Dict[str, Any]) -> Tuple[float, float, float]:
    tolerance_kg_305d = float(
        config_data.get(
            "tolerance_kg_305d",
            config_data.get("model_mae_mean", 1438.030363),
        )
    )

    tolerance_liter_305d = tolerance_kg_305d / KG_PER_LITER
    tolerance_liter_per_day = tolerance_liter_305d / DAYS_IN_TARGET

    return tolerance_kg_305d, tolerance_liter_305d, tolerance_liter_per_day


def make_range(center_value: float, tolerance_value: float) -> Tuple[float, float]:
    lower_value = max(
        0.0,
        center_value - tolerance_value,
    )

    upper_value = center_value + tolerance_value

    return lower_value, upper_value


config = load_config()
img_size = int(config.get("img_size", 224))
target_name = config.get("target_name", "milk_yield_305d")
model_type = config.get("model_type", "EfficientNetB0 feature extractor + classical regression")
best_regressor = config.get("best_regressor", "SVR_rbf_C10")
prediction_note = config.get(
    "prediction_note",
    "Prediction output is 305-day milk yield in kg before conversion to liter.",
)

tolerance_kg_305d, tolerance_liter_305d, tolerance_liter_per_day = get_tolerance_values(
    config
)

tolerance_note = config.get(
    "tolerance_note",
    "Tolerance range is estimated from cross-validation MAE. It is an error tolerance indicator, not a guaranteed confidence interval.",
)


st.title("🐄 Prediksi Produksi Susu Sapi Perah")

st.info(
    "Aplikasi ini menggunakan pipeline dua tahap: gambar sapi diproses oleh EfficientNetB0 "
    "sebagai feature extractor, lalu fitur gambar diprediksi oleh model regresi SVR RBF. "
    "Output model berupa estimasi 305-day milk yield dalam kg, kemudian dikonversi ke liter."
)

with st.expander("Informasi model", expanded=False):
    st.write(f"**Feature extractor:** EfficientNetB0")
    st.write(f"**Regressor:** {best_regressor}")
    st.write(f"**Pipeline:** {model_type}")
    st.write(f"**Ukuran input gambar:** {img_size} × {img_size} px")
    st.write(f"**Target output:** {target_name}")
    st.write("**Output mentah model:** kg/305 hari")
    st.write("**Output tampilan aplikasi:** liter/305 hari dan liter/hari")
    st.write(f"**Faktor konversi:** 1 liter susu ≈ {KG_PER_LITER:.2f} kg")
    st.write(f"**Catatan:** {prediction_note}")

    if "model_mae_mean" in config:
        st.write(f"**Cross-validation MAE model:** {float(config['model_mae_mean']):,.2f} kg")

    if "baseline_mae_mean" in config:
        st.write(f"**Baseline MAE:** {float(config['baseline_mae_mean']):,.2f} kg")

    if "r2_mean" in config:
        st.write(f"**R² rata-rata:** {float(config['r2_mean']):.4f}")

    st.write(
        f"**Toleransi estimasi:** ±{tolerance_liter_305d:,.2f} liter/305 hari "
        f"atau ±{tolerance_liter_per_day:,.2f} liter/hari"
    )
    st.caption(
        "Kisaran toleransi dihitung dari MAE cross-validation model. "
        "Ini bukan confidence interval statistik yang menjamin hasil aktual berada di dalam rentang tersebut."
    )

uploaded_file = st.file_uploader(
    "Upload gambar sapi perah",
    type=["jpg", "jpeg", "png"],
)

show_daily_average = st.checkbox(
    "Tampilkan estimasi rata-rata per hari dengan pembagian 305",
    value=True,
)

if uploaded_file is not None:
    preview_image = Image.open(uploaded_file)
    preview_image = ImageOps.exif_transpose(preview_image)
    preview_image = preview_image.convert("RGB")

    st.image(
        preview_image,
        caption="Gambar input",
        use_container_width=True,
    )

    if st.button("Prediksi produksi susu", type="primary"):
        with st.spinner("Memuat feature extractor, regressor, dan menjalankan prediksi..."):
            image_array = prepare_image(uploaded_file, img_size)
            predicted_kg_305d, predicted_liter_305d, predicted_liter_per_day = predict_milk_yield(
                image_array
            )

        st.subheader("Hasil Prediksi")

        st.metric(
            label=f"Estimasi {target_name}",
            value=f"{predicted_liter_305d:,.2f} liter/305 hari",
        )

        if show_daily_average:
            st.metric(
                label="Estimasi rata-rata per hari",
                value=f"{predicted_liter_per_day:,.2f} liter/hari",
            )

        lower_liter_305d, upper_liter_305d = make_range(
            predicted_liter_305d,
            tolerance_liter_305d,
        )

        lower_liter_per_day, upper_liter_per_day = make_range(
            predicted_liter_per_day,
            tolerance_liter_per_day,
        )

        lower_kg_305d, upper_kg_305d = make_range(
            predicted_kg_305d,
            tolerance_kg_305d,
        )

        st.subheader("Kisaran Toleransi Produksi")

        st.info(
            f"Perkiraan rentang produksi: **{lower_liter_305d:,.2f} – {upper_liter_305d:,.2f} liter/305 hari**. "
            f"Jika dirata-ratakan, kisarannya sekitar **{lower_liter_per_day:,.2f} – {upper_liter_per_day:,.2f} liter/hari**."
        )

        st.caption(
            f"Toleransi yang digunakan: ±{tolerance_liter_305d:,.2f} liter/305 hari "
            f"atau ±{tolerance_liter_per_day:,.2f} liter/hari, berdasarkan MAE cross-validation model."
        )

        with st.expander("Detail prediksi, toleransi, dan konversi", expanded=False):
            st.write(f"Prediksi mentah model: **{predicted_kg_305d:,.2f} kg/305 hari**")
            st.write(f"Hasil konversi: **{predicted_liter_305d:,.2f} liter/305 hari**")
            st.write(f"Rata-rata harian: **{predicted_liter_per_day:,.2f} liter/hari**")
            st.write(f"Rentang kg/305 hari: **{lower_kg_305d:,.2f} – {upper_kg_305d:,.2f} kg/305 hari**")
            st.write(f"Rentang liter/305 hari: **{lower_liter_305d:,.2f} – {upper_liter_305d:,.2f} liter/305 hari**")
            st.write(f"Rentang liter/hari: **{lower_liter_per_day:,.2f} – {upper_liter_per_day:,.2f} liter/hari**")
            st.write(f"Faktor konversi: **1 liter susu ≈ {KG_PER_LITER:.2f} kg**")
            st.write("Rumus konversi: **liter = kg / 1.03**")
            st.write("Rumus toleransi: **rentang = prediksi ± MAE cross-validation**")
            st.caption(tolerance_note)

        if predicted_kg_305d < 0:
            st.warning(
                "Model menghasilkan nilai negatif. Ini menandakan input mungkin jauh berbeda dari data training. "
                "Gunakan hasil sebagai indikasi bahwa prediksi tidak reliabel untuk gambar ini."
            )

        st.warning(
            "Gunakan hasil ini sebagai estimasi berbasis model, bukan sebagai pengukuran produksi aktual. "
            "Model bersifat eksperimental karena dataset training kecil dan performa cross-validation masih terbatas."
        )
else:
    st.caption("Upload gambar sapi perah untuk mulai melakukan prediksi.")

st.markdown(
    '''
    <div class="milk-footer">
        Developed by Galuh Adi Insani with training process at
        <a href="https://www.kaggle.com/code/adioranye/cow-milk-prediction-by-galuh-adi-insani/notebook" target="_blank" rel="noopener noreferrer">Kaggle</a>
    </div>
    ''',
    unsafe_allow_html=True,
)
