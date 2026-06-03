import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps


APP_TITLE = "Prediksi Produksi Susu Sapi Perah"
MODEL_PATH = Path("milk_yield_model.keras")
CONFIG_PATH = Path("preprocess_config.json")


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🐄",
    layout="centered",
)



def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            /* Hide Streamlit default branding and toolbar elements. */
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
def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    return {
        "img_size": 224,
        "model_type": "Image regression",
        "target_name": "milk_yield",
        "output_note": "Output mengikuti satuan target training.",
    }


@st.cache_resource
def load_prediction_model():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        st.error(
            "TensorFlow belum terinstal. Di Streamlit Community Cloud, pilih Python 3.11 atau 3.12 "
            "pada Advanced settings, lalu redeploy aplikasi. Python 3.14 belum kompatibel "
            "dengan paket TensorFlow yang dipakai aplikasi ini."
        )
        raise error

    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False,
        )
    except TypeError:
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )


def prepare_image(uploaded_file, img_size: int) -> np.ndarray:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize((img_size, img_size))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


config = load_config()
img_size = int(config.get("img_size", 224))
target_name = config.get("target_name", "milk_yield")
model_type = config.get("model_type", "Image regression")
output_note = config.get("output_note", "Output mengikuti satuan target training.")
source_unit = config.get("source_unit", "kg")
display_unit = config.get("output_unit", "liter")
kg_per_liter = float(config.get("kg_per_liter", 1.03))


st.title("🐄 Prediksi Produksi Susu Sapi Perah")

st.info(
    "Model ini menerima gambar sapi perah dan menghasilkan prediksi numerik. "
    "Nilai asli model mengikuti target saat training. Karena target training dianggap kg/305 hari, "
    "aplikasi ini mengonversi hasil prediksi menjadi liter dengan asumsi 1 liter susu ≈ 1.03 kg."
)

with st.expander("Informasi model", expanded=False):
    st.write(f"**Model:** {model_type}")
    st.write(f"**Ukuran input gambar:** {img_size} × {img_size} px")
    st.write(f"**Target output:** {target_name}")
    st.write(f"**Satuan sumber model:** {source_unit}/305 hari")
    st.write(f"**Satuan tampilan aplikasi:** {display_unit}/305 hari")
    st.write(f"**Faktor konversi:** 1 liter susu ≈ {kg_per_liter:.2f} kg")
    st.write(f"**Catatan:** {output_note}")

main_unit_label = "liter/305 hari"
daily_unit_label = "liter/hari"

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
        if not MODEL_PATH.exists():
            st.error("File model milk_yield_model.keras tidak ditemukan.")
            st.stop()

        with st.spinner("Memuat model dan menjalankan prediksi..."):
            model = load_prediction_model()
            image_array = prepare_image(uploaded_file, img_size)
            prediction = model.predict(image_array, verbose=0)

        predicted_kg_305d = float(np.ravel(prediction)[0])
        predicted_liter_305d = predicted_kg_305d / kg_per_liter

        st.subheader("Hasil Prediksi")

        st.metric(
            label=f"Estimasi {target_name}",
            value=f"{predicted_liter_305d:,.2f} {main_unit_label}",
        )

        if show_daily_average:
            daily_average_liter = predicted_liter_305d / 305

            st.metric(
                label="Estimasi rata-rata per hari",
                value=f"{daily_average_liter:,.2f} {daily_unit_label}",
            )

        with st.expander("Detail konversi", expanded=False):
            st.write(f"Prediksi mentah model: **{predicted_kg_305d:,.2f} kg/305 hari**")
            st.write(f"Faktor konversi: **1 liter susu ≈ {kg_per_liter:.2f} kg**")
            st.write("Rumus: **liter = kg / 1.03**")

        st.caption(
            f"Satuan hasil utama: {main_unit_label}. "
            f"Satuan rata-rata harian: {daily_unit_label}."
        )

        st.warning(
            "Gunakan hasil ini sebagai estimasi berbasis model, bukan sebagai pengukuran produksi aktual. "
            "Akurasi sangat bergantung pada kualitas dataset training, sudut foto, pencahayaan, dan kemiripan gambar input dengan data training."
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
