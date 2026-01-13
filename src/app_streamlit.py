import streamlit as st
import pandas as pd
import joblib
import re

# PAGE CONFIG
st.set_page_config(
    page_title="Prediksi Status Kredit",
    page_icon="💳",
    layout="centered"
)

# LOAD MODEL
MODEL_PATH = "../model/model_kredit.pkl"
ENCODER_PATH = "../model/label_encoder.pkl"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# CURRENCY HELPERS
def parse_currency(text):
    if not text:
        return 0
    return int(re.sub(r"[^\d]", "", text))

def format_currency(num):
    return f"Rp {num:,.0f}".replace(",", ".")

# CALLBACK FORMATTER
def format_input(key):
    raw = st.session_state[key]
    num = parse_currency(raw)
    st.session_state[key] = format_currency(num)

# BUSINESS LOGIC
def hitung_cicilan(pinjaman, bunga_persen, tenor):
    bunga = bunga_persen / 100
    bunga_total = bunga * (tenor / 12)
    total_bayar = pinjaman * (1 + bunga_total)
    return total_bayar / tenor

def validasi_input(gaji, pengeluaran, pinjaman, bunga):
    if pengeluaran >= gaji:
        return "Pengeluaran harus lebih kecil dari gaji"
    if pinjaman > gaji * 20:
        return "Pinjaman terlalu besar dibanding gaji"
    if bunga <= 0 or bunga > 100:
        return "Bunga tidak valid"
    return None

# INIT SESSION STATE
for k in ["gaji_raw", "pengeluaran_raw", "pinjaman_raw"]:
    if k not in st.session_state:
        st.session_state[k] = "Rp 0"

# UI HEADER
st.title("Prediksi Status Kredit")
st.caption("Sistem prediksi kelayakan kredit berbasis Machine Learning")

# SIDEBAR INPUT
with st.sidebar:
    st.header("Input Data")

    st.text_input(
        "Gaji Bulanan",
        key="gaji_raw",
        on_change=format_input,
        args=("gaji_raw",)
    )

    st.text_input(
        "Pengeluaran Bulanan",
        key="pengeluaran_raw",
        on_change=format_input,
        args=("pengeluaran_raw",)
    )

    st.text_input(
        "Jumlah Pinjaman",
        key="pinjaman_raw",
        on_change=format_input,
        args=("pinjaman_raw",)
    )

    tanggungan = st.number_input("Jumlah Tanggungan", min_value=0, max_value=10, step=1)
    tenor = st.selectbox("Tenor (Bulan)", [12, 24, 36, 48, 60])
    bunga = st.number_input("Bunga Tahunan (%)",
    min_value=5.00,
    max_value=30.00,
    value=12.00,
    step=0.25,
    format="%.2f")


    submit = st.button("🔍 Prediksi")

# PARSE INPUT
gaji = parse_currency(st.session_state.gaji_raw)
pengeluaran = parse_currency(st.session_state.pengeluaran_raw)
pinjaman = parse_currency(st.session_state.pinjaman_raw)

# PROCESS
if submit:
    error = validasi_input(gaji, pengeluaran, pinjaman, bunga)

    if error:
        st.error(error)
    else:
        cicilan = hitung_cicilan(pinjaman, bunga, tenor)

        rasio_pengeluaran = pengeluaran / gaji
        rasio_pinjaman = pinjaman / gaji
        rasio_cicilan = cicilan / gaji
        rasio_tanggungan = tanggungan / (gaji / 10_000_000)

        X_input = pd.DataFrame([{
            "gaji_bulanan": gaji,
            "pengeluaran_bulanan": pengeluaran,
            "jumlah_tanggungan": tanggungan,
            "jumlah_pinjaman": pinjaman,
            "tenor_bulan": tenor,
            "bunga_tahunan": bunga / 100,
            "rasio_pengeluaran": rasio_pengeluaran,
            "rasio_pinjaman": rasio_pinjaman,
            "rasio_cicilan": rasio_cicilan,
            "rasio_tanggungan": rasio_tanggungan
        }])

        pred = model.predict(X_input)
        status = label_encoder.inverse_transform(pred)[0]

        # OUTPUT
        st.subheader("Hasil Prediksi")

        if status == "Lancar":
            st.success("✅ Status Kredit: **Lancar**")
        elif status == "Kurang Lancar":
            st.warning("⚠️ Status Kredit: **Kurang Lancar**")
        else:
            st.error("❌ Status Kredit: **Macet**")

        col1, col2, col3 = st.columns(3)

        col1.markdown("**Cicilan / Bulan**")
        col1.markdown(f"### {format_currency(int(cicilan))}")

        col2.markdown("**Rasio Cicilan**")
        col2.markdown(f"### {rasio_cicilan:.2f}")

        sisa_uang = gaji - pengeluaran - cicilan
        col3.markdown("**Sisa Uang**")
        col3.markdown(f"### {format_currency(int(sisa_uang))}")


        st.subheader("Detail Rasio")
        st.write({
            "Rasio Pengeluaran": round(rasio_pengeluaran, 2),
            "Rasio Pinjaman": round(rasio_pinjaman, 2),
            "Rasio Tanggungan": round(rasio_tanggungan, 2)
        })
        
        # WARNING
        warnings = []
        if rasio_cicilan > 0.3:
            warnings.append("Cicilan relatif tinggi terhadap gaji")
        if rasio_pengeluaran > 0.6:
            warnings.append("Pengeluaran bulanan cukup besar")
        if rasio_tanggungan > 2:
            warnings.append("Tanggungan relatif tinggi terhadap gaji")

        if warnings:
            st.subheader("⚠️ Catatan Risiko")
            for w in warnings:
                st.write(f"- {w}")
