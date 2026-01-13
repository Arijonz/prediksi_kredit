import joblib
import pandas as pd

# LOAD MODEL & ENCODER
MODEL_PATH = "../model/model_kredit.pkl"
ENCODER_PATH = "../model/label_encoder.pkl"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# HELPER FUNCTIONS
def hitung_cicilan(pinjaman, bunga_persen_tahun, tenor):
    bunga = bunga_persen_tahun / 100
    bunga_total = bunga * (tenor / 12)
    total_bayar = pinjaman * (1 + bunga_total)
    return total_bayar / tenor

def validasi_input(gaji, pengeluaran, tanggungan, pinjaman, tenor, bunga):
    if gaji <= 0:
        return "Gaji harus lebih dari 0"
    if pengeluaran < 0 or pengeluaran >= gaji:
        return "Pengeluaran harus lebih kecil dari gaji"
    if tanggungan < 0:
        return "Jumlah tanggungan tidak valid"
    if pinjaman <= 0:
        return "Jumlah pinjaman harus lebih dari 0"
    if pinjaman > gaji * 20:
        return "Jumlah pinjaman terlalu besar dibanding gaji bulanan"
    if tenor <= 0:
        return "Tenor harus lebih dari 0"
    if bunga <= 0 or bunga > 100:
        return "Bunga tahunan diisi dalam persen (contoh: 12)"
    return None

# APP START
print(" APLIKASI PREDIKSI STATUS KREDIT \n")

try:
    gaji = float(input("Gaji bulanan (Rp): "))
    pengeluaran = float(input("Pengeluaran bulanan (Rp): "))
    tanggungan = int(input("Jumlah tanggungan: "))
    pinjaman = float(input("Jumlah pinjaman (Rp): "))
    tenor = int(input("Tenor (bulan): "))
    bunga_persen = float(input("Bunga tahunan (%): "))

    error = validasi_input(gaji, pengeluaran, tanggungan, pinjaman, tenor, bunga_persen)
    if error:
        print(f"\n Error input: {error}")
        exit()

    # FEATURE ENGINEERING
    cicilan = hitung_cicilan(pinjaman, bunga_persen, tenor)

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
        "bunga_tahunan": bunga_persen / 100,
        "rasio_pengeluaran": rasio_pengeluaran,
        "rasio_pinjaman": rasio_pinjaman,
        "rasio_cicilan": rasio_cicilan,
        "rasio_tanggungan": rasio_tanggungan
    }])

    # PREDICTION
    pred_encoded = model.predict(X_input)
    status_raw = label_encoder.inverse_transform(pred_encoded)[0]

    # NORMALISASI STATUS (AMAN & KONSISTEN)
    status_map = {
        "Lancar": "Lancar",
        "Kurang Lancar": "Kurang Lancar",
        "Macet": "Macet"
    }
    status = status_map.get(status_raw, status_raw)

    # WARNING SYSTEM (NON-FATAL)
    warnings = []

    if rasio_cicilan > 0.3:
        warnings.append("Cicilan tergolong tinggi dibanding gaji")

    if rasio_pengeluaran > 0.6:
        warnings.append("Pengeluaran bulanan cukup besar")

    if rasio_tanggungan > 2:
        warnings.append("Jumlah tanggungan relatif tinggi terhadap gaji")

    if rasio_pinjaman > 10:
        warnings.append("Pinjaman sangat besar dibanding gaji bulanan")

    # OUTPUT
    print("\n HASIL PREDIKSI ")
    print(f"Status Kredit     : {status}")
    print(f"Cicilan / Bulan   : Rp {cicilan:,.0f}")
    print(f"Rasio Pengeluaran : {rasio_pengeluaran:.2f}")
    print(f"Rasio Cicilan     : {rasio_cicilan:.2f}")
    print(f"Rasio Tanggungan  : {rasio_tanggungan:.2f}")
    print(f"Sisa Uang         : Rp {gaji - pengeluaran - cicilan:,.0f}")

    if warnings:
        print("\n CATATAN RISIKO:")
        for w in warnings:
            print(f"- {w}")
    else:
        print("\n Aman Untuk Mengajukan Kredit")

except Exception as e:
    print(f"\n Terjadi kesalahan: {e}")
