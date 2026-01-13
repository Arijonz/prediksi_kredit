import pandas as pd
import numpy as np
import random
import os

# ===============================
# KONFIGURASI
# ===============================
ROWS = 800
OPEN_HOUR = 8
CLOSE_HOUR = 23
JAM_OPERASIONAL = CLOSE_HOUR - OPEN_HOUR

random.seed(42)
np.random.seed(42)

data = []

for _ in range(ROWS):
    hour = random.randint(OPEN_HOUR, CLOSE_HOUR - 1)
    day_of_week = random.randint(0, 6)
    is_weekend = 1 if day_of_week >= 5 else 0

    is_nyepi = 1 if random.random() < 0.01 else 0
    is_galungan = 1 if random.random() < 0.03 else 0
    is_kuningan = 1 if random.random() < 0.02 else 0
    is_saraswati = 1 if random.random() < 0.02 else 0
    is_national_holiday = 1 if random.random() < 0.04 else 0

    has_event = 1 if random.random() < 0.25 else 0
    event_scale = random.randint(1, 3) if has_event else 0

    has_promo = 1 if random.random() < 0.35 else 0
    avg_daily_visitors = random.choice([60, 80, 100, 120, 150])

    baseline = avg_daily_visitors / JAM_OPERASIONAL
    score = baseline

    if 17 <= hour <= 21:
        score += baseline * 0.6
    elif hour <= 10:
        score -= baseline * 0.4

    if is_weekend:
        score += baseline * 0.3

    if has_event:
        score += baseline * (0.2 * event_scale)

    if has_promo:
        score += baseline * 0.2

    if is_galungan or is_saraswati:
        score += baseline * 0.4

    if is_nyepi:
        traffic_level = 0
    else:
        if score < baseline * 0.7:
            traffic_level = 0
        elif score < baseline * 1.3:
            traffic_level = 1
        else:
            traffic_level = 2

    data.append([
        hour, day_of_week, is_weekend,
        is_national_holiday, is_nyepi,
        is_galungan, is_kuningan, is_saraswati,
        has_event, event_scale,
        has_promo, avg_daily_visitors,
        traffic_level
    ])

columns = [
    "hour","day_of_week","is_weekend",
    "is_national_holiday","is_nyepi",
    "is_galungan","is_kuningan","is_saraswati",
    "has_event","event_scale",
    "has_promo","avg_daily_visitors",
    "traffic_level"
]

df = pd.DataFrame(data, columns=columns)

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/coffee_traffic_raw.csv", index=False)

print("✅ Dataset 800 baris berhasil dibuat")
