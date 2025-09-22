# типа_app.py
import os
import pickle
import pandas as pd
from datetime import timedelta

ML_MODEL_DIR = "ML_model"

# Какие переменные доступны:
PARAMS = [
    ("Температура", "°C", "Прогноз", "Минимум", "Максимум"),
    ("Скорость_ветра", "м/с", "Скорость ветра"),
    ("Давление", "гПа", "Давление"),
    ("Осадки", "мм", "Осадки"),
    ("По_ощущениям", "°C", "По ощущениям"),
]

# Загрузка всех прогнозов
def load_all_forecasts():
    fdict = {}
    for alias, *_ in PARAMS:
        forecast_fp = os.path.join(ML_MODEL_DIR, f"{alias}.forecast.pkl")
        if os.path.exists(forecast_fp):
            fdict[alias] = pd.read_pickle(forecast_fp)
    return fdict

def print_forecast_for_period(start_date, days=7):
    forecasts = load_all_forecasts()
    start = pd.to_datetime(start_date)
    end = start + timedelta(days=days - 1)
    all_dates = pd.date_range(start, end, freq="D")
    output = []

    for d in all_dates:
        row = {"Дата": d.strftime("%Y-%m-%d")}
        # Температура — особый случай
        temp_fc = forecasts.get("Температура")
        if temp_fc is not None:
            day = temp_fc[temp_fc["ds"] == d]
            if not day.empty:
                row["Прогноз"] = f"{day['yhat'].values[0]:.1f}°C"
                row["Минимум"] = f"{day['yhat_lower'].values[0]:.1f}°C"
                row["Максимум"] = f"{day['yhat_upper'].values[0]:.1f}°C"
            else:
                row["Прогноз"] = row["Минимум"] = row["Максимум"] = "-"
        else:
            row["Прогноз"] = row["Минимум"] = row["Максимум"] = "-"

        # Остальные параметры
        for alias, unit, colname in PARAMS[1:]:
            fc = forecasts.get(alias)
            val = "-"
            if fc is not None:
                t = fc[fc["ds"] == d]
                if not t.empty:
                    value = t["yhat"].values[0]
# По модулю осадки!
                    if colname == "Осадки":
                        value = abs(value)
                    val = f"{value:.1f} {unit}"
            row[colname] = val

        output.append(row)

    columns = ["Дата", "Прогноз", "Минимум", "Максимум", "По ощущениям", "Осадки", "Скорость ветра",]
    df_out = pd.DataFrame(output)
    for col in columns:
        if col not in df_out.columns:
            df_out[col] = "-"
    print(df_out[columns].to_string(index=False))

# Пример вызова
print_forecast_for_period('2025-09-20', 7)
