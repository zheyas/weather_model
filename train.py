# train.py
import os
import pandas as pd
from prophet import Prophet
import pickle

# 1. Загрузка и очистка данных
df = pd.read_excel('whezer.xlsx', engine='openpyxl')
df.columns = [col.replace('\xa0', ' ').strip() for col in df.columns]
df = df.rename(columns={'Дата': 'ds', 'Средняя температура': 'y'})
df['ds'] = pd.to_datetime(df['ds'], dayfirst=True)
df = df.dropna(subset=['ds', 'y'])

os.makedirs("ML_model", exist_ok=True)

# --- Перечень параметров ---
param_list = [
    ('y', "Температура"),
    ('Скорость ветра, м/с', "Скорость_ветра"),
    ('Атмосферное давление, гПа', "Давление"),
    ('Осадки, мм', "Осадки"),
    ('Эффективная температура', "По_ощущениям"),
]

# --- Обучение и сохранение моделей ---
for col, alias in param_list:
    if col not in df.columns:
        print(f"Нет данных для {alias}, пропускаем")
        continue
    this_df = df[['ds', col]].dropna().rename(columns={col:'y'})
    if len(this_df) < 10:
        print(f"Слишком мало данных для {alias}, пропускаем")
        continue

    print(f"Обучаем Prophet для: {alias}")
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, seasonality_mode='additive')
    model.fit(this_df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Сохраняем модель и прогноз
    with open(f"ML_model/{alias}.prophet.pkl", "wb") as f:
        pickle.dump(model, f)
    forecast.to_pickle(f"ML_model/{alias}.forecast.pkl")

print("Все возможные модели обучены и сохранены.")
