import requests
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pandas as pd
from datetime import timedelta

app = Flask(__name__)

# Папка со статикой (CSS/JS) — Flask автоматически ищет 'static'
app.static_folder = "static"

ML_MODEL_DIR = "ML_model"

# Доступные параметры (имена должны совпадать с названиями файлов)
PARAMS = [
    ("Температура", "°C", "Прогноз", "Минимум", "Максимум"),
    ("Скорость_ветра", "м/с", "Скорость ветра"),
    ("Давление", "гПа", "Давление"),
    ("Осадки", "мм", "Осадки"),
    ("По_ощущениям", "°C", "По ощущениям"),
]

def get_weather_openmeteo(lat, lon):
    """
    Получает погоду через Open-Meteo API (без ключа)
    """
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        'latitude': lat,
        'longitude': lon,
        'current': 'temperature_2m,relative_humidity_2m,apparent_temperature,pressure_msl,wind_speed_10m,weather_code',
        'timezone': 'auto',
        'forecast_days': 1
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        current = data['current']

        # Конвертируем код погоды в текст
        weather_codes = {
            0: 'ясно', 1: 'преимущественно ясно', 2: 'переменная облачность',
            3: 'пасмурно', 45: 'туман', 48: 'туман', 51: 'легкая морось',
            53: 'умеренная морось', 55: 'сильная морось', 61: 'небольшой дождь',
            63: 'умеренный дождь', 65: 'сильный дождь', 80: 'ливень',
            81: 'сильный ливень', 82: 'очень сильный ливень'
        }

        weather_description = weather_codes.get(current['weather_code'], 'неизвестно')

        weather_info = {
            'город': 'Оренбург (по координатам)',
            'температура': f"{current['temperature_2m']}°C",
            'ощущается_как': f"{current['apparent_temperature']}°C",
            'погода': weather_description,
            'влажность': f"{current['relative_humidity_2m']}%",
            'давление': f"{current['pressure_msl']} гПа",
            'ветер': f"{current['wind_speed_10m']} км/ч",
            'источник': 'Open-Meteo'
        }

        return weather_info

    except Exception as e:
        return {'ошибка': f'Ошибка запроса: {e}'}


def load_all_forecasts():
    """Загрузка всех прогнозов из папки ML_model"""
    fdict = {}
    for alias, *_ in PARAMS:
        forecast_fp = os.path.join(ML_MODEL_DIR, f"{alias}.forecast.pkl")
        if os.path.exists(forecast_fp):
            fdict[alias] = pd.read_pickle(forecast_fp)
    return fdict

def get_forecast(start_date, days=7):
    """Возвращает прогноз для указанного периода в формате списка словарей"""
    forecasts = load_all_forecasts()
    start = pd.to_datetime(start_date) + timedelta(days=1)  # увеличиваем на 1 день
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
                    if colname == "Осадки":
                        value = abs(value)
                    val = f"{value:.1f} {unit}"
            row[colname] = val

        output.append(row)
    return output

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/api/forecast", methods=["GET"])
def api_forecast():
    start_date = request.args.get("date")
    days = int(request.args.get("days", 7))
    forecast_data = get_forecast(start_date, days)
    return jsonify(forecast_data)

# Явное подключение статики (необязательно, Flask раздает автоматом)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)


@app.route('/api/weather')
def api_weather():
    """API endpoint для получения погоды"""
    lat = 51.7727  # Широта Оренбурга
    lon = 55.0988  # Долгота Оренбурга

    weather_data = get_weather_openmeteo(lat, lon)
    return jsonify(weather_data)


@app.route('/api/weather/<int:days>')
def api_weather_forecast(days):
    """API endpoint для получения прогноза на N дней"""
    lat = 51.7727  # Широта Оренбурга
    lon = 55.0988  # Долгота Оренбурга

    # Ограничиваем максимальное количество дней
    forecast_days = min(max(1, days), 21)

    weather_data = get_weather_openmeteo(lat, lon)
    return jsonify(weather_data)


if __name__ == "__main__":
    app.run(debug=True)
