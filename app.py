import requests
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pandas as pd
from datetime import timedelta, datetime
import schedule
import threading
import time
import subprocess

app = Flask(__name__)

# Папка со статикой (CSS/JS) — Flask автоматически ищет 'static'
app.static_folder = "static"

ML_MODEL_DIR = "ML_model"
DATA_FILE = "whezer.xlsx"

# Доступные параметры (имена должны совпадать с названиями файлов)
PARAMS = [
    ("Температура", "°C", "Прогноз", "Минимум", "Максимум"),
    ("Скорость_ветра", "м/с", "Скорость ветра"),
    ("Давление", "гПа", "Давление"),
    ("Осадки", "мм", "Осадки"),
    ("По_ощущениям", "°C", "По ощущениям"),
]


def get_weather_openmeteo(lat, lon):
    """Получает погоду через Open-Meteo API (без ключа)"""
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


def get_detailed_weather_data_hourly(lat, lon):
    """Получаем почасовые данные и рассчитываем дневные значения"""
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m,apparent_temperature,pressure_msl,wind_speed_10m,precipitation',
        'timezone': 'auto',
        'forecast_days': 1
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        hourly = data['hourly']
        today = datetime.now().strftime('%d.%m.%Y')

        # Рассчитываем дневные значения из почасовых данных
        temperatures = hourly['temperature_2m']
        apparent_temps = hourly['apparent_temperature']
        pressures = hourly['pressure_msl']
        wind_speeds = hourly['wind_speed_10m']
        precipitations = hourly['precipitation']

        weather_data = {
            'Дата': today,
            'Максимальная температура': round(max(temperatures), 1),
            'Минимальная температура': round(min(temperatures), 1),
            'Средняя температура': round(sum(temperatures) / len(temperatures), 1),
            'Атмосферное давление, гПа': round(sum(pressures) / len(pressures), 1),
            'Скорость ветра, м/с': round(max(wind_speeds), 1),
            'Осадки, мм': round(sum(precipitations), 1),
            'Эффективная температура': round(sum(apparent_temps) / len(apparent_temps), 1)
        }

        return weather_data
    except Exception as e:
        print(f"Ошибка получения почасовых данных: {e}")
        return None


def get_detailed_weather_data(lat, lon):
    """Получает детальные данные погоды для записи в файл"""
    return get_detailed_weather_data_hourly(lat, lon)


def save_weather_data_to_excel():
    """Сохраняет данные погоды в Excel-файл, если записи за сегодня еще нет"""
    try:
        lat, lon = 51.7727, 55.0988  # Координаты Оренбурга
        weather_data = get_detailed_weather_data(lat, lon)

        if not weather_data:
            print("Не удалось получить данные погоды")
            return False

        today = datetime.now().strftime('%d.%m.%Y')
        print(f"Получены данные за {today}: {weather_data}")

        # Проверяем существование файла
        if os.path.exists(DATA_FILE):
            try:
                # Читаем существующий файл без изменения шапки
                df_existing = pd.read_excel(DATA_FILE)

                # Проверяем, есть ли уже запись за сегодня
                if today in df_existing['Дата'].astype(str).values:
                    print(f"Запись за {today} уже существует")
                    return True

                # Добавляем новую запись
                df_new = pd.DataFrame([weather_data])
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)

                # Сохраняем файл без перезаписи шапки
                with pd.ExcelWriter(DATA_FILE, engine='openpyxl', mode='w') as writer:
                    df_combined.to_excel(writer, index=False)

                print(f"Данные за {today} успешно сохранены")

                # Обучаем модель после добавления новых данных
                retrain_models()
                return True

            except Exception as e:
                print(f"Ошибка чтения/записи существующего файла: {e}")
                return False
        else:
            # Создаем новый файл с правильной шапкой
            df_new = pd.DataFrame([weather_data])

            # Убедимся, что порядок колонок правильный
            correct_columns = [
                'Дата', 'Максимальная температура', 'Минимальная температура',
                'Средняя температура', 'Атмосферное давление, гПа',
                'Скорость ветра, м/с', 'Осадки, мм', 'Эффективная температура'
            ]

            # Переупорядочиваем колонки если нужно
            for col in correct_columns:
                if col not in df_new.columns:
                    df_new[col] = None

            df_new = df_new[correct_columns]
            df_new.to_excel(DATA_FILE, index=False)
            print(f"Создан новый файл {DATA_FILE}")

            # Обучаем модель после создания файла с новыми данными
            retrain_models()
            return True

    except Exception as e:
        print(f"Ошибка сохранения данных: {e}")
        return False


def retrain_models():
    """Переобучает модели машинного обучения"""
    try:
        # Запускаем скрипт обучения
        result = subprocess.run(['python', 'train.py'],
                                capture_output=True, text=True)

        if result.returncode == 0:
            print("Модели успешно переобучены")
            print(result.stdout)
            return True
        else:
            print("Ошибка переобучения моделей:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"Ошибка при переобучении моделей: {e}")
        return False


def schedule_tasks():
    """Запускает планировщик задач"""
    # Ежедневное сохранение данных в 01:00
    schedule.every().day.at("01:00").do(save_weather_data_to_excel)

    # Ежедневное переобучение моделей в 02:00
    schedule.every().day.at("02:00").do(retrain_models)

    while True:
        schedule.run_pending()
        time.sleep(60)  # Проверяем каждую минуту


def start_scheduler():
    """Запускает планировщик в отдельном потоке"""
    scheduler_thread = threading.Thread(target=schedule_tasks)
    scheduler_thread.daemon = True
    scheduler_thread.start()


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


@app.route('/api/save-weather-data', methods=['POST'])
def api_save_weather_data():
    """Ручное сохранение данных погоды"""
    success = save_weather_data_to_excel()
    return jsonify({'success': success})


@app.route('/api/retrain-models', methods=['POST'])
def api_retrain_models():
    """Ручное переобучение моделей"""
    success = retrain_models()
    return jsonify({'success': success})


@app.route('/api/check-data')
def api_check_data():
    """Проверка существующих данных"""
    try:
        if os.path.exists(DATA_FILE):
            df = pd.read_excel(DATA_FILE)
            return jsonify({
                'exists': True,
                'records': len(df),
                'columns': list(df.columns),
                'last_records': df.tail(3).to_dict('records')
            })
        else:
            return jsonify({'exists': False, 'records': 0})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    # Запускаем планировщик задач
    start_scheduler()

    # Сохраняем данные при запуске приложения (если еще не сохранены за сегодня)
    print("Запуск приложения...")
    save_weather_data_to_excel()

    app.run(debug=True, host='0.0.0.0', port=5050)