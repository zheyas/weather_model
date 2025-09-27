import os
import pandas as pd
from prophet import Prophet
import pickle


def retrain_models():
    try:
        # 1. Проверяем существование файла
        if not os.path.exists('whezer.xlsx'):
            print("Файл whezer.xlsx не найден")
            return False

        # 2. Загрузка данных
        df = pd.read_excel('whezer.xlsx', engine='openpyxl')

        # 3. Проверяем, есть ли достаточно данных
        if len(df) < 2:  # Минимум 2 записи для обучения
            print("Недостаточно данных для обучения моделей")
            return False

        # 4. Очищаем названия колонок
        df.columns = [col.replace('\xa0', ' ').strip() for col in df.columns]

        print(f"Загружено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")

        # 5. Проверяем наличие обязательных колонок
        required_columns = ['Дата', 'Средняя температура']
        for col in required_columns:
            if col not in df.columns:
                print(f"Отсутствует обязательная колонка: {col}")
                print(f"Доступные колонки: {list(df.columns)}")
                return False

        # 6. Подготовка данных
        df = df.rename(columns={'Дата': 'ds', 'Средняя температура': 'y'})

        # Конвертируем дату с учетом формата DD.MM.YYYY
        df['ds'] = pd.to_datetime(df['ds'], format='%d.%m.%Y', errors='coerce')
        df = df.dropna(subset=['ds', 'y'])

        if len(df) < 2:
            print("Недостаточно данных после очистки")
            return False

        print(f"Осталось {len(df)} записей после очистки")
        print(f"Диапазон дат: от {df['ds'].min()} до {df['ds'].max()}")

        # 7. Создаем папку для моделей
        os.makedirs("ML_model", exist_ok=True)

        # 8. Перечень параметров для обучения
        param_list = [
            ('y', "Температура"),
        ]

        # Добавляем дополнительные параметры если они есть в данных
        additional_params = [
            ('Скорость ветра, м/с', "Скорость_ветра"),
            ('Атмосферное давление, гПа', "Давление"),
            ('Осадки, мм', "Осадки"),
            ('Эффективная температура', "По_ощущениям"),
        ]

        original_columns_mapping = {
            'Скорость ветра, м/с': 'Скорость ветра, м/с',
            'Атмосферное давление, гПа': 'Атмосферное давление, гПа',
            'Осадки, мм': 'Осадки, мм',
            'Эффективная температура': 'Эффективная температура'
        }

        # Восстанавливаем оригинальные названия колонок для поиска
        df_original = pd.read_excel('whezer.xlsx', engine='openpyxl')
        df_original.columns = [col.replace('\xa0', ' ').strip() for col in df_original.columns]

        for col, alias in additional_params:
            if col in df_original.columns:
                # Добавляем данные из оригинального DataFrame
                df[alias] = df_original[col]
                param_list.append((alias, alias))
                print(f"Добавлен параметр для обучения: {alias}")

        # 9. Обучение и сохранение моделей
        trained_models = 0
        for col, alias in param_list:
            if col not in df.columns:
                print(f"Нет данных для {alias}, пропускаем")
                continue

            this_df = df[['ds', col]].dropna().rename(columns={col: 'y'})

            if len(this_df) < 2:
                print(f"Слишком мало данных для {alias} ({len(this_df)} записей), пропускаем")
                continue

            print(f"Обучаем Prophet для: {alias} (на основе {len(this_df)} записей)")

            try:
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='additive'
                )
                model.fit(this_df)

                # Создаем прогноз на будущее
                future = model.make_future_dataframe(periods=365, include_history=False)
                forecast = model.predict(future)

                # Сохраняем модель и прогноз
                with open(f"ML_model/{alias}.prophet.pkl", "wb") as f:
                    pickle.dump(model, f)
                forecast.to_pickle(f"ML_model/{alias}.forecast.pkl")

                trained_models += 1
                print(f"Модель {alias} успешно обучена и сохранена")

            except Exception as e:
                print(f"Ошибка при обучении модели {alias}: {e}")

        print(f"Обучено {trained_models} моделей из {len(param_list)} возможных")
        return trained_models > 0

    except Exception as e:
        print(f"Ошибка при обучении моделей: {e}")
        return False


if __name__ == "__main__":
    success = retrain_models()
    if success:
        print("Обучение моделей завершено успешно")
    else:
        print("Обучение моделей завершено с ошибками")