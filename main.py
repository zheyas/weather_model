import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import timedelta

# 1. Настройка шрифта для поддержки русского языка в графиках
plt.rcParams['font.family'] = 'DejaVu Sans'

# 2. Загрузка и очистка данных
df = pd.read_excel('whezer.xlsx', engine='openpyxl')
df.columns = [col.replace('\xa0', ' ').strip() for col in df.columns]  # Чистим заголовки столбцов
df = df.rename(columns={'Дата': 'ds', 'Средняя температура': 'y'})
df['ds'] = pd.to_datetime(df['ds'], dayfirst=True)
df = df.dropna(subset=['ds', 'y'])

# 3. Визуализация исторических данных
plt.figure(figsize=(14, 6))
plt.plot(df['ds'], df['y'], marker='.', linestyle='-', label='Средняя температура')
plt.xlabel('Дата')
plt.ylabel('Средняя температура, °C')
plt.title('Историческая средняя температура')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 4. Обучение модели Prophet
model = Prophet(yearly_seasonality=True, daily_seasonality=False, seasonality_mode='additive',
                weekly_seasonality=True)  # weekly_seasonality=True включаем для наглядности

model.fit(df)

# 5. Прогноз на следующий год
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# 6. Визуализация основного прогноза
fig1 = model.plot(forecast)
plt.title("Прогноз средней температуры на год вперёд")
plt.xlabel('Дата')
plt.ylabel('Средняя температура, °C')
plt.grid()
plt.tight_layout()
plt.show()

# 7. Визуализация тренда и сезонных компонент — вручную подпишем графики на русском!
fig2 = model.plot_components(forecast)
fig2.suptitle('Тренд и сезонные компоненты (детализация)', fontsize=16)
axes = fig2.get_axes()

# Верхний график (тренд)
axes[0].set_ylabel('Тренд\n(долгосрочное изменение температуры)')

# Средний график (неделя)
axes[1].set_ylabel('Еженедельная\nсезонность\n(календарные сдвиги)')
axes[1].set_xlabel('День недели')
axes[1].set_xticklabels(['ВС', 'ПН', 'ВТ', 'СР', 'ЧТ', 'ПТ', 'СБ'])

# Нижний график (год)
axes[2].set_ylabel('Годовая\nсезонность\n(времена года)')
axes[2].set_xlabel('День года')

plt.tight_layout()
plt.show()
print(df.columns.tolist())
# --- ФУНКЦИЯ ПРОГНОЗА НА ЛЮБУЮ НЕДЕЛЮ/ДИАПАЗОН ---


def print_forecast_for_period(start_date, days=7):
    """
    Универсальный прогноз на заданный период по температуре, ветру, давлению, осадкам и эффективной температуре.
    """
    start = pd.to_datetime(start_date)
    end = start + timedelta(days=days - 1)
    all_dates = pd.date_range(start, end, freq='D')

    # Чёткое соответствие: (colname, column_alias_for_output, формат)
    param_list = [
        ('Скорость ветра, м/с', 'Скорость ветра', '{:.1f} м/с'),
        ('Атмосферное давление, гПа', 'Давление', '{:.0f} гПа'),
        ('Осадки, мм', 'Осадки', '{:.1f} мм'),
        ('Эффективная температура', 'По ощущениям', '{:.1f}°C')
    ]

    # extra_forecasts: ключи - имена для вывода, значения - (forecast, fmt)
    extra_forecasts = dict()
    for col, alias, fmt in param_list:
        if col in df.columns:
            df_tmp = df[['ds', col]].dropna()
            if len(df_tmp) > 10:
                m = Prophet(yearly_seasonality=True, seasonality_mode='additive')
                m.fit(df_tmp.rename(columns={col: 'y'}))
                future = m.make_future_dataframe(periods=365)
                f = m.predict(future)
                extra_forecasts[alias] = (f, fmt)

    # Собираем итоговую таблицу
    output = []
    for d in all_dates:
        row = {'Дата': d.strftime('%Y-%m-%d')}
        temp_row = forecast[forecast['ds'] == d]
        if not temp_row.empty:
            row['Прогноз'] = f"{temp_row['yhat'].values[0]:.1f}°C"
            row['Минимум'] = f"{temp_row['yhat_lower'].values[0]:.1f}°C"
            row['Максимум'] = f"{temp_row['yhat_upper'].values[0]:.1f}°C"
        else:
            row['Прогноз'] = row['Минимум'] = row['Максимум'] = '-'
        for _, alias, fmtstr in param_list:
            val = '-'
            if alias in extra_forecasts:
                fc, _ = extra_forecasts[alias]
                t = fc[fc['ds'] == d]
                if not t.empty:
                    val_value = t['yhat'].values[0]
                    if alias == 'Осадки':
                        val_value = abs(val_value)
                    val = fmtstr.format(val_value)
            row[alias] = val
        output.append(row)

    columns = ['Дата', 'Прогноз', 'Минимум', 'Максимум',
               'По ощущениям', 'Осадки', 'Скорость ветра']
    df_out = pd.DataFrame(output)
    for col in columns:
        if col not in df_out.columns:
            df_out[col] = '-'

    print('\nПояснение: "По ощущениям" — эффективная температура (температура, ощущаемая человеком, учитывает ветер, влажность и др. условия).\n')
    print(df_out[columns].to_string(index=False))


print_forecast_for_period('2025-09-22',7)