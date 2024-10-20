#importar as bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#importar o csv
data = pd.read_csv('D:/project_hub/Linear-Regression for sales/input/VendasMensais_processed.csv', parse_dates=['Period'])
data.set_index('Period', inplace=True)
data = data.asfreq('MS')  # Define a frequência como mensal

#identificar a data mais recente na coluna Period
last_date = data.index[-1]
last_year = last_date.year
last_month = last_date.month

#gerar proximas datas
def future_dates(last_year, last_month):
    future_steps = 24 #2 anos
    future_dates = []
    for i in range(future_steps):
        last_month += 1
        if last_month > 12:
            last_month = 1
            last_year += 1
        future_dates.append(f'{last_year}-{last_month:02d}-01')
    return future_dates

#gerar as proximas datas
future_dates_list = future_dates(last_year, last_month)
print(future_dates_list)

#definir as variaveis para o modelo
y = data['Revenue']

#treinar o modelo com auto_arima
best_model = auto_arima(y, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=False, max_order=None, stepwise=True)
print(best_model.summary())

#utilizar o modelo para prever as proximas datas (future_dates)
future_steps = 24
future_forecast = best_model.predict(n_periods=future_steps)

#df com previsoes futuras
future_dates_df = pd.DataFrame({'Predicted_Revenue': future_forecast}, index=pd.to_datetime(future_dates_list))

#exibir previsoes futuras
print(future_dates_df)

#calcular as metricas de precisao  
y_true = data['Revenue'][-24:]  # Ultimos 24 meses como dados reais
y_pred = future_forecast  # Previsões feitas pelo modelo

#calcular as metricas
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

#rmse, mae, r2
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R2: {r2}')

#graficos
plt.figure(figsize=(12, 6))
plt.plot(data.index, y, label='Dados Históricos', color='blue')
plt.plot(future_dates_df.index, future_dates_df['Predicted_Revenue'], label='Previsão Futura', color='green')
plt.title('Previsão Futura com ARIMA')
plt.xlabel('Data')
plt.ylabel('Receita')
plt.legend()
plt.grid()
plt.show()
