import requests
import pandas as pd
import numpy as np
import ta
import pytz
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score

# --- CONFIGURACIÓN ---
SYMBOL = 'BTC-USDT'
TIME_FRAMES = ['5min', '15min', '1hour', '4hour']
TF_PRINCIPAL = '15min'
UMBRAL_PROBABILIDAD = 0.70 
MIN_CONFIANZA_MODELO = 0.60 # Tu observación: Filtro de 60%
ZONA_HORARIA = pytz.timezone('America/Bogota')

def get_kucoin_klines(symbol, timeframe):
    endpoint = 'https://api.kucoin.com/api/v1/market/candles'
    params = {'symbol': symbol, 'type': timeframe}
    try:
        response = requests.get(endpoint, params=params)
        data = response.json()
        if data['code'] == '200000' and data['data']:
            df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            for col in ['open', 'close', 'high', 'low', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.sort_values('timestamp', ascending=True)
            return df.set_index('timestamp')
        return None
    except Exception: return None

def add_technical_indicators(df):
    df = df.copy()
    
    # INDICADORES EXISTENTES
    df['EMA_55'] = ta.trend.ema_indicator(df['close'], window=55)
    df['DIST_EMA'] = (df['close'] - df['EMA_55']) / df['EMA_55']
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    adx_obj = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['ADX'] = adx_obj.adx()
    df['MFI'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
    
    # --- NUEVOS INDICADORES PARA PRECISIÓN ---
    # 1. Williams %R (Ayuda a ver si el movimiento fuerte terminó)
    df['WILLR'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
    
    # 2. Estocástico (Para filtrar rebotes falsos)
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['STOCH_K'] = stoch.stoch()
    
    # 3. OBV (Volumen acumulado para confirmar tendencia)
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['OBV_ROC'] = df['OBV'].pct_change(periods=5) # Cambio porcentual del volumen
    
    # 4. ATR Normalizado (Volatilidad)
    df['ATR_NORM'] = ta.volatility.average_true_range(df['high'], df['low'], df['close']) / df['close']
    
    return df.dropna()

def send_telegram_alert(message):
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={message}"
    try: requests.get(url)
    except: pass

def run_prediction_cycle():
    results = {}
    features = ['RSI', 'DIST_EMA', 'ADX', 'MFI', 'WILLR', 'STOCH_K', 'OBV_ROC', 'ATR_NORM']

    for tf in TIME_FRAMES:
        df = get_kucoin_klines(SYMBOL, tf)
        if df is not None and len(df) > 60:
            df_ind = add_technical_indicators(df)
            df_ind['target'] = (df_ind['close'].shift(-1) > df_ind['close']).astype(int)
            df_ind.dropna(inplace=True)

            X = df_ind[features]
            y = df_ind['target']

            split = int(len(X) * 0.8)
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X.iloc[:split])
            X_test = scaler.transform(X.iloc[split:])
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            model = RandomForestClassifier(n_estimators=250, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            acc = accuracy_score(y_test, model.predict(X_test))
            last_row = df_ind.tail(1)
            prob_up = model.predict_proba(scaler.transform(last_row[features]))[0][1]
            
            ts_utc = last_row.index[0].tz_localize('UTC')
            results[tf] = {
                'prob': prob_up, 'acc': acc, 'price': df_ind['close'].iloc[-1],
                'adx': last_row['ADX'].iloc[0], 'timestamp_marca': ts_utc.astimezone(ZONA_HORARIA).strftime('%H:%M:%S')
            }

    if TF_PRINCIPAL in results:
        res_p = results[TF_PRINCIPAL]
        es_subida = res_p['prob'] >= 0.5
        prob_final = res_p['prob'] if es_subida else (1 - res_p['prob'])
        direccion = "Subir 📈" if es_subida else "Bajar 📉"

        # --- FILTROS DE ALTA EXIGENCIA ---
        mismo_sentido = (results['1hour']['prob'] >= 0.5) == es_subida
        
        # CONDICIÓN: Probabilidad > 70% Y Confianza > 60% Y Confluencia con 1H
        if prob_final >= UMBRAL_PROBABILIDAD and res_p['acc'] >= MIN_CONFIANZA_MODELO and mismo_sentido:
            
            meta_label = "💎 SEÑAL DE ALTA CONFLUENCIA"
            hora_actual = pd.Timestamp.now(tz='UTC').astimezone(ZONA_HORARIA).strftime('%H:%M:%S')

            msg = (f"{meta_label}\n\n"
                   f"🔮 *Predicción ({TF_PRINCIPAL}):* *{direccion}*\n"
                   f"🎯 *Probabilidad:* {prob_final:.2%}\n"
                   f"📊 *Confianza Modelo:* {res_p['acc']:.2%}\n"
                   f"🕒 *Vela:* {res_p['timestamp_marca']}\n\n"
                   f"💪 *Fuerza (ADX):* {res_p['adx']:.1f}\n"
                   f"💰 *Precio Actual:* {res_p['price']:.2f}\n\n"
                   f"⏰ *Hora Bogotá:* {hora_actual}")
            
            send_telegram_alert(msg)

if __name__ == "__main__":
    run_prediction_cycle()
