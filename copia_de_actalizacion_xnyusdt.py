import requests
import pandas as pd
import numpy as np
import ta
import pytz
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score

# --- CONFIGURACIÓN ---
# El código buscará estas variables en tus "Secrets" de GitHub o entorno local
SYMBOL = 'BTC-USDT'
TIME_FRAMES = ['5min', '15min', '1hour', '4hour']
TF_PRINCIPAL = '15min'

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
            # CORRECCIÓN: Orden ascendente para indicadores técnicos
            df = df.sort_values('timestamp', ascending=True)
            return df.set_index('timestamp')
        return None
    except Exception as e:
        print(f"Error capturando datos: {e}")
        return None

def add_technical_indicators(df):
    df = df.copy()
    # Tendencia y Ratios
    df['EMA_55'] = ta.trend.ema_indicator(df['close'], window=55)
    df['DIST_EMA'] = (df['close'] - df['EMA_55']) / df['EMA_55']
    # Momentum
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['RSI_DIFF'] = df['RSI'].diff()
    # Volatilidad
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['BBL_PERC'] = bb.bollinger_pband()
    # Volumen
    df['VOL_CHG'] = df['volume'].pct_change()
    return df.dropna()

def send_telegram_alert(message):
    token = os.environ.get('TELEGRAM_BOT_TOKEN') # Mantiene tus nombres de variables
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print("⚠️ Error: No se encontraron las llaves de Telegram en el entorno.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={message}"
    requests.get(url)

def run_prediction_cycle():
    results = {}
    print(f"--- Iniciando análisis para {SYMBOL} ---")

    for tf in TIME_FRAMES:
        df = get_kucoin_klines(SYMBOL, tf)
        if df is not None and len(df) > 60:
            df = add_technical_indicators(df)
            # Target: ¿Sube en la siguiente vela?
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df.dropna(inplace=True)

            features = ['RSI', 'RSI_DIFF', 'DIST_EMA', 'BBL_PERC', 'VOL_CHG']
            X = df[features]
            y = df['target']

            # Entrenamiento temporal
            split = int(len(X) * 0.8)
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X.iloc[:split])
            X_test = scaler.transform(X.iloc[split:])
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            acc = accuracy_score(y_test, model.predict(X_test))
            prob_up = model.predict_proba(scaler.transform(X.tail(1)))[0][1]
            
            results[tf] = {'prob': prob_up, 'acc': acc, 'price': df['close'].iloc[-1]}

    # Lógica de Alerta
    if TF_PRINCIPAL in results:
        res_p = results[TF_PRINCIPAL]
        prob_final = res_p['prob'] if res_p['prob'] >= 0.5 else (1 - res_p['prob'])
        direccion = "Subir 📈" if res_p['prob'] >= 0.5 else "Bajar 📉"

        # CONDICIÓN: Probabilidad > 70% y Precisión aceptable
        if prob_final >= 0.70:
            resumen_otros = ""
            for tf, r in results.items():
                if tf != TF_PRINCIPAL:
                    icon = "⬆️" if r['prob'] > 0.5 else "⬇️"
                    resumen_otros += f"- {tf}: {icon} ({max(r['prob'], 1-r['prob']):.0%})\n"

            bogota_tz = pytz.timezone('America/Bogota')
            hora = pd.Timestamp.now(tz='UTC').astimezone(bogota_tz).strftime('%H:%M:%S')

            msg = (f"🚨 *ALERTA {SYMBOL} ({TF_PRINCIPAL})*\n\n"
                   f"🔮 *Predicción:* *{direccion}*\n"
                   f"🎯 *Probabilidad:* {prob_final:.2%}\n"
                   f"📊 *Confianza Modelo:* {res_p['acc']:.2%}\n"
                   f"💰 *Precio:* {res_p['price']:.2f}\n\n"
                   f"🌐 *Otras Temporalidades:*\n{resumen_otros}\n"
                   f"⏰ *Hora Bogotá:* {hora}")
            
            send_telegram_alert(msg)
            print("✅ Alerta enviada.")
        else:
            print(f"Probabilidad de {prob_final:.2%} no es suficiente.")

if __name__ == "__main__":
    run_prediction_cycle()
