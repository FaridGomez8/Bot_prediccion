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
# Añadimos '1day' a las temporalidades para el contexto completo
TIME_FRAMES = ['5min', '15min', '1hour', '4hour', '1day']
TF_PRINCIPAL = '15min'
UMBRAL_PROBABILIDAD = 0.70 
MIN_CONFIANZA_MODELO = 0.60 
ZONA_HORARIA = pytz.timezone('America/Bogota')

def get_kucoin_klines(symbol, timeframe):
    endpoint = 'https://api.kucoin.com/api/v1/market/candles'
    # KuCoin usa '1day' para diario
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
    except: return None

def add_technical_indicators(df):
    df = df.copy()
    # Tendencia Inmediata (Filtro de Oro)
    df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['EMA_55'] = ta.trend.ema_indicator(df['close'], window=55)
    
    # Osciladores y Fuerza
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    df['WILLR'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
    
    # Volumen y Volatilidad
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['ATR_NORM'] = ta.volatility.average_true_range(df['high'], df['low'], df['close']) / df['close']
    
    # Distancia a la EMA (para detectar sobre-extensión)
    df['DIST_EMA'] = (df['close'] - df['EMA_55']) / df['EMA_55']
    
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
    features = ['RSI', 'DIST_EMA', 'ADX', 'WILLR', 'ATR_NORM']

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

            model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            acc = accuracy_score(y_test, model.predict(X_test))
            last_row = df_ind.tail(1)
            prob_up = model.predict_proba(scaler.transform(last_row[features]))[0][1]
            
            ts_utc = last_row.index[0].tz_localize('UTC')
            results[tf] = {
                'prob': prob_up, 'acc': acc, 'price': df_ind['close'].iloc[-1],
                'adx': last_row['ADX'].iloc[0], 
                'ema_20': last_row['EMA_20'].iloc[0],
                'timestamp_marca': ts_utc.astimezone(ZONA_HORARIA).strftime('%H:%M')
            }

    if TF_PRINCIPAL in results:
        res_p = results[TF_PRINCIPAL]
        es_subida = res_p['prob'] >= 0.5
        prob_final = res_p['prob'] if es_subida else (1 - res_p['prob'])
        direccion = "Subir 📈" if es_subida else "Bajar 📉"

        # --- LÓGICA DE FILTRADO PROFESIONAL ---
        # 1. Confluencia con 1 Hora
        mismo_sentido = (results['1hour']['prob'] >= 0.5) == es_subida
        
        # 2. Filtro de Tendencia EMA 20 (No ir contra el precio actual)
        filtro_tendencia = False
        if es_subida and res_p['price'] > res_p['ema_20']:
            filtro_tendencia = True
        elif not es_subida and res_p['price'] < res_p['ema_20']:
            filtro_tendencia = True

        # Ejecutar envío solo si cumple TODOS los filtros
        if prob_final >= UMBRAL_PROBABILIDAD and res_p['acc'] >= MIN_CONFIANZA_MODELO and mismo_sentido and filtro_tendencia:
            
            # Construir el contexto de las demás temporalidades
            contexto_msg = ""
            for tf in ['5min', '1hour', '4hour', '1day']:
                if tf in results:
                    r = results[tf]
                    dir_tf = "⬆️" if r['prob'] >= 0.5 else "⬇️"
                    p_tf = r['prob'] if r['prob'] >= 0.5 else (1 - r['prob'])
                    contexto_msg += f"- *{tf}:* {dir_tf} ({p_tf:.1%}) | ${r['price']:.0f}\n"

            hora_actual = pd.Timestamp.now(tz='UTC').astimezone(ZONA_HORARIA).strftime('%H:%M:%S')

            msg = (f"💎 *ALTA CONFLUENCIA DETECTADA*\n\n"
                   f"🎯 *Predicción {TF_PRINCIPAL}:* {direccion}\n"
                   f"🔥 *Probabilidad:* {prob_final:.1%}\n"
                   f"📊 *Confianza:* {res_p['acc']:.1%}\n"
                   f"💪 *ADX:* {res_p['adx']:.1f}\n\n"
                   f"🌐 *Contexto de Mercado:*\n{contexto_msg}\n"
                   f"⏰ *Hora Bogotá:* {hora_actual}")
            
            send_telegram_alert(msg)

if __name__ == "__main__":
    run_prediction_cycle()
