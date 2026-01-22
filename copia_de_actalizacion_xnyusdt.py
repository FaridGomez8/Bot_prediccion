import requests
import pandas as pd
import numpy as np
import ta
import pytz
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

# --- CONFIGURACIÓN ---
SYMBOL = 'BTC-USDT'
TIME_FRAMES = ['5min', '15min', '1hour', '4hour', '1day']
TF_PRINCIPAL = '15min'
UMBRAL_PROBABILIDAD = 0.65 
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
    except: return None

def add_technical_indicators(df):
    df = df.copy()
    # Medias Móviles
    df['EMA_9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['EMA_55'] = ta.trend.ema_indicator(df['close'], window=55)
    
    # Fuerza y Tendencia
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    
    # Volatilidad (ATR) - Clave para movimientos grandes
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['ATR_NORM'] = df['ATR'] / df['close'] 
    
    # Distancia a la EMA 55
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
    # Añadimos ATR_NORM para que la IA aprenda a detectar grandes volatilidades
    features = ['RSI', 'DIST_EMA', 'ADX', 'ATR_NORM']

    for tf in TIME_FRAMES:
        df = get_kucoin_klines(SYMBOL, tf)
        if df is not None and len(df) > 60:
            df_ind = add_technical_indicators(df)
            # Para capturar movimientos GRANDES, la IA ahora mira 2 velas adelante
            df_ind['target'] = (df_ind['close'].shift(-2) > df_ind['close']).astype(int)
            df_ind.dropna(inplace=True)

            X = df_ind[features]
            y = df_ind['target']

            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
            model.fit(X_scaled, y)
            
            last_row = df_ind.tail(1)
            prob_up = model.predict_proba(scaler.transform(last_row[features]))[0][1]
            
            ts_utc = last_row.index[0].tz_localize('UTC')
            results[tf] = {
                'prob': prob_up, 
                'price': df_ind['close'].iloc[-1],
                'ema_9': last_row['EMA_9'].iloc[0],
                'rsi': last_row['RSI'].iloc[0],
                'adx': last_row['ADX'].iloc[0],
                'timestamp_marca': ts_utc.astimezone(ZONA_HORARIA).strftime('%H:%M')
            }

    if TF_PRINCIPAL in results:
        res_p = results[TF_PRINCIPAL]
        es_subida = res_p['prob'] >= 0.5
        prob_final = res_p['prob'] if es_subida else (1 - res_p['prob'])
        direccion = "Subir 📈" if es_subida else "Bajar 📉"

        # --- LÓGICA DE FILTRADO PARA GRANDES MOVIMIENTOS ---
        
        # 1. Confluencia con 1H y 4H (Obligatoria para filtrar ruiditos)
        conf_1h = (results['1hour']['prob'] >= 0.5) == es_subida
        conf_4h = (results['4hour']['prob'] >= 0.5) == es_subida
        
        # 2. Filtro de Tendencia EMA 9
        filtro_ema = (es_subida and res_p['price'] > res_p['ema_9']) or \
                     (not es_subida and res_p['price'] < res_p['ema_9'])

        # 3. FRENOS DE AGOTAMIENTO (Evita entrar al final del movimiento)
        filtro_agotamiento = True
        if not es_subida and res_p['rsi'] < 38: # No vender si ya está muy abajo
            filtro_agotamiento = False
        elif es_subida and res_p['rsi'] > 62: # No comprar si ya está muy arriba
            filtro_agotamiento = False

        # Solo enviamos si pasa los 3 filtros y hay confluencia extendida
        if prob_final >= UMBRAL_PROBABILIDAD and conf_1h and conf_4h and filtro_ema and filtro_agotamiento:
            
            contexto_msg = ""
            for tf in ['5min', '1hour', '4hour', '1day']:
                if tf in results:
                    r = results[tf]
                    dir_icon = "⬆️" if r['prob'] >= 0.5 else "⬇️"
                    p_val = r['prob'] if r['prob'] >= 0.5 else (1 - r['prob'])
                    contexto_msg += f"- *{tf}:* {dir_icon} ({p_val:.1%}) | ${r['price']:.0f}\n"

            hora_actual = pd.Timestamp.now(tz='UTC').astimezone(ZONA_HORARIA).strftime('%H:%M:%S')

            msg = (f"💎 *ALERTA DE TENDENCIA FUERTE*\n\n"
                   f"🎯 *Predicción {TF_PRINCIPAL}:* {direccion}\n"
                   f"🔥 *Probabilidad:* {prob_final:.1%}\n"
                   f"📉 *RSI Actual:* {res_p['rsi']:.1f}\n"
                   f"💪 *Fuerza ADX:* {res_p['adx']:.1f}\n\n"
                   f"🌐 *Contexto Multitemporal:*\n{contexto_msg}\n"
                   f"⏰ *Hora Bogotá:* {hora_actual}")
            
            send_telegram_alert(msg)

if __name__ == "__main__":
    run_prediction_cycle()
