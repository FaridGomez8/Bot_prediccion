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
    except Exception as e:
        print(f"Error capturando datos: {e}")
        return None

def add_technical_indicators(df):
    df = df.copy()
    
    # 1. INDICADORES BASE
    df['EMA_55'] = ta.trend.ema_indicator(df['close'], window=55)
    df['DIST_EMA'] = (df['close'] - df['EMA_55']) / df['EMA_55']
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['BBL_PERC'] = bb.bollinger_pband()
    
    # 2. FILTROS DE CALIDAD (CORRECCIÓN MFI)
    adx_obj = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['ADX'] = adx_obj.adx()
    
    # CAMBIO AQUÍ: MFI está en ta.volume, no en ta.momentum
    df['MFI'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
    
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['ATR_NORM'] = df['ATR'] / df['close'] 
    
    return df.dropna()

def send_telegram_alert(message):
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={message}"
    try:
        requests.get(url)
    except:
        pass

def run_prediction_cycle():
    results = {}
    print(f"--- Ejecutando ciclo completo para {SYMBOL} ---")

    for tf in TIME_FRAMES:
        df = get_kucoin_klines(SYMBOL, tf)
        if df is not None and len(df) > 60:
            df_ind = add_technical_indicators(df)
            df_ind['target'] = (df_ind['close'].shift(-1) > df_ind['close']).astype(int)
            df_ind.dropna(inplace=True)

            features = ['RSI', 'DIST_EMA', 'BBL_PERC', 'ADX', 'MFI', 'ATR_NORM']
            X = df_ind[features]
            y = df_ind['target']

            split = int(len(X) * 0.8)
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X.iloc[:split])
            X_test = scaler.transform(X.iloc[split:])
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
            model.fit(X_train, y_train)
            
            acc = accuracy_score(y_test, model.predict(X_test))
            last_row = df_ind.tail(1)
            prob_up = model.predict_proba(scaler.transform(last_row[features]))[0][1]
            
            ts_utc = last_row.index[0].tz_localize('UTC')
            ts_bogota = ts_utc.astimezone(ZONA_HORARIA)

            results[tf] = {
                'prob': prob_up, 
                'acc': acc, 
                'price': df_ind['close'].iloc[-1],
                'adx': last_row['ADX'].iloc[0],
                'mfi': last_row['MFI'].iloc[0],
                'timestamp_marca': ts_bogota.strftime('%H:%M:%S')
            }

    if TF_PRINCIPAL in results:
        res_p = results[TF_PRINCIPAL]
        es_subida = res_p['prob'] >= 0.5
        prob_final = res_p['prob'] if es_subida else (1 - res_p['prob'])
        direccion = "Subir 📈" if es_subida else "Bajar 📉"

        if prob_final >= UMBRAL_PROBABILIDAD and res_p['adx'] > 20:
            confluencia_1h = (results['1hour']['prob'] >= 0.5) == es_subida
            meta_label = "💎 SEÑAL DE ALTA CONFLUENCIA" if confluencia_1h else "⚠️ SEÑAL INDIVIDUAL"

            resumen_otros = ""
            for tf, r in results.items():
                if tf != TF_PRINCIPAL:
                    icon = "⬆️" if r['prob'] >= 0.5 else "⬇️"
                    prob_tf = r['prob'] if r['prob'] >= 0.5 else (1 - r['prob'])
                    resumen_otros += f"- *{tf}:* {icon} ({prob_tf:.2%}) | 🕒 Marca: {r['timestamp_marca']}\n"

            hora_actual = pd.Timestamp.now(tz='UTC').astimezone(ZONA_HORARIA).strftime('%H:%M:%S')

            msg = (f"{meta_label}\n\n"
                   f"🔮 *Predicción ({TF_PRINCIPAL}):* *{direccion}*\n"
                   f"🎯 *Probabilidad:* {prob_final:.2%}\n"
                   f"📊 *Confianza Modelo:* {res_p['acc']:.2%}\n"
                   f"🕒 *Marca Vela {TF_PRINCIPAL}:* {res_p['timestamp_marca']}\n\n"
                   f"💪 *Fuerza (ADX):* {res_p['adx']:.1f}\n"
                   f"💰 *Precio Actual:* {res_p['price']:.2f}\n\n"
                   f"🌐 *Contexto Temporal (Bogotá):*\n{resumen_otros}\n"
                   f"⏰ *Hora de Alerta:* {hora_actual}")
            
            send_telegram_alert(msg)

if __name__ == "__main__":
    run_prediction_cycle()
