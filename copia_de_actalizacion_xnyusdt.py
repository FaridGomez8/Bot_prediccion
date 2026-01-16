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
SYMBOL = 'BTC-USDT'
TIME_FRAMES = ['5min', '15min', '1hour', '4hour']
TF_PRINCIPAL = '15min'
UMBRAL_PROBABILIDAD = 0.70  # Umbral de éxito objetivo al 70%

def get_kucoin_klines(symbol, timeframe):
    """Obtiene datos de Kucoin y los organiza cronológicamente."""
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
            # Orden cronológico para cálculo de indicadores
            df = df.sort_values('timestamp', ascending=True)
            return df.set_index('timestamp')
        return None
    except Exception as e:
        print(f"Error capturando datos: {e}")
        return None

def add_technical_indicators(df):
    """Calcula indicadores técnicos usando ratios para mejor predicción."""
    df = df.copy()
    # Tendencia: Distancia a la EMA 55
    df['EMA_55'] = ta.trend.ema_indicator(df['close'], window=55)
    df['DIST_EMA'] = (df['close'] - df['EMA_55']) / df['EMA_55']
    
    # Momentum: RSI y su velocidad de cambio
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['RSI_DIFF'] = df['RSI'].diff()
    
    # Volatilidad: Ubicación porcentual en Bandas de Bollinger
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['BBL_PERC'] = bb.bollinger_pband()
    
    # Volumen: Cambio porcentual de flujo
    df['VOL_CHG'] = df['volume'].pct_change()
    
    return df.dropna()

def send_telegram_alert(message):
    """Envía la alerta usando las variables de entorno de GitHub Secrets."""
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("⚠️ Error: Faltan variables TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={message}"
    try:
        requests.get(url)
        print("✅ Alerta enviada con éxito.")
    except Exception as e:
        print(f"❌ Error al enviar mensaje: {e}")

def run_prediction_cycle():
    results = {}
    print(f"--- Iniciando análisis para {SYMBOL} ---")

    for tf in TIME_FRAMES:
        df = get_kucoin_klines(SYMBOL, tf)
        if df is not None and len(df) > 60:
            df = add_technical_indicators(df)
            
            # Target: ¿Cierre siguiente > Cierre actual?
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df.dropna(inplace=True)

            features = ['RSI', 'RSI_DIFF', 'DIST_EMA', 'BBL_PERC', 'VOL_CHG']
            X = df[features]
            y = df['target']

            # Validación temporal
            split = int(len(X) * 0.8)
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X.iloc[:split])
            X_test = scaler.transform(X.iloc[split:])
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            # Modelo Random Forest
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            # Resultado de precisión y probabilidad actual
            acc = accuracy_score(y_test, model.predict(X_test))
            prob_up = model.predict_proba(scaler.transform(X.tail(1)))[0][1]
            
            results[tf] = {'prob': prob_up, 'acc': acc, 'price': df['close'].iloc[-1]}

    # Lógica de Alerta Principal (15min)
    if TF_PRINCIPAL in results:
        res_p = results[TF_PRINCIPAL]
        
        # Determinar dirección predominante
        es_subida = res_p['prob'] >= 0.5
        prob_final = res_p['prob'] if es_subida else (1 - res_p['prob'])
        direccion = "Subir 📈" if es_subida else "Bajar 📉"

        # Solo envía si supera el 70%
        if prob_final >= UMBRAL_PROBABILIDAD:
            resumen_otros = ""
            for tf, r in results.items():
                if tf != TF_PRINCIPAL:
                    dir_tf = "Sube" if r['prob'] >= 0.5 else "Baja"
                    icon_tf = "⬆️" if r['prob'] >= 0.5 else "⬇️"
                    # Probabilidad de esa dirección específica
                    prob_val = r['prob'] if r['prob'] >= 0.5 else (1 - r['prob'])
                    resumen_otros += f"- *{tf}:* {icon_tf} {dir_tf} ({prob_val:.2%})\n"

            # Formatear Hora Bogotá
            bogota_tz = pytz.timezone('America/Bogota')
            hora_bog = pd.Timestamp.now(tz='UTC').astimezone(bogota_tz).strftime('%H:%M:%S')

            msg = (f"🚨 *ALERTA DE PROBABILIDAD ALTA ({SYMBOL})*\n\n"
                   f"🔮 *Predicción Principal ({TF_PRINCIPAL}):*\n"
                   f"El precio tiende a *{direccion}*\n"
                   f"🎯 *Probabilidad:* {prob_final:.2%}\n"
                   f"📊 *Confianza del Modelo:* {res_p['acc']:.2%}\n"
                   f"💰 *Precio Actual:* {res_p['price']:.2f}\n\n"
                   f"🌐 *Resumen otras temporalidades:*\n{resumen_otros}\n"
                   f"⏰ *Hora Bogotá:* {hora_bog}")
            
            send_telegram_alert(msg)
        else:
            print(f"Probabilidad de {prob_final:.2%} por debajo del umbral del 70%.")

if __name__ == "__main__":
    run_prediction_cycle()
