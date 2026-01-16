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
UMBRAL_PROBABILIDAD = 0.10  # Cambiado al 10% para tu prueba de conexión

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
            # IMPORTANTE: Orden viejo -> nuevo para indicadores
            df = df.sort_values('timestamp', ascending=True)
            return df.set_index('timestamp')
        return None
    except Exception as e:
        print(f"Error capturando datos: {e}")
        return None

def add_technical_indicators(df):
    """Calcula indicadores técnicos usando ratios para mejor predicción."""
    df = df.copy()
    # Tendencia: Distancia porcentual a la EMA
    df['EMA_55'] = ta.trend.ema_indicator(df['close'], window=55)
    df['DIST_EMA'] = (df['close'] - df['EMA_55']) / df['EMA_55']
    
    # Momentum: RSI y su cambio
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['RSI_DIFF'] = df['RSI'].diff()
    
    # Volatilidad: Ubicación en Bandas de Bollinger (0 a 1)
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['BBL_PERC'] = bb.bollinger_pband()
    
    # Volumen: Cambio porcentual
    df['VOL_CHG'] = df['volume'].pct_change()
    
    return df.dropna()

def send_telegram_alert(message):
    """Envía la alerta usando las variables de entorno existentes."""
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("⚠️ Error: Faltan variables de entorno TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={message}"
    try:
        requests.get(url)
        print("✅ Alerta enviada a Telegram.")
    except Exception as e:
        print(f"❌ Error al enviar mensaje: {e}")

def run_prediction_cycle():
    results = {}
    print(f"--- Iniciando análisis para {SYMBOL} ---")

    for tf in TIME_FRAMES:
        df = get_kucoin_klines(SYMBOL, tf)
        if df is not None and len(df) > 60:
            df = add_technical_indicators(df)
            
            # El "Target": 1 si el precio sube en la SIGUIENTE vela
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df.dropna(inplace=True)

            # Selección de variables para el modelo
            features = ['RSI', 'RSI_DIFF', 'DIST_EMA', 'BBL_PERC', 'VOL_CHG']
            X = df[features]
            y = df['target']

            # Entrenamiento con validación temporal (no aleatoria)
            split = int(len(X) * 0.8)
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X.iloc[:split])
            X_test = scaler.transform(X.iloc[split:])
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            # Modelo Random Forest (más estable que Regresión Logística)
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluación y Predicción actual
            acc = accuracy_score(y_test, model.predict(X_test))
            last_row_scaled = scaler.transform(X.tail(1))
            prob_up = model.predict_proba(last_row_scaled)[0][1]
            
            results[tf] = {'prob': prob_up, 'acc': acc, 'price': df['close'].iloc[-1]}
            print(f"TF {tf}: Precisión {acc:.2%} | Prob. Subir {prob_up:.2%}")

    # Lógica de Alerta basada en la temporalidad principal (15min)
    if TF_PRINCIPAL in results:
        res_p = results[TF_PRINCIPAL]
        
        # Determinar dirección predominante
        es_subida = res_p['prob'] >= 0.5
        prob_final = res_p['prob'] if es_subida else (1 - res_p['prob'])
        direccion = "Subir 📈" if es_subida else "Bajar 📉"

        # Verificación de umbral
        if prob_final >= UMBRAL_PROBABILIDAD:
            # Construir resumen de confluencia
            resumen_otros = ""
            for tf, r in results.items():
                if tf != TF_PRINCIPAL:
                    icon = "⬆️" if r['prob'] > 0.5 else "⬇️"
                    prob_val = r['prob'] if r['prob'] > 0.5 else (1 - r['prob'])
                    resumen_otros += f"- *{tf}:* {icon} ({prob_val:.0%})\n"

            # Tiempo local
            bogota_tz = pytz.timezone('America/Bogota')
            hora_bog = pd.Timestamp.now(tz='UTC').astimezone(bogota_tz).strftime('%H:%M:%S')

            msg = (f"🧪 *MODO PRUEBA (Umbral {UMBRAL_PROBABILIDAD:.0%})*\n"
                   f"🚨 *ALERTA {SYMBOL} ({TF_PRINCIPAL})*\n\n"
                   f"🔮 *Predicción:* El precio tiende a *{direccion}*\n"
                   f"🎯 *Probabilidad:* {prob_final:.2%}\n"
                   f"📊 *Confianza del Modelo:* {res_p['acc']:.2%}\n"
                   f"💰 *Precio Actual:* {res_p['price']:.2f}\n\n"
                   f"🌐 *Confluencia Temporal:*\n{resumen_otros}\n"
                   f"⏰ *Hora Bogotá:* {hora_bog}")
            
            send_telegram_alert(msg)
        else:
            print(f"No se superó el umbral. Probabilidad máxima: {prob_final:.2%}")

if __name__ == "__main__":
    run_prediction_cycle()
