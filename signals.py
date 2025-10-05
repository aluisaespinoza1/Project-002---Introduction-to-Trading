# signals.py
import pandas as pd
import ta
#from ta.momentum import RSIIndicator
#from ta.trend import MACD, SMAIndicator, EMAIndicator

def get_indicators(data: pd.DataFrame, 
                   rsi_window: int,
                   macd_fast: int,
                   macd_slow: int,
                   macd_signal: int,
                   stoch_window: int,
                   smooth_window: int) -> pd.DataFrame:
    """
    Calcula indicadores técnicos: RSI, MACD, Stochastic.
    """

    # --- RSI ---
    rsi_indicator = ta.momentum.RSIIndicator(
        close=data["Close"], 
        window=rsi_window
    )
    data["rsi"] = rsi_indicator.rsi()
    
    # --- MACD ---
    macd = ta.trend.MACD(
        close=data["Close"],
        window_fast=macd_fast,
        window_slow=macd_slow,
        window_sign=macd_signal
    )
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_diff"] = macd.macd_diff()
    
    # --- Stochastic Oscillator ---
    stoch = ta.momentum.StochasticOscillator(
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        window=stoch_window,
        smooth_window=smooth_window
    )
    data["stoch_k"] = stoch.stoch()
    data["stoch_d"] = stoch.stoch_signal()
    
    return data

def get_signals(data: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Genera señales de compra y venta con base en RSI, MACD y Stochastic.
    La señal final es TRUE si al menos 2 de los 3 indicadores coinciden.
    """
    
    # calcular indicadores
    data = get_indicators(
        data.copy(),
        rsi_window=params["rsi_window"],
        macd_fast=params["macd_fast"],
        macd_slow=params["macd_slow"],
        macd_signal=params["macd_signal"],
        stoch_window=params["stoch_window"],
        smooth_window=params["smooth_window"]
    )
    
    # --- señales individuales ---
    # RSI
    rsi_buy  = data["rsi"] < params["rsi_buy"]
    rsi_sell = data["rsi"] > params["rsi_sell"]
    
    # MACD
    macd_buy  = data["macd"] > data["macd_signal"]
    macd_sell = data["macd"] < data["macd_signal"]
    
    # Stochastic
    stoch_buy  = data["stoch_k"] > data["stoch_d"]
    stoch_sell = data["stoch_k"] < data["stoch_d"]
    
    # --- mayoría de señales (al menos 2 de 3) ---
    buy_votes  = rsi_buy.astype(int) + macd_buy.astype(int) + stoch_buy.astype(int)
    sell_votes = rsi_sell.astype(int) + macd_sell.astype(int) + stoch_sell.astype(int)
    
    data["buy_sig"]  = buy_votes >= 2
    data["sell_sig"] = sell_votes >= 2
    
    # columnas originales
    cols_originales = [
        "Unix", "Date", "Symbol", "Open", "High", "Low", "Close",
        "Volume BTC", "Volume USDT", "tradecount"
    ]
    
    data = data.dropna()
    df = data[cols_originales + ["buy_sig", "sell_sig"]]
    
    return df, data
