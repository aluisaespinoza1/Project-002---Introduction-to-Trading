from backtest import backtest
from signals import get_signals
from metrics import get_calmar
import numpy as np
import optuna
import pandas as pd

def optimize(trial, train_data):
    """
    Función objetivo para Optuna que maximiza el ratio de Calmar con cross-validation para evitar overfitting.

    Parameters:
    trial : optuna.trial.Trial
        Objeto trial de Optuna que sugiere los parámetros a optimizar
    train_data : pandas.DataFrame
        DataFrame con datos de precios históricos para entrenamiento.
        Debe contener al menos las columnas: 'Open', 'High', 'Low', 'Close', 'Volume'

    Returns:
    float
        Promedio del ratio de Calmar obtenido en validación cruzada temporal.
        Retorna -np.inf si no hubo operaciones válidas en ningún chunk.
    """

    data = train_data.copy()
    
    # --- 1. Definir los parámetros que Optuna optimizará ---
    params = {
        # RSI - rangos más estándar
        "rsi_window": trial.suggest_int("rsi_window", 10, 21),  # Más corto, más reactivo
        "rsi_buy": trial.suggest_int("rsi_buy", 25, 35),  # Zona de sobreventa clásica
        "rsi_sell": trial.suggest_int("rsi_sell", 65, 75),  # Zona de sobrecompra clásica
        
        # MACD - rangos estándar de la industria
        "macd_fast": trial.suggest_int("macd_fast", 8, 15),
        "macd_slow": trial.suggest_int("macd_slow", 20, 30),
        "macd_signal": trial.suggest_int("macd_signal", 5, 12),
        
        # Stochastic - ventanas más cortas para crypto
        "stoch_window": trial.suggest_int("stoch_window", 10, 21),
        "smooth_window": trial.suggest_int("smooth_window", 3, 7),
        
        # Backtest - CRÍTICO para minimizar pérdidas
        "stop_loss": trial.suggest_float("stop_loss", 0.02, 0.08),  # 2-8% SL más apretado
        "take_profit": trial.suggest_float("take_profit", 0.03, 0.12),  # 3-12% TP proporcional
        "n_shares": trial.suggest_float("n_shares", 0.05, 1.5),  # Posiciones más pequeñas
    }

    # --- 2. Calcular señales con los parámetros sugeridos ---
    signals_df, _ = get_signals(data, **params)

    # --- 3. Validación cruzada temporal (n_splits chunks) ---
    n_splits = 5
    len_data = len(signals_df)
    size = len_data // n_splits
    calmars = []

    for i in range(n_splits):
        start_idx = i * size
        end_idx = (i + 1) * size if i < n_splits - 1 else len_data
        chunk = signals_df.iloc[start_idx:end_idx, :]

        # correr backtest con este chunk
        port_vals = backtest(chunk, **params)

        # calcular calmar ratio
        calmar = get_calmar(port_vals)
        if not np.isnan(calmar):
            calmars.append(calmar)

    # --- 4. Retornar el promedio de Calmar en validación cruzada ---
    if len(calmars) == 0:
        return -np.inf  # si no hubo operaciones válidas
    return np.mean(calmars)

def run_optuna(train_data: pd.DataFrame, n_trials: int = 50):
    """
    Args:
        train_data (pd.DataFrame): DataFrame con datos de entrenamiento.
        n_trials (int, optional): Defaults to 70.

    Returns:
        optuna.Study: Estudio de Optuna con los mejores parámetros encontrados.
        Maximiza el ratio de Calmar con la función objetivo definida optimize.
    """

    study = optuna.create_study(direction="maximize", study_name="crypto_calmar_opt")
    study.optimize(lambda trial: optimize(trial, train_data), n_trials=n_trials)

    print("\n✅ Mejor trial:")
    print(f"Score (Calmar): {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study