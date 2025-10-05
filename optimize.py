from backtest import backtest
from signals import get_signals
from metrics import get_calmar
import numpy as np
import optuna

def optimize(trial, train_data):
    """
    Función objetivo para Optuna que maximiza el ratio de Calmar.
    """
    data = train_data.copy()
    
    # --- 1. Definir los parámetros que Optuna optimizará ---
    params = {
        # RSI
        "rsi_window": trial.suggest_int("rsi_window",7, 30),
        "rsi_buy": trial.suggest_int("rsi_buy", 20, 40),
        "rsi_sell": trial.suggest_int("rsi_sell", 70, 90),

        # MACD
        "macd_fast": trial.suggest_int("macd_fast", 6, 20),
        "macd_slow": trial.suggest_int("macd_slow", 20, 40),
        "macd_signal": trial.suggest_int("macd_signal", 5, 15),

        # Stochastic
        "stoch_window": trial.suggest_int("stoch_window", 10, 48),
        "smooth_window": trial.suggest_int("smooth_window", 3, 15),

        # Backtest
        "stop_loss": trial.suggest_float("stop_loss", 0.1, 0.5),
        "take_profit": trial.suggest_float("take_profit", 0.1, 0.3),
        "n_shares": trial.suggest_float("n_shares", 0.1, 25),
    }

    # --- 2. Calcular señales con los parámetros sugeridos ---
    signals_df, _ = get_signals(data, **params)

    # --- 3. Validación cruzada temporal (n_splits chunks) ---
    n_splits = 6
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

def run_optuna(train_data: pd.DataFrame, n_trials: int = 70):
    study = optuna.create_study(direction="maximize", study_name="crypto_calmar_opt")
    study.optimize(lambda trial: optimize(trial, train_data), n_trials=n_trials)

    print("\n✅ Mejor trial:")
    print(f"Score (Calmar): {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study