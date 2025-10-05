import pandas as pd
import numpy as np


def get_calmar(portfolio_df: pd.DataFrame) -> float:
    """
    Calcula el ratio de Calmar a partir de un dataframe con columnas:
    ['Date', 'portfolio_value'].
    Los datos son horarios (24 datos por día, 8760 por año aprox).
    Penaliza casos con drawdown cero.
    """
    df = portfolio_df.copy().dropna(subset=["portfolio_value"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Calcular retornos horarios
    df["returns"] = df["portfolio_value"].pct_change()
    
    # Calcular CAGR (asumiendo 8760 horas por año)
    total_return = df["portfolio_value"].iloc[-1] / df["portfolio_value"].iloc[0]
    n_hours = (df["Date"].iloc[-1] - df["Date"].iloc[0]).total_seconds() / 3600
    years = n_hours / 8760
    CAGR = total_return ** (1 / years) - 1 if years > 0 else np.nan
    
    # Calcular drawdown
    df["cum_max"] = df["portfolio_value"].cummax()
    df["drawdown"] = df["portfolio_value"] / df["cum_max"] - 1
    max_dd = df["drawdown"].min()
    
    # Calmar ratio
    if max_dd == 0 or np.isnan(max_dd):
        calmar = -9999  # penaliza casos con drawdown cero
    else:
        calmar = CAGR / abs(max_dd)
    
    # Manejo de infinitos
    if not np.isfinite(calmar):
        calmar = -9999 #penaliza casos con calmar inf
    
    return calmar

def get_sharpe_ratio(df, freq='Monthly', risk_free_rate=0):
    """Calcula el Sharpe Ratio anualizado"""

    freq_map = {'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
    annualization_factors = {'Monthly': np.sqrt(12), 'Quarterly': np.sqrt(4), 'Yearly': 1.0}
    
    if freq not in freq_map:
        return np.nan
    
    ret = df.set_index('Date')['portfolio_value'].resample(freq_map[freq]).last().pct_change().dropna()
    
    if len(ret) < 2 or np.std(ret) == 0:
        return np.nan
    
    periods_per_year = {'Monthly': 12, 'Quarterly': 4, 'Yearly': 1}
    rf_adjusted = risk_free_rate / periods_per_year[freq]
    
    excess_return = ret - rf_adjusted
    sharpe = np.mean(excess_return) / np.std(ret) * annualization_factors[freq]
    
    return sharpe


def get_sortino_ratio(df, freq='Monthly', risk_free_rate=0, target_return=0):
    """Calcula el Sortino Ratio anualizado"""

    freq_map = {'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
    annualization_factors = {'Monthly': np.sqrt(12), 'Quarterly': np.sqrt(4), 'Yearly': 1.0}
    
    if freq not in freq_map:
        return np.nan
    
    ret = df.set_index('Date')['portfolio_value'].resample(freq_map[freq]).last().pct_change().dropna()
    
    if len(ret) < 2:
        return np.nan
    
    downside_returns = ret[ret < target_return] - target_return
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_std = np.sqrt(np.mean(downside_returns**2))
    
    if downside_std == 0:
        return np.nan
    
    periods_per_year = {'Monthly': 12, 'Quarterly': 4, 'Yearly': 1}
    rf_adjusted = risk_free_rate / periods_per_year[freq]
    
    excess_return = np.mean(ret) - rf_adjusted
    sortino = excess_return / downside_std * annualization_factors[freq]
    
    return sortino


def get_max_drawdown(df):
    """Calcula el máximo drawdown"""

    values = df['portfolio_value'].values
    cummax = np.maximum.accumulate(values)
    drawdown = (values - cummax) / cummax
    return drawdown.min()


def get_win_rate(df, freq='Monthly'):
    """Calcula el win rate (% de períodos positivos)"""

    freq_map = {'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
    
    if freq not in freq_map:
        return np.nan
    
    ret = df.set_index('Date')['portfolio_value'].resample(freq_map[freq]).last().pct_change().dropna()
    
    if len(ret) == 0:
        return np.nan
    
    return (ret > 0).mean()


def get_calmar_ratio(df):
    """Calcula el Calmar Ratio (retorno anualizado / máximo drawdown absoluto)"""

    if len(df) < 2:
        return np.nan
    
    tiempo_total = df['Date'].iloc[-1] - df['Date'].iloc[0]
    años = tiempo_total.total_seconds() / (365.25 * 24 * 3600)
    
    if años <= 0:
        return np.nan
    
    valor_inicial = df['portfolio_value'].iloc[0]
    valor_final = df['portfolio_value'].iloc[-1]
    
    if valor_inicial <= 0:
        return np.nan
    
    retorno_total = (valor_final / valor_inicial) - 1
    retorno_anualizado = (1 + retorno_total) ** (1 / años) - 1
    
    max_dd = abs(get_max_drawdown(df))
    
    if max_dd == 0:
        return np.inf
    
    return retorno_anualizado / max_dd