from dataclasses import dataclass
import pandas as pd


@dataclass
class Position:
    time: str
    price: float
    stop_loss: float
    take_profit: float
    n_shares: float
    type: str

def backtest(data, **params):
    """
    Backtest de portafolio tomando solo los parámetros financieros:
    stop_loss, take_profit y n_shares del diccionario params.
    Devuelve un DataFrame con fecha y valor del portafolio, cash final y posiciones activas.
    """
    data = data.copy()
    
    SL = params["stop_loss"]
    TP = params["take_profit"]
    n_shares = params["n_shares"]
    
    active_long = []
    active_short = []
    cash = 1000000
    com = 0.00125
    port_hist = []
    
    for i, row in data.iterrows():
        # --- OPEN POSITIONS ---
        if row.sell_sig:
            cost = row.Close * n_shares * (1 + com)
            if cash > cost:
                cash -= cost
                active_short.append(
                    Position(
                        time=row.Date,
                        price=row.Close,
                        stop_loss=row.Close * (1 + SL),
                        take_profit=row.Close * (1 - TP),
                        n_shares=n_shares,
                        type='short'
                    )
                )

        if row.buy_sig:
            cost = row.Close * n_shares * (1 + com)
            if cash > cost:
                cash -= cost
                active_long.append(
                    Position(
                        time=row.Date,
                        price=row.Close,
                        stop_loss=row.Close * (1 - SL),
                        take_profit=row.Close * (1 + TP),
                        n_shares=n_shares,
                        type='long'
                    )
                )
        
        # --- CLOSE POSITIONS ---
        for pos in active_long.copy():
            if (pos.stop_loss > row.Close) or (pos.take_profit < row.Close):
                cash += row.Close * pos.n_shares * (1 - com)
                active_long.remove(pos)
        
        for pos in active_short.copy():
            # Stop-loss: precio actual supera precio de stop (perdemos)
            # Take-profit: precio actual está bajo precio target (ganamos)
            if (row.Close >= pos.stop_loss) or (row.Close <= pos.take_profit):
                # Profit/Loss = (Precio de entrada - Precio de salida) * n_shares
                pnl = (pos.price - row.Close) * pos.n_shares
                # Devolvemos el colateral inicial más/menos el P&L, menos comisión
                cash += (pos.price * pos.n_shares) + pnl * (1 - com)
                active_short.remove(pos)
        
        # --- PORTFOLIO VALUE ---
        port_v = cash
        for pos in active_long:
            port_v += row.Close * pos.n_shares
        for pos in active_short:
            port_v += (pos.price * n_shares) + (pos.price - row.Close) * n_shares
        
        # guardar fecha y valor del portafolio
        port_hist.append({"Date": row.Date, "portfolio_value": port_v})
    
    # convertir a DataFrame
    port_df = pd.DataFrame(port_hist)
    
    return port_df