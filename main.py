from optimize import run_optuna
import pandas as pd
from utils import split_data, plot_portfolio_subplots, plot_portfolio_history, evaluate_best_params, calculate_returns_summary, plot_metrics_summary_table, plot_returns_summary_table



def main():
    # 1️⃣ Dividir datos

    train_df, test_df, val_df = split_data(df=pd.read_csv("data/BTC-USD-hourly.csv"))

    # 2️⃣ Estudio Optuna
    study = run_optuna(train_data=train_df, n_trials=55)

    print("\n✅ Mejor trial:")
    print(f"Score (Calmar): {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    # 3️⃣ Evaluar en train, test y val
    full_df = pd.concat([train_df, test_df, val_df])
    portfolios_dict, metrics_dict = evaluate_best_params(train_df, test_df, val_df,full_df, study.best_params)

    
    train_hist = portfolios_dict['train']
    test_hist = portfolios_dict['test']
    val_hist = portfolios_dict['val']
    full_hist = portfolios_dict['full']

    summary_df = calculate_returns_summary(train_hist,test_hist,val_hist, full_hist)
    fig1 = plot_returns_summary_table(summary_df, title="Annualized Returns by Period")
    fig1.show()
    
    
    # 4️⃣ Graficar resultados
    fig2 = plot_portfolio_subplots(portfolios_dict, metrics_dict)
    fig2.show()

    fig3 = plot_portfolio_history(portfolios_dict)
    fig3.show()

    f = ['Monthly', 'Quarterly', 'Yearly']
    for i in f:
        fig4 = plot_metrics_summary_table(metrics_dict, freq=i)
        fig4.show()

if __name__ == "__main__":
    main()


