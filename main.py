from optimize import run_optuna
import pandas as pd
from utils import split_data, plot_portfolio_subplots, plot_portfolio_history, evaluate_best_params, plot_all_metrics_heatmap, plot_metrics_summary_table 



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
    portfolios_dict, metrics_dict = evaluate_best_params(train_df, test_df, val_df, study.best_params)

    # 4️⃣ Graficar resultados
    fig1 = plot_portfolio_subplots(portfolios_dict, metrics_dict)
    fig1.show()

    f = ['Monthly', 'Quarterly', 'Yearly']
    for i in f:
        fig2 = plot_metrics_summary_table(metrics_dict, freq=i)
        fig2.show()

if __name__ == "__main__":
    main()


