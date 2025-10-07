
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from signals import get_signals
from backtest import backtest
import dateparser as dt
from metrics import get_sharpe_ratio, get_sortino_ratio, get_max_drawdown, get_win_rate, get_calmar



def split_data(df):
    """ 
    Divide el DataFrame en train (60%), test (20%) y validation (20%) manteniendo el orden temporal.
    Devuelve tres DataFrames: train, test, val.
    Ordena por fecha si no está ordenado y maneja errores de conversión de fechas.
    """

    n = len(df)

    train_size = int(n * 0.6)
    test_size = int(n * 0.2)
    val_size = n - train_size - test_size  # para cubrir el resto
    
    #df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    #df = df.sort_values("Date").reset_index(drop=True)

    # Parseo de fechas con dateparser para manejar errores y sort por fecha
    df['Date'] = df['Date'].apply(lambda x: dt.parse(str(x))) 
    df = df.sort_values(by='Date', ascending=True, ignore_index=True)

    train = df.iloc[:train_size]
    test  = df.iloc[train_size:train_size + test_size]
    val   = df.iloc[train_size + test_size:]

    print("Train:", train.shape)
    print("Test:", test.shape)
    print("Validation:", val.shape)
    
    return train, test, val

def evaluate_best_params(train_df, test_df, val_df, full_df, best_params):
    """
    Usa best_params para correr el backtest en train, test, val y full,
    y calcula métricas (monthly, quarterly, yearly).
    
    Parámetros:
        train_df, test_df, val_df, full_df: DataFrames con datos
        best_params: dict con parámetros óptimos
    
    Retorna:
        portfolios_dict: { 'train': df, 'test': df, 'val': df, 'full': df }
        metrics_dict: { 'train': df_métricas, 'test': df_métricas,
                        'val': df_métricas, 'full': df_métricas }
    """
    
    # 🔹 Added full dataset here
    datasets = {
        'train': train_df,
        'test': test_df,
        'val': val_df,
        'full': full_df
    }
    
    portfolios_dict = {}
    metrics_dict = {}
    
    freqs = {'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
    
    for name, df in datasets.items():
        # Generar señales
        df_signals, _ = get_signals(df.copy(), **best_params)
        
        # Correr backtest
        port_df = backtest(df_signals, **best_params)
        
        portfolios_dict[name] = port_df
        
        # Calcular métricas
        metrics = {
            'Calmar Ratio': [],
            'Sharpe Ratio': [],
            'Sortino Ratio': [],
            'Max Drawdown': [],
            'Win Rate': []
        }
        
        # Calmar es único (no depende de freq)
        calmar = get_calmar(port_df)
        
        # Max Drawdown es único (no depende de freq)
        max_dd = get_max_drawdown(port_df)
        
        for label in freqs.keys():
            metrics['Calmar Ratio'].append(calmar)
            metrics['Sharpe Ratio'].append(get_sharpe_ratio(port_df, freq=label))
            metrics['Sortino Ratio'].append(get_sortino_ratio(port_df, freq=label))
            metrics['Max Drawdown'].append(max_dd)
            metrics['Win Rate'].append(get_win_rate(port_df, freq=label))
        
        # Crear DataFrame de métricas
        metrics_df = pd.DataFrame(metrics, index=freqs.keys()).T
        metrics_dict[name] = metrics_df
    
    return portfolios_dict, metrics_dict


# ----- PLOTS -----

def plot_portfolio_history(portfolios_dict, title="Portfolio Performance Comparison"):
    """
    Plotea los históricos de portafolios para train, test y val.
    
    Parámetros:
        portfolios_dict: dict con formato {'train': df, 'test': df, 'val': df}
        title: título del gráfico
    
    Retorna:
        fig: objeto de figura de Plotly
    """
    
    fig = go.Figure()
    
    # Colores para cada dataset
    colors = {
        'train': '#1f77b4',  # Azul
        'test': '#ff7f0e',   # Naranja
        'val': '#2ca02c'     # Verde
    }
    
    # Agregar cada portafolio
    for name, df in portfolios_dict.items():
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['portfolio_value'],
            mode='lines',
            name=name.capitalize(),
            line=dict(color=colors.get(name, '#333333'), width=2),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Date: %{x}<br>' +
                         'Portfolio Value: $%{y:,.2f}<br>' +
                         '<extra></extra>'
        ))
    
    # Actualizar layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#333333')
        ),
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridcolor='#e0e0e0',
            showline=True,
            linecolor='#333333'
        ),
        yaxis=dict(
            title='Portfolio Value ($)',
            showgrid=True,
            gridcolor='#e0e0e0',
            showline=True,
            linecolor='#333333',
            tickformat='$,.0f'
        ),
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=12)
        ),
        height=600,
        margin=dict(l=80, r=40, t=100, b=80)
    )
    
    return fig


def plot_portfolio_subplots(portfolios_dict, metrics_dict=None):
    """
    Plotea los portafolios en subplots separados con métricas opcionales.
    
    Parámetros:
        portfolios_dict: dict con formato {'train': df, 'test': df, 'val': df}
        metrics_dict: dict opcional con métricas {'train': df_metrics, ...}
    
    Retorna:
        fig: objeto de figura de Plotly
    """
    
    datasets = list(portfolios_dict.keys())
    n_plots = len(datasets)
    
    # Crear subplots
    fig = make_subplots(
        rows=n_plots, 
        cols=1,
        subplot_titles=[f'{name.capitalize()} Portfolio' for name in datasets],
        vertical_spacing=0.1,
        row_heights=[1/n_plots] * n_plots
    )
    
    colors = {
        'train': '#1f77b4',
        'test': '#ff7f0e',
        'val': '#2ca02c'
    }
    
    for idx, name in enumerate(datasets, start=1):
        df = portfolios_dict[name]
        
        # Agregar línea de portfolio
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['portfolio_value'],
                mode='lines',
                name=name.capitalize(),
                line=dict(color=colors.get(name, '#333333'), width=2),
                showlegend=(idx == 1),
                hovertemplate='Date: %{x}<br>' +
                             'Value: $%{y:,.2f}<br>' +
                             '<extra></extra>'
            ),
            row=idx, 
            col=1
        )
        
        # Agregar línea horizontal del valor inicial
        initial_value = df['portfolio_value'].iloc[0]
        fig.add_hline(
            y=initial_value,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            row=idx,
            col=1
        )
        
        # Agregar anotación con métricas si están disponibles
        if metrics_dict and name in metrics_dict:
            metrics = metrics_dict[name]
            # Usar métricas Monthly (primera columna)
            sharpe = metrics.loc['Sharpe Ratio', 'Monthly']
            calmar = metrics.loc['Calmar Ratio', 'Monthly']
            max_dd = metrics.loc['Max Drawdown', 'Monthly']
            
            annotation_text = (
                f"Sharpe: {sharpe:.2f} | "
                f"Calmar: {calmar:.2f} | "
                f"Max DD: {max_dd:.2%}"
            )
            
            fig.add_annotation(
                text=annotation_text,
                xref=f'x{idx}',
                yref=f'y{idx}',
                x=df['Date'].iloc[-1],
                y=df['portfolio_value'].max(),
                showarrow=False,
                font=dict(size=10, color='#666666'),
                align='right',
                xanchor='right',
                yanchor='top',
                row=idx,
                col=1
            )
    
    # Actualizar ejes
    for idx in range(1, n_plots + 1):
        fig.update_xaxes(
            title_text='Date' if idx == n_plots else '',
            showgrid=True,
            gridcolor='#e0e0e0',
            row=idx,
            col=1
        )
        fig.update_yaxes(
            title_text='Portfolio Value ($)',
            showgrid=True,
            gridcolor='#e0e0e0',
            tickformat='$,.0f',
            row=idx,
            col=1
        )
    
    # Actualizar layout
    fig.update_layout(
        title=dict(
            text='Portfolio Performance by Dataset',
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#333333')
        ),
        hovermode='x',
        template='plotly_white',
        height=300 * n_plots,
        showlegend=False,
        margin=dict(l=80, r=40, t=100, b=80)
    )
    
    return fig

def plot_all_metrics_heatmap(metrics_dict, dataset='train'):
    """
    Crea un heatmap de todas las métricas para un dataset específico.
    
    Parámetros:
        metrics_dict: dict con métricas
        dataset: 'train', 'test', o 'val'
    
    Retorna:
        fig: objeto de figura de Plotly
    """
    
    df_metrics = metrics_dict[dataset]
    
    fig = go.Figure(data=go.Heatmap(
        z=df_metrics.values,
        x=df_metrics.columns,
        y=df_metrics.index,
        colorscale='RdYlGn',
        text=df_metrics.values,
        texttemplate='%{text:.3f}',
        textfont={"size": 12},
        hovertemplate='Metric: %{y}<br>' +
                     'Frequency: %{x}<br>' +
                     'Value: %{z:.3f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'Metrics Heatmap - {dataset.capitalize()} Dataset',
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#333333')
        ),
        xaxis=dict(
            title='Frequency',
            side='top'
        ),
        yaxis=dict(
            title='Metric'
        ),
        template='plotly_white',
        height=500,
        margin=dict(l=150, r=40, t=100, b=80)
    )
    
    return fig


def plot_metrics_summary_table(metrics_dict, frequency='Monthly'):
    """
    Crea una tabla visual comparativa de todas las métricas.
    
    Parámetros:
        metrics_dict: dict con métricas
        frequency: frecuencia a mostrar ('Monthly', 'Quarterly', 'Yearly')
    
    Retorna:
        fig: objeto de figura de Plotly
    """
    
    datasets = list(metrics_dict.keys())
    metrics = metrics_dict[datasets[0]].index.tolist()
    
    # Preparar datos para la tabla
    table_data = []
    for metric in metrics:
        row = [metric]
        for dataset in datasets:
            value = metrics_dict[dataset].loc[metric, frequency]
            # Formatear según el tipo de métrica
            if metric == 'Max Drawdown':
                row.append(f'{value:.2%}')
            elif metric == 'Win Rate':
                row.append(f'{value:.2%}')
            else:
                row.append(f'{value:.3f}')
        table_data.append(row)
    
    # Header de la tabla
    header = ['Metric'] + [d.capitalize() for d in datasets]
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header,
            fill_color='#1f77b4',
            font=dict(color='white', size=14, family='Arial Black'),
            align='left',
            height=40
        ),
        cells=dict(
            values=list(zip(*table_data)),  # Transponer
            fill_color=[['#f0f0f0', 'white'] * len(metrics)],
            font=dict(size=12),
            align='left',
            height=35
        )
    )])
    
    fig.update_layout(
        title=dict(
            text=f'Metrics Summary Table - {frequency}',
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#333333')
        ),
        height=400,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig

def calculate_returns_summary(train_df, test_df, val_df, full_df):
    """
    Calcula rendimientos ANUALIZADOS promedio para diferentes periodos de resampling.
    
    Parameters
    ----------
    train_df, test_df, val_df, full_df : pandas.DataFrame
        Dataframes que deben contener las columnas 'Date' y 'portfolio_value'
    
    Returns
    -------
    pandas.DataFrame
        DataFrame con rendimientos anualizados para cada frecuencia de medición
    """
    
    def calculate_returns(df):
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df_indexed = df.set_index('Date')

        # Duración total del periodo en años
        tiempo_total = df['Date'].iloc[-1] - df['Date'].iloc[0]
        años_totales = tiempo_total.total_seconds() / (365.25 * 24 * 3600)

        valor_inicial = df['portfolio_value'].iloc[0]
        valor_final = df['portfolio_value'].iloc[-1]

        # CAGR general (rendimiento anualizado total)
        if años_totales > 0 and valor_inicial > 0:
            cagr = (valor_final / valor_inicial) ** (1 / años_totales) - 1
        else:
            cagr = 0

        # Rendimiento mensual compuesto
        monthly_prices = df_indexed['portfolio_value'].resample('M').last()
        monthly_returns = monthly_prices.pct_change().dropna()
        if len(monthly_returns) >= 2:
            n_months = len(monthly_returns)
            monthly_comp = (1 + monthly_returns).prod() ** (12 / n_months) - 1
        else:
            monthly_comp = cagr

        # Rendimiento trimestral compuesto
        quarterly_prices = df_indexed['portfolio_value'].resample('Q').last()
        quarterly_returns = quarterly_prices.pct_change().dropna()
        if len(quarterly_returns) >= 2:
            n_quarters = len(quarterly_returns)
            quarterly_comp = (1 + quarterly_returns).prod() ** (4 / n_quarters) - 1
        else:
            quarterly_comp = cagr

        yearly_comp = cagr

        return monthly_comp * 100, quarterly_comp * 100, yearly_comp * 100

    results = {
        'Train': calculate_returns(train_df),
        'Test': calculate_returns(test_df),
        'Val': calculate_returns(val_df),
        'Full': calculate_returns(full_df)
    }

    summary_df = pd.DataFrame.from_dict(
        results, 
        orient='index', 
        columns=['Monthly %', 'Quarterly %', 'Yearly %']
    ).round(2)

    return summary_df

def plot_returns_summary_table(summary_df, title="Returns Summary"):
    """
    Crea una tabla visual de Plotly para el resumen de rendimientos.
    
    Parameters
    ----------
    summary_df : pandas.DataFrame
        DataFrame con rendimientos (output de calculate_returns_summary)
    title : str
        Título de la tabla
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    
    # Preparar datos para la tabla
    datasets = summary_df.index.tolist()
    columns = summary_df.columns.tolist()
    
    # Crear valores formateados con colores condicionales
    cell_colors = []
    cell_values = []
    
    # Header values
    header_values = ['Dataset'] + columns
    
    # Preparar cada columna
    for col in ['Dataset'] + columns:
        if col == 'Dataset':
            cell_values.append(datasets)
            cell_colors.append(['#f8f9fa'] * len(datasets))
        else:
            values = summary_df[col].values
            # Formatear como porcentaje con 2 decimales
            formatted = [f'{v:.2f}%' for v in values]
            cell_values.append(formatted)
            
            # Colores condicionales: verde si positivo, rojo si negativo
            colors = []
            for v in values:
                if v > 1.0:  # Verde fuerte para >1%
                    colors.append('#d4edda')
                elif v > 0:  # Verde claro para positivo
                    colors.append('#e8f5e9')
                elif v > -1.0:  # Rojo claro para ligeramente negativo
                    colors.append('#fff3cd')
                else:  # Rojo fuerte para <-1%
                    colors.append('#f8d7da')
            cell_colors.append(colors)
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header_values,
            fill_color='#2c3e50',
            font=dict(color='white', size=14, family='Arial Black'),
            align='center',
            height=40
        ),
        cells=dict(
            values=cell_values,
            fill_color=cell_colors,
            font=dict(size=13, family='Arial'),
            align=['left'] + ['center'] * len(columns),
            height=35,
            line_color='#dee2e6',
            line_width=1
        )
    )])
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2c3e50', family='Arial Black')
        ),
        height=300,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor='white'
    )
    
    return fig