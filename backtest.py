#!/usr/bin/env python
import argparse
import logging
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import parallel_dqn_trading as pdt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Backtest DQN Trading System')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights file')
    parser.add_argument('--stocks', type=str, nargs='+', default=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                        help='Stocks to backtest on')
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2020-12-31',
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--benchmark', type=str, choices=['sp500', 'nasdaq', 'buy_hold'], default='buy_hold',
                        help='Benchmark to compare against')
    parser.add_argument('--output', type=str, default='backtest_results',
                        help='Output directory for results')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                        help='Initial balance for portfolio')
    
    return parser.parse_args()

def load_model(model_path, num_stocks, market_features=19, action_space=3):
    """Load a trained model from disk"""
    model = pdt.DQNetwork(num_stocks, market_features, action_space)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def run_backtest(model, data_loader, dates, stock_symbols, initial_balance=10000.0):
    """
    Run a backtest of the model on historical data
    
    Args:
        model: Trained DQN model
        data_loader: AzureStockDataLoader instance
        dates: List of dates to backtest on
        stock_symbols: List of stock symbols
        initial_balance: Initial portfolio balance
        
    Returns:
        DataFrame with portfolio performance data
    """
    # Create environment with backtest dates
    env = pdt.TradingEnvironment(
        data_loader,
        dates,
        stock_symbols,
        transaction_cost=0.001  # Use 0.1% transaction cost for backtest
    )
    
    # Override initial balance
    env.initial_balance = initial_balance
    
    # Reset environment
    state = env.reset()
    
    # Track portfolio performance
    portfolio_history = []
    trades = []
    
    # Run backtest
    done = False
    total_steps = 0
    
    with tqdm(total=len(dates) * 26, desc="Backtesting") as pbar:  # Assuming ~26 steps per day (market hours)
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get actions from model
            actions = {}
            with torch.no_grad():
                q_values = model(state_tensor)
                for i, symbol in enumerate(stock_symbols):
                    action = q_values[i].argmax(dim=1).item()
                    actions[symbol] = action
            
            # Take step in environment
            next_state, rewards, done, info = env.step(actions)
            
            # Record trade information
            for symbol, trade_info in info.get('trades', {}).items():
                trade_type, price = trade_info
                trades.append({
                    'date': info.get('date'),
                    'time_idx': info.get('time_idx'),
                    'symbol': symbol,
                    'type': trade_type,
                    'price': price
                })
            
            # Record portfolio performance
            portfolio_history.append({
                'date': info.get('date'),
                'time_idx': info.get('time_idx'),
                'portfolio_value': info.get('portfolio_value'),
                'balance': info.get('balance'),
                'positions': info.get('positions').copy() if info.get('positions') else {},
                'actions': actions.copy()
            })
            
            # Move to next state
            state = next_state
            total_steps += 1
            pbar.update(1)
    
    # Convert to DataFrame
    portfolio_df = pd.DataFrame(portfolio_history)
    trades_df = pd.DataFrame(trades)
    
    return portfolio_df, trades_df

def create_benchmark(portfolio_df, stock_data, stock_symbols, benchmark_type='buy_hold', initial_balance=10000.0):
    """
    Create benchmark performance data for comparison
    
    Args:
        portfolio_df: Portfolio performance DataFrame from backtest
        stock_data: Dictionary of stock DataFrames
        stock_symbols: List of stock symbols
        benchmark_type: Type of benchmark ('buy_hold', 'sp500', 'nasdaq')
        initial_balance: Initial portfolio balance
        
    Returns:
        DataFrame with benchmark performance data
    """
    if benchmark_type == 'buy_hold':
        # Buy and hold strategy - buy equal amounts of each stock on day 1 and hold
        benchmark_data = []
        
        # Get start and end dates
        start_date = portfolio_df['date'].min()
        dates = sorted(portfolio_df['date'].unique())
        
        # Calculate initial allocation per stock
        allocation_per_stock = initial_balance / len(stock_symbols)
        
        # Calculate shares to buy for each stock
        shares = {}
        for symbol in stock_symbols:
            # Get first day's price
            first_day_data = stock_data.get(symbol)
            if first_day_data is None or first_day_data.empty:
                shares[symbol] = 0
                continue
            
            # Use first price of the day
            first_price = first_day_data.iloc[0]['close']
            shares[symbol] = allocation_per_stock / first_price
        
        # Calculate daily portfolio value
        for date in dates:
            portfolio_value = 0
            
            for symbol in stock_symbols:
                if symbol not in stock_data or date not in stock_data[symbol]['date']:
                    continue
                
                # Get closing price for the day
                day_data = stock_data[symbol][stock_data[symbol]['date'] == date]
                if day_data.empty:
                    continue
                
                last_price = day_data.iloc[-1]['close']
                portfolio_value += shares[symbol] * last_price
            
            benchmark_data.append({
                'date': date,
                'portfolio_value': portfolio_value
            })
        
        benchmark_df = pd.DataFrame(benchmark_data)
        
    elif benchmark_type in ['sp500', 'nasdaq']:
        # Use SP500 or NASDAQ index as benchmark
        # This would normally fetch index data from an external source, but for simplicity
        # we'll simulate it with a simple growth model
        benchmark_data = []
        dates = sorted(portfolio_df['date'].unique())
        
        # Simplified model: 8% annual growth with some daily fluctuation
        if benchmark_type == 'sp500':
            daily_return_mean = 0.08 / 252  # Average daily return for 8% annual return
            daily_return_std = 0.01  # Daily volatility
        else:  # nasdaq
            daily_return_mean = 0.1 / 252  # Higher average for NASDAQ
            daily_return_std = 0.015  # Higher volatility
        
        # Simulate index performance
        value = initial_balance
        for date in dates:
            # Generate random daily return
            daily_return = np.random.normal(daily_return_mean, daily_return_std)
            value *= (1 + daily_return)
            
            benchmark_data.append({
                'date': date,
                'portfolio_value': value
            })
        
        benchmark_df = pd.DataFrame(benchmark_data)
    
    return benchmark_df

def calculate_performance_metrics(portfolio_df, benchmark_df=None):
    """
    Calculate performance metrics for the portfolio
    
    Args:
        portfolio_df: Portfolio performance DataFrame
        benchmark_df: Benchmark performance DataFrame (optional)
        
    Returns:
        Dictionary of performance metrics
    """
    # Extract daily portfolio values
    daily_values = portfolio_df.groupby('date')['portfolio_value'].last()
    
    # Calculate daily returns
    daily_returns = daily_values.pct_change().dropna()
    
    # Calculate total return
    initial_value = daily_values.iloc[0]
    final_value = daily_values.iloc[-1]
    total_return = (final_value - initial_value) / initial_value
    
    # Calculate annualized return
    days = (portfolio_df['date'].max() - portfolio_df['date'].min()).days
    annual_factor = 252 / max(days, 1)  # Trading days in a year
    annualized_return = (1 + total_return) ** annual_factor - 1
    
    # Calculate volatility (standard deviation of returns)
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized
    
    # Calculate Sharpe Ratio (assuming 0% risk-free rate for simplicity)
    sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
    
    # Calculate Sortino Ratio (downside risk only)
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = (daily_returns.mean() * 252) / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else np.nan
    
    # Calculate Maximum Drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Calculate metrics vs benchmark if provided
    benchmark_metrics = {}
    if benchmark_df is not None:
        # Extract daily benchmark values
        daily_benchmark = benchmark_df.groupby('date')['portfolio_value'].last()
        
        # Calculate benchmark returns
        benchmark_returns = daily_benchmark.pct_change().dropna()
        
        # Align returns (ensure same dates)
        aligned_returns = pd.concat([daily_returns, benchmark_returns], axis=1, join='inner')
        aligned_returns.columns = ['portfolio', 'benchmark']
        
        # Calculate Beta (correlation with benchmark times ratio of volatilities)
        portfolio_returns = aligned_returns['portfolio']
        benchmark_returns = aligned_returns['benchmark']
        
        beta = (portfolio_returns.cov(benchmark_returns) / benchmark_returns.var()) if benchmark_returns.var() > 0 else np.nan
        
        # Calculate Alpha (excess return over benchmark, risk-adjusted)
        alpha = annualized_return - beta * (benchmark_returns.mean() * 252)
        
        # Calculate Information Ratio (active return divided by tracking error)
        active_returns = portfolio_returns - benchmark_returns
        information_ratio = (active_returns.mean() * 252) / (active_returns.std() * np.sqrt(252)) if active_returns.std() > 0 else np.nan
        
        # Calculate benchmark metrics
        benchmark_total_return = (daily_benchmark.iloc[-1] - daily_benchmark.iloc[0]) / daily_benchmark.iloc[0]
        benchmark_annual_return = (1 + benchmark_total_return) ** annual_factor - 1
        
        benchmark_metrics = {
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'benchmark_total_return': benchmark_total_return,
            'benchmark_annual_return': benchmark_annual_return
        }
    
    # Compile all metrics
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'initial_value': initial_value,
        'final_value': final_value,
        'num_days': days
    }
    
    metrics.update(benchmark_metrics)
    
    return metrics

def create_performance_charts(portfolio_df, trades_df, benchmark_df=None, output_dir='backtest_results'):
    """
    Create performance charts for the backtest
    
    Args:
        portfolio_df: Portfolio performance DataFrame
        trades_df: DataFrame with trade information
        benchmark_df: Benchmark performance DataFrame (optional)
        output_dir: Directory to save charts
        
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set(font_scale=1.2)
    
    # Chart 1: Portfolio Value Over Time
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract daily portfolio values
    daily_values = portfolio_df.groupby('date')['portfolio_value'].last()
    daily_values.plot(ax=ax, linewidth=2, label='Portfolio Value')
    
    # Add benchmark if available
    if benchmark_df is not None:
        daily_benchmark = benchmark_df.groupby('date')['portfolio_value'].last()
        daily_benchmark.plot(ax=ax, linewidth=2, label='Benchmark', linestyle='--')
    
    # Add buy/sell markers
    if not trades_df.empty:
        buy_trades = trades_df[trades_df['type'] == 'buy']
        sell_trades = trades_df[trades_df['type'] == 'sell']
        
        for _, trade in buy_trades.iterrows():
            date = trade['date']
            if date in daily_values.index:
                value = daily_values[date]
                ax.scatter(date, value, color='green', s=50, marker='^', alpha=0.7)
        
        for _, trade in sell_trades.iterrows():
            date = trade['date']
            if date in daily_values.index:
                value = daily_values[date]
                ax.scatter(date, value, color='red', s=50, marker='v', alpha=0.7)
    
    ax.set_title('Portfolio Value Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value ($)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'portfolio_value.png'), dpi=300)
    
    # Chart 2: Cumulative Returns
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate daily returns
    daily_returns = daily_values.pct_change().dropna()
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    cumulative_returns.plot(ax=ax, linewidth=2, label='Portfolio')
    
    # Add benchmark if available
    if benchmark_df is not None:
        benchmark_returns = benchmark_df.groupby('date')['portfolio_value'].last().pct_change().dropna()
        cumulative_benchmark_returns = (1 + benchmark_returns).cumprod() - 1
        cumulative_benchmark_returns.plot(ax=ax, linewidth=2, label='Benchmark', linestyle='--')
    
    ax.set_title('Cumulative Returns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (%)')
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cumulative_returns.png'), dpi=300)
    
    # Chart 3: Drawdown
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    drawdown.plot(ax=ax, linewidth=2, label='Portfolio')
    
    # Add benchmark drawdown if available
    if benchmark_df is not None:
        benchmark_returns = benchmark_df.groupby('date')['portfolio_value'].last().pct_change().dropna()
        benchmark_cum_returns = (1 + benchmark_returns).cumprod()
        benchmark_peak = benchmark_cum_returns.cummax()
        benchmark_drawdown = (benchmark_cum_returns - benchmark_peak) / benchmark_peak
        benchmark_drawdown.plot(ax=ax, linewidth=2, label='Benchmark', linestyle='--')
    
    ax.set_title('Drawdown Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'drawdown.png'), dpi=300)
    
    # Chart 4: Position by Stock
    positions_by_date = {}
    
    for _, row in portfolio_df.iterrows():
        date = row['date']
        if date not in positions_by_date:
            positions_by_date[date] = {}
        
        for symbol, position in row['positions'].items():
            positions_by_date[date][symbol] = position
    
    # Convert to DataFrame
    position_df = pd.DataFrame(positions_by_date).T
    position_df = position_df.fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    position_df.plot(ax=ax, kind='area', stacked=True)
    ax.set_title('Portfolio Positions Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Position (1=Long, 0=None)')
    ax.legend(title='Stock')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'positions.png'), dpi=300)
    
    # Chart 5: Monthly Returns Heatmap
    if len(daily_returns) > 20:  # Only if we have enough data
        # Calculate monthly returns
        monthly_returns = daily_returns.groupby([daily_returns.index.year, daily_returns.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Reshape to heatmap format
        monthly_returns_pivot = monthly_returns.unstack()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(monthly_returns_pivot, annot=True, fmt='.1%', cmap='RdYlGn', ax=ax,
                    cbar=True, cbar_kws={'label': 'Monthly Return'})
        ax.set_title('Monthly Returns (%)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'monthly_returns.png'), dpi=300)

def load_stock_data_for_dates(data_loader, dates, stock_symbols):
    """Load stock data for all specified dates"""
    all_stock_data = {}
    
    for symbol in stock_symbols:
        all_stock_data[symbol] = []
        
        for date in dates:
            df = data_loader.load_stock_data(symbol, date)
            if df is not None and not df.empty:
                # Add date column for easier grouping
                df['date'] = date
                all_stock_data[symbol].append(df)
        
        # Concatenate all data for this stock
        if all_stock_data[symbol]:
            all_stock_data[symbol] = pd.concat(all_stock_data[symbol])
        else:
            all_stock_data[symbol] = pd.DataFrame()
    
    return all_stock_data

def main():
    args = parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create data loader
    data_loader = pdt.AzureStockDataLoader(
        pdt.AZURE_STORAGE_CONNECTION_STRING,
        pdt.CONTAINER_NAME,
        pdt.BLOB_BASE_URL,
        args.stocks
    )
    
    # Get available dates in the range
    logger.info("Getting available dates...")
    available_dates = data_loader.get_training_dates(
        start_date=start_date,
        end_date=end_date
    )
    logger.info(f"Found {len(available_dates)} available dates")
    
    if not available_dates:
        logger.error("No dates available in the specified range")
        return
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, len(args.stocks))
    
    # Run backtest
    logger.info("Running backtest...")
    portfolio_df, trades_df = run_backtest(
        model, 
        data_loader, 
        available_dates, 
        args.stocks,
        initial_balance=args.initial_balance
    )
    
    # Save raw results
    portfolio_df.to_csv(os.path.join(args.output, 'portfolio.csv'), index=False)
    trades_df.to_csv(os.path.join(args.output, 'trades.csv'), index=False)
    
    # Load stock data for benchmark creation
    logger.info("Loading stock data for benchmark...")
    stock_data = load_stock_data_for_dates(data_loader, available_dates, args.stocks)
    
    # Create benchmark
    logger.info(f"Creating benchmark ({args.benchmark})...")
    benchmark_df = create_benchmark(
        portfolio_df, 
        stock_data, 
        args.stocks,
        benchmark_type=args.benchmark,
        initial_balance=args.initial_balance
    )
    
    # Calculate performance metrics
    logger.info("Calculating performance metrics...")
    metrics = calculate_performance_metrics(portfolio_df, benchmark_df)
    
    # Print performance metrics
    logger.info("Performance Metrics:")
    for key, value in metrics.items():
        if 'return' in key or key in ['alpha', 'beta', 'sharpe_ratio', 'sortino_ratio', 'information_ratio', 'max_drawdown']:
            logger.info(f"  {key}: {value:.2%}")
        elif key in ['initial_value', 'final_value', 'volatility']:
            logger.info(f"  {key}: ${value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Save metrics to CSV
    pd.DataFrame([metrics]).to_csv(os.path.join(args.output, 'metrics.csv'), index=False)
    
    # Create performance charts
    logger.info("Creating performance charts...")
    create_performance_charts(portfolio_df, trades_df, benchmark_df, args.output)
    
    logger.info(f"Backtest completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()