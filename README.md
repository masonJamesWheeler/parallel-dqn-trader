# Parallel DQN Trading System

This project implements a distributed, parallel Deep Q-Network (DQN) trading system that can learn across multiple securities while maintaining a "collective intelligence" structure.

## Security Updates

**IMPORTANT**: This project now uses environment variables for API keys and connection strings. Set up your environment by:

1. Creating a `.cloudkeys/api_keys.env` file with:
```
ALPHAVANTAGE_API_KEY=your_alphavantage_key
AZURE_STORAGE_CONNECTION_STRING=your_azure_connection_string
```

2. Run scripts using the provided helper:
```bash
./run_with_env.sh run_dqn_trading.py --mode single
```

## Architecture

The system consists of the following key components:

1. **Worker Nodes**: Each worker handles a different stock but contributes to collective intelligence
   - Local DQN: Processes stock-specific data and makes trading decisions
   - Local Buffer: Stores immediate experiences before synchronization
   - Time-step Synchronization: Ensures all agents process the same market time periods

2. **Cross-Stock Relationship Module**: Models relationships between different stocks
   - Extracts correlation patterns between stocks
   - Detects arbitrage opportunities across markets
   - Maintains a dynamic relationship graph that evolves as markets change

3. **Parameter Server Architecture**: Provides the backbone for knowledge sharing
   - Central parameter server maintains global model weights
   - Workers periodically push gradients and pull updated parameters
   - Asynchronous updates with prioritized experience replay

4. **Meta-Policy Layer**: Handles the high-level coordination between agents
   - Allocates different trading strategies across workers
   - Optimizes portfolio-level objectives
   - Prevents agent actions from interfering with each other

## Requirements

- Python 3.8+
- PyTorch
- MPI4py (for parallel training)
- pandas
- numpy
- Azure Storage SDK
- pyarrow

You can install the dependencies with:

```bash
pip install torch pandas numpy azure-storage-blob pyarrow mpi4py
```

## Data Storage

The system uses Azure Blob Storage for stock data. The data is organized in the following format:

```
stocks_intraday/{symbol}/{year}/{month}/{symbol}_{date}.parquet
```

For example:
```
stocks_intraday/AAPL/2020/01/AAPL_2020-01-03.parquet
```

## Usage

### Single-Process Training

To train the system on a single process:

```bash
./run_with_env.sh run_dqn_trading.py --mode single --stocks AAPL MSFT GOOGL --start_date 2018-01-01 --end_date 2019-12-31 --output results
```

### Parallel Training with MPI

To train the system using multiple processes with MPI:

```bash
mpirun -n 6 ./run_with_env.sh run_dqn_trading.py --mode parallel --stocks AAPL MSFT GOOGL AMZN META
```

This will start 1 parameter server and 5 worker nodes (1 for each stock).

### Command-Line Arguments

- `--mode`: Training mode, either 'single' or 'parallel' (default: 'single')
- `--stocks`: List of stock symbols to train on (default: ["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
- `--start_date`: Start date for training data in YYYY-MM-DD format (default: '2018-01-01')
- `--end_date`: End date for training data in YYYY-MM-DD format (default: '2019-12-31')
- `--eval_start`: Start date for evaluation data in YYYY-MM-DD format (default: '2020-01-01')
- `--eval_end`: End date for evaluation data in YYYY-MM-DD format (default: '2020-12-31')
- `--episodes`: Number of training episodes (default: 100)
- `--output`: Output directory for results (default: 'results')

## Key Features and Innovations

1. **Cross-Stock Relationship Module**: Dynamically learns relationships between different stocks using an attention mechanism to emphasize important cross-stock signals

2. **Distributed Reinforcement Learning**: Uses a parameter server architecture where workers handle different stocks independently but share knowledge through global model weights

3. **Meta-Policy Layer**: Ensures trading strategies are coordinated and optimizes for portfolio-level returns rather than individual stock performance

4. **Advanced Market State Representation**: Captures technical indicators across different time scales, current positions, price movements, volatility, and more

5. **Synchronized Time Steps**: Maintains strict time synchronization to enable genuine learning of cross-asset relationships without future information leakage

## Performance Metrics

The system tracks the following performance metrics:

- Total reward
- Portfolio returns
- Sharpe ratio
- Maximum drawdown

## File Structure

- `parallel_dqn_trading.py`: Core implementation of the parallel DQN trading system
- `run_dqn_trading.py`: Script to run the training in single or parallel mode
- `results/`: Directory for output files (models, performance data)

## Output Files

- `trained_model.pth`: Saved PyTorch model after training
- `portfolio_ep{episode}.csv`: Portfolio performance data for evaluation periods# parallel-dqn-trader
# parallel-dqn-trader
# parallel-dqn-trader
