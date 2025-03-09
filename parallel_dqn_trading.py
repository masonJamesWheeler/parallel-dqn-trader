import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from mpi4py import MPI
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import time
import pyarrow.parquet as pq
from io import BytesIO
import requests
from datetime import datetime, timedelta
import logging
from azure.storage.blob import BlobServiceClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set up device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Azure Storage settings
AZURE_STORAGE_CONNECTION_STRING = os.environ.get('AZURE_STORAGE_CONNECTION_STRING', '')
CONTAINER_NAME = "stock-data"
BLOB_BASE_URL = "https://wheelermasonstockdata.blob.core.windows.net/stock-data/stocks_intraday"

# Training Constants
GAMMA = 0.99            # Discount factor
BATCH_SIZE = 64         # Minibatch size for training
BUFFER_SIZE = 100000    # Replay buffer size
MIN_REPLAY_SIZE = 1000  # Min replay buffer size before training
EPSILON_START = 1.0     # Start value of epsilon
EPSILON_END = 0.1       # End value of epsilon
EPSILON_DECAY = 10000   # Decay rate of epsilon
TARGET_UPDATE_FREQ = 1000  # How often to update target network
SYNC_FREQ = 500         # How often to sync with parameter server
LEARNING_RATE = 1e-4    # Learning rate
MARKET_FEATURES = 20    # Number of market features per stock
ACTION_SPACE = 3        # Buy, Sell, Hold

# Stocks to train on
TRAINING_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
NUM_STOCKS = len(TRAINING_STOCKS)  # Number of stocks for training

# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'next_state', 'reward', 'done', 'stock_id'])

class AzureStockDataLoader:
    """Loads stock data from Azure Blob Storage"""
    def __init__(self, connection_string, container_name, base_url, stocks):
        self.connection_string = connection_string
        self.container_name = container_name
        self.base_url = base_url
        self.stocks = stocks
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.data_cache = {}
        
    def list_available_dates(self, symbol, start_year=2000, end_year=2020):
        """List all available dates for a stock symbol"""
        available_dates = []
        try:
            # List all blobs with the prefix for this stock
            blob_prefix = f"stocks_intraday/{symbol}/"
            blobs = self.container_client.list_blobs(name_starts_with=blob_prefix)
            
            for blob in blobs:
                # Extract date from blob name
                filename = blob.name.split('/')[-1]
                if filename.endswith('.parquet'):
                    date_str = filename.split('_')[-1].replace('.parquet', '')
                    date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    year = date.year
                    if start_year <= year <= end_year:
                        available_dates.append(date)
                        
            return sorted(available_dates)
        except Exception as e:
            logger.error(f"Error listing available dates for {symbol}: {e}")
            return []
            
    def load_stock_data(self, symbol, date):
        """Load stock data for a specific symbol and date"""
        # Check if data is in cache
        cache_key = f"{symbol}_{date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Format date components for URL
        year = date.year
        month = date.month
        day = date.day
        
        # Construct URL path
        file_path = f"stocks_intraday/{symbol}/{year}/{month:02d}/{symbol}_{date.strftime('%Y-%m-%d')}.parquet"
        
        try:
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=file_path
            )
            
            # Download blob
            blob_data = blob_client.download_blob()
            data = blob_data.readall()
            
            # Load Parquet file
            table = pq.read_table(BytesIO(data))
            df = table.to_pandas()
            
            # Process dataframe
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                # Add technical indicators
                self._add_technical_indicators(df)
                
                # Cache the result
                self.data_cache[cache_key] = df
                return df
            else:
                logger.error(f"Invalid data format for {symbol} on {date}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading data for {symbol} on {date}: {e}")
            return None
            
    def _add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate moving averages
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        
        # Relative price positions
        df['pct_ma5'] = (df['close'] - df['ma5']) / df['ma5']
        df['pct_ma10'] = (df['close'] - df['ma10']) / df['ma10']
        df['pct_ma20'] = (df['close'] - df['ma20']) / df['ma20']
        
        # Calculate RSI (14-period)
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['ema12'] = df['close'].ewm(span=12).mean()
        df['ema26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=10).std()
        
        # Normalized volume
        df['volume_ma10'] = df['volume'].rolling(window=10).mean()
        df['norm_volume'] = df['volume'] / df['volume_ma10']
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
        
    def load_synchronous_market_data(self, date, lookback_days=10):
        """
        Load market data for all stocks on the same date
        Returns a dictionary of dataframes, one for each stock
        """
        market_data = {}
        
        for symbol in self.stocks:
            # Get data for this symbol on this date
            df = self.load_stock_data(symbol, date)
            if df is not None and not df.empty:
                market_data[symbol] = df
            else:
                logger.warning(f"No data for {symbol} on {date}")
                
        return market_data
        
    def get_training_dates(self, start_date=None, end_date=None):
        """
        Get a list of dates for training that have data for all stocks
        """
        # Default date range
        if start_date is None:
            start_date = datetime(2018, 1, 1).date()
        if end_date is None:
            end_date = datetime(2020, 12, 31).date()
        
        # Get available dates for each stock
        available_dates_by_stock = {}
        for symbol in self.stocks:
            available_dates = self.list_available_dates(symbol, 
                                                      start_date.year, 
                                                      end_date.year)
            available_dates_by_stock[symbol] = set(
                date for date in available_dates
                if start_date <= date <= end_date
            )
        
        # Find dates that have data for all stocks
        common_dates = set.intersection(*available_dates_by_stock.values())
        
        return sorted(list(common_dates))

class MarketStateEncoder(nn.Module):
    """Encodes market state across multiple stocks"""
    def __init__(self, num_stocks, market_features, hidden_dim=128):
        super(MarketStateEncoder, self).__init__()
        self.num_stocks = num_stocks
        self.market_features = market_features
        
        # Individual stock encoders
        self.stock_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(market_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            ) for _ in range(num_stocks)
        ])
        
        # Cross-stock attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2, 
            num_heads=4,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
    
    def forward(self, market_states):
        """
        Args:
            market_states: Tensor of shape [batch_size, num_stocks, market_features]
        """
        batch_size = market_states.shape[0]
        
        # Encode each stock's state
        encoded_stocks = []
        for i in range(self.num_stocks):
            stock_state = market_states[:, i, :]
            encoded = self.stock_encoders[i](stock_state)
            encoded_stocks.append(encoded)
        
        # Stack encodings [batch_size, num_stocks, hidden_dim//2]
        encoded_stack = torch.stack(encoded_stocks, dim=1)
        
        # Apply cross-stock attention
        attn_output, _ = self.attention(
            encoded_stack, encoded_stack, encoded_stack
        )
        
        # Project output
        output = self.output_projection(attn_output)
        
        return output

class CrossStockRelationship(nn.Module):
    """Models relationships between different stocks"""
    def __init__(self, num_stocks, hidden_dim=64):
        super(CrossStockRelationship, self).__init__()
        self.num_stocks = num_stocks
        
        # Relationship embedding matrix
        self.relationship_matrix = nn.Parameter(
            torch.zeros(num_stocks, num_stocks, hidden_dim)
        )
        
        # Initialize with small random values
        nn.init.xavier_uniform_(self.relationship_matrix)
        
        # Relationship encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, stock_idx, encoded_states):
        """
        Args:
            stock_idx: Index of the stock we're looking at
            encoded_states: Tensor of shape [batch_size, num_stocks, hidden_dim]
        """
        batch_size = encoded_states.shape[0]
        
        # Get relationships for this stock
        relationships = self.relationship_matrix[stock_idx]  # [num_stocks, hidden_dim]
        
        # Compute relationship influence
        relationship_influence = torch.zeros(batch_size, encoded_states.shape[-1]).to(device)
        
        for i in range(self.num_stocks):
            if i == stock_idx:
                continue  # Skip self
                
            # Encode relationship
            rel_encoded = self.encoder(relationships[i])
            
            # Apply relationship to encoded state
            influence = encoded_states[:, i, :] * rel_encoded
            
            # Add to total influence
            relationship_influence += influence
            
        return relationship_influence

class DQNetwork(nn.Module):
    """Deep Q-Network for trading decisions"""
    def __init__(self, num_stocks, market_features, action_space, hidden_dim=128):
        super(DQNetwork, self).__init__()
        self.num_stocks = num_stocks
        self.market_features = market_features
        self.action_space = action_space
        
        # Market state encoder
        self.market_encoder = MarketStateEncoder(
            num_stocks, market_features, hidden_dim
        )
        
        # Cross-stock relationship module
        self.relationship_module = CrossStockRelationship(
            num_stocks, hidden_dim // 2
        )
        
        # Stock-specific Q-value heads
        self.q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_space)
            ) for _ in range(num_stocks)
        ])
    
    def forward(self, states, stock_idx=None):
        """
        Args:
            states: Tensor of shape [batch_size, num_stocks, market_features]
            stock_idx: If provided, only return Q-values for this stock
        """
        # Encode market states
        encoded_states = self.market_encoder(states)
        
        if stock_idx is not None:
            # Get relationship influence
            rel_influence = self.relationship_module(stock_idx, encoded_states)
            
            # Combine with stock's encoded state
            combined = torch.cat([
                encoded_states[:, stock_idx, :], 
                rel_influence
            ], dim=-1)
            
            # Compute Q-values for this stock
            q_values = self.q_heads[stock_idx](combined)
            
            return q_values
        else:
            # Return Q-values for all stocks
            q_values = []
            
            for i in range(self.num_stocks):
                # Get relationship influence
                rel_influence = self.relationship_module(i, encoded_states)
                
                # Combine with stock's encoded state
                combined = torch.cat([
                    encoded_states[:, i, :], 
                    rel_influence
                ], dim=-1)
                
                # Compute Q-values for this stock
                stock_q = self.q_heads[i](combined)
                q_values.append(stock_q)
            
            return q_values

class ExperienceReplay:
    """Prioritized experience replay buffer"""
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling exponent
        self.beta_increment = 0.001  # Increment for beta
        self.epsilon = 1e-6  # Small constant to avoid zero priority
    
    def add(self, state, action, next_state, reward, done, stock_id, error=None):
        """Add experience to buffer"""
        experience = Experience(state, action, next_state, reward, done, stock_id)
        self.buffer.append(experience)
        
        # If error is not provided, use max priority
        if error is None:
            priority = max(self.priorities) if self.priorities else 1.0
        else:
            priority = abs(error) + self.epsilon
            
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        """Sample batch of experiences"""
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priority
        indices = np.random.choice(
            len(self.buffer), 
            batch_size, 
            replace=False, 
            p=probs
        )
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** -self.beta
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, weights, indices
    
    def update_priorities(self, indices, errors):
        """Update priorities based on new TD errors"""
        for i, error in zip(indices, errors):
            if i < len(self.priorities):
                self.priorities[i] = abs(error) + self.epsilon
    
    def __len__(self):
        return len(self.buffer)

class TradingEnvironment:
    """Environment for stock trading simulation"""
    def __init__(self, data_loader, training_dates, stock_symbols, window_size=10, transaction_cost=0.001):
        """
        Args:
            data_loader: AzureStockDataLoader instance
            training_dates: List of dates for training
            stock_symbols: List of stock symbols to trade
            window_size: Number of past time steps to include in state
            transaction_cost: Cost of making a trade as a fraction of trade value
        """
        self.data_loader = data_loader
        self.training_dates = training_dates
        self.stock_symbols = stock_symbols
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.num_stocks = len(stock_symbols)
        
        # Mapping from stock symbol to index
        self.stock_to_idx = {symbol: i for i, symbol in enumerate(stock_symbols)}
        
        # Current position for each stock (0: no position, 1: long)
        self.positions = {symbol: 0 for symbol in stock_symbols}
        
        # Current date index
        self.current_date_idx = 0
        
        # Current time step within the day
        self.current_time_idx = 0
        
        # Market data for current day
        self.current_market_data = None
        
        # Track portfolio value
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        
        # Feature columns to use
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'ma5', 'ma10', 'ma20',
            'pct_ma5', 'pct_ma10', 'pct_ma20',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'volatility', 'norm_volume'
        ]
        
        # Initialize environment
        self._initialize()
    
    def _initialize(self):
        """Initialize the environment with the first date's data"""
        if len(self.training_dates) > 0:
            date = self.training_dates[0]
            self.current_market_data = self.data_loader.load_synchronous_market_data(date)
            
            # Check if we have data for all stocks
            missing_stocks = [s for s in self.stock_symbols if s not in self.current_market_data]
            if missing_stocks:
                logger.warning(f"Missing data for stocks: {missing_stocks} on {date}")
        else:
            logger.error("No training dates available")
    
    def _get_state_features(self, symbol):
        """Extract features for a stock at the current time step"""
        df = self.current_market_data.get(symbol)
        if df is None or df.empty or self.current_time_idx >= len(df):
            return np.zeros(len(self.feature_columns) + 1)  # +1 for position
        
        # Get data up to current time
        current_data = df.iloc[self.current_time_idx]
        
        # Extract features
        features = current_data[self.feature_columns].values
        
        # Add position feature
        position_feature = np.array([self.positions[symbol]])
        
        # Combine features
        state_features = np.concatenate([features, position_feature])
        
        return state_features
    
    def _get_state(self):
        """Get current state representation for all stocks"""
        state = np.zeros((self.num_stocks, len(self.feature_columns) + 1))
        
        for i, symbol in enumerate(self.stock_symbols):
            state[i] = self._get_state_features(symbol)
        
        return state
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_date_idx = 0
        self.current_time_idx = 0
        self.positions = {symbol: 0 for symbol in self.stock_symbols}
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        
        # Load market data for the first date
        date = self.training_dates[self.current_date_idx]
        self.current_market_data = self.data_loader.load_synchronous_market_data(date)
        
        return self._get_state()
    
    def step(self, actions):
        """
        Take a step in the environment
        
        Args:
            actions: Dictionary mapping stock_symbol to action (0: hold, 1: buy, 2: sell)
        
        Returns:
            next_state: New state
            rewards: Dictionary of rewards for each stock
            done: Whether episode is done
            info: Additional information
        """
        rewards = {}
        info = {'trades': {}}
        
        # Track portfolio value before actions
        prev_portfolio_value = self.portfolio_value
        
        # Process actions for each stock
        for symbol, action in actions.items():
            idx = self.stock_to_idx.get(symbol)
            if idx is None:
                continue
                
            # Get current market data
            df = self.current_market_data.get(symbol)
            if df is None or df.empty or self.current_time_idx >= len(df):
                rewards[symbol] = 0
                continue
                
            # Get current price data
            current_data = df.iloc[self.current_time_idx]
            current_price = current_data['close']
            
            # Get previous price (for calculating returns)
            prev_price = current_price
            if self.current_time_idx > 0:
                prev_price = df.iloc[self.current_time_idx - 1]['close']
            
            price_change_pct = (current_price - prev_price) / prev_price if prev_price > 0 else 0
            
            # Initialize reward
            reward = 0
            
            # Execute action
            if action == 1:  # Buy
                if self.positions[symbol] == 0:  # Only buy if no position
                    # Calculate transaction cost
                    cost = current_price * self.transaction_cost
                    
                    # Update position and balance
                    self.positions[symbol] = 1
                    self.balance -= (current_price + cost)
                    
                    info['trades'][symbol] = ('buy', current_price)
                    reward = -cost  # Immediate reward is negative (transaction cost)
                else:
                    reward = 0  # No reward for invalid action
                    
            elif action == 2:  # Sell
                if self.positions[symbol] == 1:  # Only sell if holding
                    # Calculate transaction cost and profit/loss
                    cost = current_price * self.transaction_cost
                    
                    # Update position and balance
                    self.positions[symbol] = 0
                    self.balance += (current_price - cost)
                    
                    info['trades'][symbol] = ('sell', current_price)
                    reward = price_change_pct - cost  # Reward is price change minus cost
                else:
                    reward = 0  # No reward for invalid action
                    
            else:  # Hold
                if self.positions[symbol] == 1:
                    # If holding, reward is price change (no transaction cost)
                    reward = price_change_pct
                else:
                    reward = 0  # No reward for holding cash
            
            rewards[symbol] = reward
        
        # Move to next time step
        self.current_time_idx += 1
        
        # Check if we need to move to the next day
        done = False
        first_symbol = next(iter(self.current_market_data.keys()))
        if self.current_time_idx >= len(self.current_market_data[first_symbol]):
            # Move to next day
            self.current_date_idx += 1
            self.current_time_idx = 0
            
            # Check if we're done with all days
            if self.current_date_idx >= len(self.training_dates):
                done = True
            else:
                # Load data for the next day
                date = self.training_dates[self.current_date_idx]
                self.current_market_data = self.data_loader.load_synchronous_market_data(date)
        
        # Calculate new portfolio value
        self.portfolio_value = self.balance
        for symbol in self.stock_symbols:
            if self.positions[symbol] == 1:
                df = self.current_market_data.get(symbol)
                if df is not None and not df.empty and self.current_time_idx < len(df):
                    current_price = df.iloc[self.current_time_idx]['close']
                    self.portfolio_value += current_price
        
        # Add portfolio return to rewards
        portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        for symbol in self.stock_symbols:
            rewards[symbol] += portfolio_return / len(self.stock_symbols)
        
        # Get new state
        next_state = self._get_state()
        
        # Additional info
        info['portfolio_value'] = self.portfolio_value
        info['balance'] = self.balance
        info['positions'] = self.positions.copy()
        info['date'] = self.training_dates[self.current_date_idx] if not done else None
        info['time_idx'] = self.current_time_idx
        
        return next_state, rewards, done, info

class MetaPolicyLayer:
    """Coordinates strategies across multiple trading agents"""
    def __init__(self, num_stocks, stock_symbols, hidden_dim=64):
        self.num_stocks = num_stocks
        self.stock_symbols = stock_symbols
        
        # Create correlation matrix
        self.correlation_matrix = torch.zeros(num_stocks, num_stocks).to(device)
        
        # Strategy allocation matrix (initially uniform)
        self.strategy_allocation = torch.ones(num_stocks) / num_stocks
        
        # Create strategy embedding network
        self.strategy_network = nn.Sequential(
            nn.Linear(num_stocks * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_stocks),
            nn.Softmax(dim=0)
        ).to(device)
    
    def update_correlation(self, returns):
        """Update correlation matrix based on returns"""
        # returns: tensor of shape [num_stocks]
        returns_expanded = returns.unsqueeze(1)
        
        # Compute pairwise products
        correlation_update = torch.matmul(
            returns_expanded, returns_expanded.transpose(0, 1)
        )
        
        # Update correlation matrix with moving average
        alpha = 0.05  # Weight for new observation
        self.correlation_matrix = (1 - alpha) * self.correlation_matrix + alpha * correlation_update
    
    def allocate_strategies(self, portfolio_state):
        """Determine strategy allocation based on portfolio state and correlations"""
        # portfolio_state: tensor of shape [num_stocks * 2] 
        # (positions and portfolio weights)
        
        # Use network to determine allocation
        allocation = self.strategy_network(portfolio_state)
        
        # Update strategy allocation
        self.strategy_allocation = allocation
        
        return allocation
    
    def get_strategy_for_stock(self, stock_symbol):
        """Get the strategy allocation for a specific stock"""
        if stock_symbol in self.stock_symbols:
            idx = self.stock_symbols.index(stock_symbol)
            return self.strategy_allocation[idx].item()
        return 0.0

class ParameterServer:
    """Manages global model parameters and synchronization"""
    def __init__(self, model, num_workers, stock_symbols):
        self.model = model
        self.num_workers = num_workers
        self.stock_symbols = stock_symbols
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.global_step = 0
        
        # Initialize target network
        self.target_model = DQNetwork(
            NUM_STOCKS, MARKET_FEATURES, ACTION_SPACE
        ).to(device)
        self.target_model.load_state_dict(model.state_dict())
        
        # Create experience buffer
        self.replay_buffer = ExperienceReplay(BUFFER_SIZE)
        
        # Create meta-policy layer
        self.meta_policy = MetaPolicyLayer(NUM_STOCKS, stock_symbols)
    
    def update(self, worker_gradients, experiences):
        """Update global model with worker gradients"""
        # Apply gradients from worker
        self.optimizer.zero_grad()
        
        # Manually set gradients
        for param, grad in zip(self.model.parameters(), worker_gradients):
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad
        
        # Update parameters
        self.optimizer.step()
        
        # Add experiences to replay buffer
        for exp in experiences:
            self.replay_buffer.add(*exp)
        
        # Global training step
        if len(self.replay_buffer) > MIN_REPLAY_SIZE:
            self._train_step()
        
        # Update target network periodically
        if self.global_step % TARGET_UPDATE_FREQ == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        self.global_step += 1
        
        return self.model.state_dict()
    
    def _train_step(self):
        """Perform one training step using the replay buffer"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        
        # Sample batch from replay buffer
        experiences, weights, indices = self.replay_buffer.sample(BATCH_SIZE)
        
        # Organize batch data
        states = torch.FloatTensor(
            np.array([exp.state for exp in experiences])
        ).to(device)
        
        actions = torch.LongTensor(
            np.array([exp.action for exp in experiences])
        ).to(device)
        
        next_states = torch.FloatTensor(
            np.array([exp.next_state for exp in experiences])
        ).to(device)
        
        rewards = torch.FloatTensor(
            np.array([exp.reward for exp in experiences])
        ).to(device)
        
        dones = torch.FloatTensor(
            np.array([float(exp.done) for exp in experiences])
        ).to(device)
        
        stock_ids = [exp.stock_id for exp in experiences]
        
        # Calculate current Q-values
        current_q_values = []
        for i, exp in enumerate(experiences):
            q_values = self.model(states[i:i+1], exp.stock_id)
            current_q_values.append(q_values[0, actions[i]])
        
        current_q_values = torch.stack(current_q_values)
        
        # Calculate target Q-values
        target_q_values = []
        for i, exp in enumerate(experiences):
            with torch.no_grad():
                # Get next state Q-values from target network
                next_q = self.target_model(next_states[i:i+1], exp.stock_id)
                
                # Double DQN: use online network to select action
                online_next_q = self.model(next_states[i:i+1], exp.stock_id)
                best_action = online_next_q.argmax(dim=1)
                
                # Use target network to evaluate action
                next_q_value = next_q[0, best_action]
                
                # Calculate target using Bellman equation
                target = rewards[i] + GAMMA * next_q_value * (1 - dones[i])
                target_q_values.append(target)
        
        target_q_values = torch.stack(target_q_values)
        
        # Calculate TD errors
        td_errors = target_q_values - current_q_values
        
        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Calculate loss with importance sampling weights
        weights_tensor = torch.FloatTensor(weights).to(device)
        loss = (weights_tensor * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        # Perform optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Worker:
    """Worker process for trading a specific stock"""
    def __init__(self, stock_symbol, stock_idx, env, local_model):
        self.stock_symbol = stock_symbol
        self.stock_idx = stock_idx  # Index in the list of stocks
        self.env = env
        self.local_model = local_model
        self.optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE)
        
        # Create local experience buffer
        self.local_buffer = []
        
        # Training state
        self.epsilon = EPSILON_START
        self.step = 0
    
    def collect_experience(self, num_steps=10):
        """Collect experience using current policy"""
        experiences = []
        
        state = self.env.reset()
        for _ in range(num_steps):
            # Select action (epsilon-greedy)
            if random.random() < self.epsilon:
                action = random.randint(0, ACTION_SPACE - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = self.local_model(state_tensor, self.stock_idx)
                    action = q_values.argmax(dim=1).item()
            
            # Execute action in environment
            actions = {self.stock_symbol: action}
            next_state, rewards, done, info = self.env.step(actions)
            reward = rewards[self.stock_symbol]
            
            # Store experience
            experiences.append((
                state, action, next_state, reward, done, self.stock_idx
            ))
            
            # Update local buffer
            self.local_buffer.append((
                state, action, next_state, reward, done, self.stock_idx
            ))
            
            # Move to next state
            state = next_state
            
            # Update epsilon
            self.epsilon = max(
                EPSILON_END,
                EPSILON_START - (EPSILON_START - EPSILON_END) * self.step / EPSILON_DECAY
            )
            
            self.step += 1
            
            if done:
                break
        
        return experiences
    
    def compute_gradients(self):
        """Compute gradients using local buffer"""
        if len(self.local_buffer) < BATCH_SIZE:
            return None
        
        # Sample from local buffer
        batch = random.sample(self.local_buffer, BATCH_SIZE)
        
        # Organize batch data
        states = torch.FloatTensor(
            np.array([exp[0] for exp in batch])
        ).to(device)
        
        actions = torch.LongTensor(
            np.array([exp[1] for exp in batch])
        ).to(device)
        
        next_states = torch.FloatTensor(
            np.array([exp[2] for exp in batch])
        ).to(device)
        
        rewards = torch.FloatTensor(
            np.array([exp[3] for exp in batch])
        ).to(device)
        
        dones = torch.FloatTensor(
            np.array([float(exp[4]) for exp in batch])
        ).to(device)
        
        # Get current Q-values
        q_values = self.local_model(states, self.stock_idx)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate target Q-values
        with torch.no_grad():
            # Get next state Q-values
            next_q = self.local_model(next_states, self.stock_idx)
            next_q_values = next_q.max(1)[0]
            
            # Calculate targets using Bellman equation
            targets = rewards + GAMMA * next_q_values * (1 - dones)
        
        # Calculate loss
        loss = F.mse_loss(q_values, targets)
        
        # Compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        
        # Get gradients
        gradients = [param.grad.clone() for param in self.local_model.parameters()]
        
        # Clear local buffer after computing gradients
        self.local_buffer = []
        
        return gradients, batch
    
    def update_model(self, model_state_dict):
        """Update local model with global parameters"""
        self.local_model.load_state_dict(model_state_dict)

def run_parameter_server():
    """Run the parameter server process"""
    print("Starting parameter server (rank 0)")
    
    # Create global model
    global_model = DQNetwork(
        NUM_STOCKS, MARKET_FEATURES, ACTION_SPACE
    ).to(device)
    
    # Create parameter server
    param_server = ParameterServer(global_model, size - 1, TRAINING_STOCKS)
    
    # Main server loop
    running = True
    while running:
        # Wait for gradient update from any worker
        status = MPI.Status()
        worker_rank = comm.recv(source=MPI.ANY_SOURCE, tag=1, status=status)
        
        # Receive gradients from worker
        gradients = comm.recv(source=worker_rank, tag=2)
        
        # Receive experiences from worker
        experiences = comm.recv(source=worker_rank, tag=3)
        
        # Update global model
        model_state_dict = param_server.update(gradients, experiences)
        
        # Send updated model back to worker
        comm.send(model_state_dict, dest=worker_rank, tag=4)

def run_worker(stock_symbol, stock_idx):
    """Run a worker process for a specific stock"""
    print(f"Starting worker for stock {stock_symbol} (rank {rank})")
    
    # Create data loader
    data_loader = AzureStockDataLoader(
        AZURE_STORAGE_CONNECTION_STRING,
        CONTAINER_NAME,
        BLOB_BASE_URL,
        TRAINING_STOCKS
    )
    
    # Get training dates
    training_dates = data_loader.get_training_dates()
    
    # Create trading environment
    env = TradingEnvironment(data_loader, training_dates, TRAINING_STOCKS)
    
    # Create local model
    local_model = DQNetwork(
        NUM_STOCKS, MARKET_FEATURES, ACTION_SPACE
    ).to(device)
    
    # Create worker
    worker = Worker(stock_symbol, stock_idx, env, local_model)
    
    # Main worker loop
    while True:
        # Collect experience
        experiences = worker.collect_experience()
        
        # Compute gradients
        result = worker.compute_gradients()
        if result is not None:
            gradients, batch = result
            
            # Send gradients to parameter server
            comm.send(rank, dest=0, tag=1)
            comm.send(gradients, dest=0, tag=2)
            comm.send(batch, dest=0, tag=3)
            
            # Receive updated model
            model_state_dict = comm.recv(source=0, tag=4)
            
            # Update local model
            worker.update_model(model_state_dict)

def evaluate_model(model, data_loader, evaluation_dates, stock_symbols):
    """Evaluate the trained model on a set of dates"""
    env = TradingEnvironment(data_loader, evaluation_dates, stock_symbols)
    state = env.reset()
    
    total_reward = 0
    done = False
    
    # Create a DataFrame to track portfolio performance
    portfolio_history = []
    
    while not done:
        # Get actions for all stocks
        actions = {}
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            for i, symbol in enumerate(stock_symbols):
                q_values = model(state_tensor, i)
                action = q_values.argmax(dim=1).item()
                actions[symbol] = action
        
        # Take step in environment
        next_state, rewards, done, info = env.step(actions)
        
        # Calculate total reward for this step
        step_reward = sum(rewards.values())
        total_reward += step_reward
        
        # Track portfolio value
        portfolio_history.append({
            'date': info['date'],
            'time_idx': info['time_idx'],
            'portfolio_value': info['portfolio_value'],
            'balance': info['balance'],
            'positions': info['positions'].copy(),
            'actions': actions.copy()
        })
        
        # Move to next state
        state = next_state
    
    # Create DataFrame from portfolio history
    portfolio_df = pd.DataFrame(portfolio_history)
    
    # Calculate performance metrics
    initial_value = portfolio_df['portfolio_value'].iloc[0]
    final_value = portfolio_df['portfolio_value'].iloc[-1]
    returns = (final_value - initial_value) / initial_value
    
    # Calculate daily returns
    if 'date' in portfolio_df.columns:
        daily_values = portfolio_df.groupby('date')['portfolio_value'].last()
        daily_returns = daily_values.pct_change().dropna()
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        
        # Calculate drawdown
        peak = daily_values.cummax()
        drawdown = (daily_values - peak) / peak
        max_drawdown = drawdown.min()
    else:
        sharpe_ratio = None
        max_drawdown = None
    
    # Print performance metrics
    print(f"Total reward: {total_reward}")
    print(f"Returns: {returns:.2%}")
    if sharpe_ratio is not None:
        print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    if max_drawdown is not None:
        print(f"Max drawdown: {max_drawdown:.2%}")
    
    return portfolio_df

def main():
    """Main function to start the appropriate process based on rank"""
    if rank == 0:
        run_parameter_server()
    else:
        # Assign stock to this worker
        worker_idx = rank - 1  # Adjust for parameter server at rank 0
        if worker_idx < len(TRAINING_STOCKS):
            stock_symbol = TRAINING_STOCKS[worker_idx]
            run_worker(stock_symbol, worker_idx)
        else:
            print(f"No stock assigned to worker {rank}")

def train_single_process():
    """Train in a single process for testing"""
    # Create data loader
    data_loader = AzureStockDataLoader(
        AZURE_STORAGE_CONNECTION_STRING,
        CONTAINER_NAME,
        BLOB_BASE_URL,
        TRAINING_STOCKS
    )
    
    # Get training dates
    print("Getting training dates...")
    training_dates = data_loader.get_training_dates(
        start_date=datetime(2018, 1, 1).date(),
        end_date=datetime(2019, 12, 31).date()
    )
    print(f"Found {len(training_dates)} training dates")
    
    # Create model
    model = DQNetwork(
        NUM_STOCKS, MARKET_FEATURES, ACTION_SPACE
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create target network
    target_model = DQNetwork(
        NUM_STOCKS, MARKET_FEATURES, ACTION_SPACE
    ).to(device)
    target_model.load_state_dict(model.state_dict())
    
    # Create replay buffer
    replay_buffer = ExperienceReplay(BUFFER_SIZE)
    
    # Create environment
    env = TradingEnvironment(data_loader, training_dates, TRAINING_STOCKS)
    
    # Training loop
    num_episodes = 100
    epsilon = EPSILON_START
    global_step = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select actions for all stocks
            actions = {}
            for i, symbol in enumerate(TRAINING_STOCKS):
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = random.randint(0, ACTION_SPACE - 1)
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                        q_values = model(state_tensor, i)
                        action = q_values.argmax(dim=1).item()
                
                actions[symbol] = action
            
            # Take step in environment
            next_state, rewards, done, info = env.step(actions)
            episode_reward += sum(rewards.values())
            
            # Store experiences in replay buffer
            for i, symbol in enumerate(TRAINING_STOCKS):
                replay_buffer.add(
                    state, actions[symbol], next_state, rewards[symbol], done, i
                )
            
            # Train if replay buffer has enough samples
            if len(replay_buffer) > MIN_REPLAY_SIZE:
                # Sample from replay buffer
                experiences, weights, indices = replay_buffer.sample(BATCH_SIZE)
                
                # Organize batch data
                states = torch.FloatTensor(
                    np.array([exp.state for exp in experiences])
                ).to(device)
                
                act = torch.LongTensor(
                    np.array([exp.action for exp in experiences])
                ).to(device)
                
                next_states = torch.FloatTensor(
                    np.array([exp.next_state for exp in experiences])
                ).to(device)
                
                rews = torch.FloatTensor(
                    np.array([exp.reward for exp in experiences])
                ).to(device)
                
                dones = torch.FloatTensor(
                    np.array([float(exp.done) for exp in experiences])
                ).to(device)
                
                stock_ids = [exp.stock_id for exp in experiences]
                
                # Calculate current Q-values
                current_q_values = []
                for i, exp in enumerate(experiences):
                    q_values = model(states[i:i+1], exp.stock_id)
                    current_q_values.append(q_values[0, act[i]])
                
                current_q_values = torch.stack(current_q_values)
                
                # Calculate target Q-values
                target_q_values = []
                for i, exp in enumerate(experiences):
                    with torch.no_grad():
                        # Get next state Q-values from target network
                        next_q = target_model(next_states[i:i+1], exp.stock_id)
                        
                        # Double DQN: use online network to select action
                        online_next_q = model(next_states[i:i+1], exp.stock_id)
                        best_action = online_next_q.argmax(dim=1)
                        
                        # Use target network to evaluate action
                        next_q_value = next_q[0, best_action]
                        
                        # Calculate target using Bellman equation
                        target = rews[i] + GAMMA * next_q_value * (1 - dones[i])
                        target_q_values.append(target)
                
                target_q_values = torch.stack(target_q_values)
                
                # Calculate TD errors
                td_errors = target_q_values - current_q_values
                
                # Update priorities in replay buffer
                replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
                
                # Calculate loss with importance sampling weights
                weights_tensor = torch.FloatTensor(weights).to(device)
                loss = (weights_tensor * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
                
                # Perform optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update target network periodically
                if global_step % TARGET_UPDATE_FREQ == 0:
                    target_model.load_state_dict(model.state_dict())
                
                global_step += 1
            
            # Move to next state
            state = next_state
            
            # Update epsilon
            epsilon = max(
                EPSILON_END,
                EPSILON_START - (EPSILON_START - EPSILON_END) * global_step / EPSILON_DECAY
            )
        
        # Print episode results
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")
        
        # Evaluate model periodically
        if (episode + 1) % 10 == 0:
            # Get evaluation dates
            eval_dates = data_loader.get_training_dates(
                start_date=datetime(2020, 1, 1).date(),
                end_date=datetime(2020, 12, 31).date()
            )[:10]  # Use first 10 days for evaluation
            
            if eval_dates:
                print(f"Evaluating model on {len(eval_dates)} days...")
                evaluate_model(model, data_loader, eval_dates, TRAINING_STOCKS)
    
    # Save trained model
    torch.save(model.state_dict(), "trained_model.pth")
    print("Training complete. Model saved to trained_model.pth")

if __name__ == "__main__":
    if size > 1:
        main()
    else:
        train_single_process()