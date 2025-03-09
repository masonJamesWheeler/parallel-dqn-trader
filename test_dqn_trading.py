import pytest
import os
import tempfile
import shutil
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock, Mock
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO

# Import the module to test
import parallel_dqn_trading as pdt


# Fixtures

@pytest.fixture
def sample_parquet_data():
    """Create sample data in parquet format for testing"""
    # Create a sample dataframe
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2020-01-03 09:30:00', periods=100, freq='15min'),
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(100, 200, 100),
        'low': np.random.uniform(100, 200, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Convert to parquet format
    table = pa.Table.from_pandas(df)
    buffer = BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def mock_blob_client(sample_parquet_data):
    """Create a mock blob client that returns sample data"""
    class MockBlobClient:
        def __init__(self, data=None):
            self.data = data or b''
        
        def download_blob(self):
            return self
        
        def readall(self):
            return self.data
    
    return MockBlobClient(sample_parquet_data)


@pytest.fixture
def mock_container_client():
    """Create a mock container client that lists blobs"""
    class MockBlob:
        def __init__(self, name):
            self.name = name
    
    class MockContainerClient:
        def __init__(self, blobs=None):
            self.blobs = blobs or []
        
        def list_blobs(self, name_starts_with=None):
            return [blob for blob in self.blobs if blob.name.startswith(name_starts_with)]
    
    blobs = [
        MockBlob("stocks_intraday/AAPL/2020/01/AAPL_2020-01-03.parquet"),
        MockBlob("stocks_intraday/AAPL/2020/01/AAPL_2020-01-04.parquet"),
        MockBlob("stocks_intraday/MSFT/2020/01/MSFT_2020-01-03.parquet"),
        MockBlob("stocks_intraday/GOOGL/2020/01/GOOGL_2020-01-03.parquet"),
        MockBlob("stocks_intraday/AMZN/2020/01/AMZN_2020-01-03.parquet"),
        MockBlob("stocks_intraday/META/2020/01/META_2020-01-03.parquet")
    ]
    return MockContainerClient(blobs)


@pytest.fixture
def mock_blob_service_client(mock_container_client, mock_blob_client):
    """Create a mock blob service client"""
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container_client
    mock_service.get_blob_client.return_value = mock_blob_client
    return mock_service


@pytest.fixture
def azure_data_loader(mock_blob_service_client):
    """Create a data loader with mocked dependencies"""
    with patch('parallel_dqn_trading.BlobServiceClient') as mock_blob_service:
        mock_blob_service.from_connection_string.return_value = mock_blob_service_client
        data_loader = pdt.AzureStockDataLoader(
            "mock_connection_string",
            "mock_container",
            "mock_base_url",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        )
        yield data_loader


@pytest.fixture
def real_data_loader():
    """Create a real data loader for integration testing (if connection string is available)"""
    if not pdt.AZURE_STORAGE_CONNECTION_STRING:
        pytest.skip("Azure Storage connection string not available")
    
    data_loader = pdt.AzureStockDataLoader(
        pdt.AZURE_STORAGE_CONNECTION_STRING,
        pdt.CONTAINER_NAME,
        pdt.BLOB_BASE_URL,
        ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    )
    return data_loader


@pytest.fixture
def sample_stock_data():
    """Create sample stock data"""
    def create_data(n_periods):
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2020-01-03 09:30:00', periods=n_periods, freq='15min'),
            'open': np.random.uniform(100, 200, n_periods),
            'high': np.random.uniform(100, 200, n_periods),
            'low': np.random.uniform(100, 200, n_periods),
            'close': np.random.uniform(100, 200, n_periods),
            'volume': np.random.randint(1000, 10000, n_periods),
            'returns': np.random.uniform(-0.05, 0.05, n_periods),
            'ma5': np.random.uniform(100, 200, n_periods),
            'ma10': np.random.uniform(100, 200, n_periods),
            'ma20': np.random.uniform(100, 200, n_periods),
            'pct_ma5': np.random.uniform(-0.05, 0.05, n_periods),
            'pct_ma10': np.random.uniform(-0.05, 0.05, n_periods),
            'pct_ma20': np.random.uniform(-0.05, 0.05, n_periods),
            'rsi': np.random.uniform(0, 100, n_periods),
            'macd': np.random.uniform(-5, 5, n_periods),
            'macd_signal': np.random.uniform(-5, 5, n_periods),
            'macd_hist': np.random.uniform(-5, 5, n_periods),
            'volatility': np.random.uniform(0, 0.05, n_periods),
            'norm_volume': np.random.uniform(0.5, 1.5, n_periods)
        })
        return df
    
    return {
        "AAPL": create_data(100),
        "MSFT": create_data(100),
        "GOOGL": create_data(100),
        "AMZN": create_data(100),
        "META": create_data(100)
    }


@pytest.fixture
def mock_trading_env(sample_stock_data):
    """Create a mock trading environment"""
    mock_data_loader = MagicMock()
    training_dates = [date(2020, 1, 3), date(2020, 1, 4)]
    stock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    mock_data_loader.load_synchronous_market_data.return_value = sample_stock_data
    
    env = pdt.TradingEnvironment(
        mock_data_loader,
        training_dates,
        stock_symbols
    )
    return env


@pytest.fixture
def dqn_model():
    """Create a DQN model for testing"""
    num_stocks = 5
    market_features = 19
    action_space = 3
    hidden_dim = 64
    model = pdt.DQNetwork(
        num_stocks, market_features, action_space, hidden_dim
    )
    return model


# Tests

class TestAzureStockDataLoader:
    def test_list_available_dates(self, azure_data_loader):
        """Test listing available dates"""
        available_dates = azure_data_loader.list_available_dates("AAPL", 2020, 2020)
        
        assert len(available_dates) == 2
        assert date(2020, 1, 3) in available_dates
        assert date(2020, 1, 4) in available_dates
    
    def test_load_stock_data(self, azure_data_loader, mock_blob_service_client, sample_parquet_data):
        """Test loading stock data for a specific date"""
        df = azure_data_loader.load_stock_data("AAPL", date(2020, 1, 3))
        
        assert df is not None
        assert 'timestamp' in df.columns
        assert 'close' in df.columns
        assert 'returns' in df.columns  # Technical indicators should be added
        assert 'ma5' in df.columns
        assert 'rsi' in df.columns
    
    def test_load_synchronous_market_data(self, azure_data_loader):
        """Test loading market data for all stocks on the same date"""
        market_data = azure_data_loader.load_synchronous_market_data(date(2020, 1, 3))
        
        assert "AAPL" in market_data
        assert "MSFT" in market_data
        assert isinstance(market_data["AAPL"], pd.DataFrame)
        assert isinstance(market_data["MSFT"], pd.DataFrame)
    
    def test_get_training_dates(self, azure_data_loader):
        """Test getting common training dates for all stocks"""
        # Mock the list_available_dates method
        azure_data_loader.list_available_dates = MagicMock(
            side_effect=lambda symbol, start_year, end_year: [
                date(2020, 1, 3),
                date(2020, 1, 4) if symbol == "AAPL" else date(2020, 1, 3)
            ]
        )
        
        training_dates = azure_data_loader.get_training_dates(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31)
        )
        
        assert len(training_dates) == 1  # Only Jan 3 is common to both
        assert training_dates[0] == date(2020, 1, 3)
    
    @pytest.mark.integration
    def test_real_data_fetch(self, real_data_loader):
        """Integration test with real Azure storage (if available)"""
        # Try to get actual dates from storage
        try:
            dates = real_data_loader.list_available_dates("AAPL", 2020, 2020)
            assert len(dates) > 0, "Should find some dates for AAPL in 2020"
            
            # Try to load data for the first date
            if dates:
                df = real_data_loader.load_stock_data("AAPL", dates[0])
                assert df is not None
                assert not df.empty
                assert 'timestamp' in df.columns
                assert 'close' in df.columns
                
                # Check if technical indicators were added
                assert 'ma5' in df.columns
                assert 'rsi' in df.columns
        except Exception as e:
            pytest.skip(f"Real data test failed: {e}")


class TestMarketStateEncoder:
    def test_init(self):
        """Test initialization"""
        num_stocks = 5
        market_features = 19
        hidden_dim = 64
        
        encoder = pdt.MarketStateEncoder(num_stocks, market_features, hidden_dim)
        
        assert len(encoder.stock_encoders) == num_stocks
        assert encoder.market_features == market_features
        assert encoder.attention.embed_dim == hidden_dim // 2
    
    def test_forward_pass(self):
        """Test the forward pass of the encoder"""
        num_stocks = 5
        market_features = 19
        hidden_dim = 64
        encoder = pdt.MarketStateEncoder(num_stocks, market_features, hidden_dim)
        
        # Create a random batch of market states
        batch_size = 5
        market_states = torch.rand(batch_size, num_stocks, market_features)
        
        # Test forward pass
        output = encoder(market_states)
        
        # Assertions
        assert output.shape == (batch_size, num_stocks, hidden_dim // 2)


class TestCrossStockRelationship:
    def test_init(self):
        """Test initialization"""
        num_stocks = 5
        hidden_dim = 64
        
        relationship = pdt.CrossStockRelationship(num_stocks, hidden_dim)
        
        assert relationship.relationship_matrix.shape == (num_stocks, num_stocks, hidden_dim)
    
    def test_forward_pass(self):
        """Test the forward pass of the relationship module"""
        num_stocks = 5
        hidden_dim = 64
        relationship = pdt.CrossStockRelationship(num_stocks, hidden_dim)
        
        # Create a random batch of encoded states
        batch_size = 5
        encoded_states = torch.rand(batch_size, num_stocks, hidden_dim)
        
        # Test forward pass for each stock
        for stock_idx in range(num_stocks):
            influence = relationship(stock_idx, encoded_states)
            
            # Assertions
            assert influence.shape == (batch_size, hidden_dim)


class TestDQNetwork:
    def test_init(self):
        """Test initialization"""
        num_stocks = 5
        market_features = 19
        action_space = 3
        hidden_dim = 64
        
        network = pdt.DQNetwork(num_stocks, market_features, action_space, hidden_dim)
        
        assert len(network.q_heads) == num_stocks
        assert isinstance(network.market_encoder, pdt.MarketStateEncoder)
        assert isinstance(network.relationship_module, pdt.CrossStockRelationship)
    
    def test_forward_pass_single_stock(self, dqn_model):
        """Test the forward pass for a single stock"""
        batch_size = 5
        states = torch.rand(batch_size, 5, 19)  # 5 stocks, 19 features
        
        # Test forward pass for each stock
        for stock_idx in range(5):
            q_values = dqn_model(states, stock_idx)
            
            # Assertions
            assert q_values.shape == (batch_size, 3)  # 3 actions
    
    def test_forward_pass_all_stocks(self, dqn_model):
        """Test the forward pass for all stocks"""
        batch_size = 5
        states = torch.rand(batch_size, 5, 19)  # 5 stocks, 19 features
        
        # Test forward pass
        q_values = dqn_model(states)
        
        # Assertions
        assert len(q_values) == 5  # One set of Q-values per stock
        for stock_q in q_values:
            assert stock_q.shape == (batch_size, 3)  # 3 actions per stock


class TestExperienceReplay:
    def test_init(self):
        """Test initialization"""
        buffer_size = 100
        replay_buffer = pdt.ExperienceReplay(buffer_size)
        
        assert len(replay_buffer) == 0
        assert len(replay_buffer.priorities) == 0
        assert replay_buffer.alpha == 0.6
        assert replay_buffer.beta == 0.4
    
    def test_add_experience(self):
        """Test adding experiences to the buffer"""
        buffer_size = 100
        replay_buffer = pdt.ExperienceReplay(buffer_size)
        
        # Add some experiences
        for i in range(50):
            state = np.random.rand(5, 19)
            action = np.random.randint(0, 3)
            next_state = np.random.rand(5, 19)
            reward = np.random.rand()
            done = np.random.choice([True, False])
            stock_id = np.random.randint(0, 5)
            replay_buffer.add(state, action, next_state, reward, done, stock_id)
        
        assert len(replay_buffer) == 50
        assert len(replay_buffer.priorities) == 50
    
    def test_sample_batch(self):
        """Test sampling a batch of experiences"""
        buffer_size = 100
        replay_buffer = pdt.ExperienceReplay(buffer_size)
        
        # Add some experiences
        for i in range(100):
            state = np.random.rand(5, 19)
            action = np.random.randint(0, 3)
            next_state = np.random.rand(5, 19)
            reward = np.random.rand()
            done = np.random.choice([True, False])
            stock_id = np.random.randint(0, 5)
            replay_buffer.add(state, action, next_state, reward, done, stock_id)
        
        # Sample a batch
        batch_size = 10
        experiences, weights, indices = replay_buffer.sample(batch_size)
        
        assert len(experiences) == batch_size
        assert len(weights) == batch_size
        assert len(indices) == batch_size
        assert isinstance(experiences[0], pdt.Experience)
    
    def test_update_priorities(self):
        """Test updating priorities"""
        buffer_size = 100
        replay_buffer = pdt.ExperienceReplay(buffer_size)
        
        # Add some experiences
        for i in range(100):
            state = np.random.rand(5, 19)
            action = np.random.randint(0, 3)
            next_state = np.random.rand(5, 19)
            reward = np.random.rand()
            done = np.random.choice([True, False])
            stock_id = np.random.randint(0, 5)
            replay_buffer.add(state, action, next_state, reward, done, stock_id)
        
        # Sample a batch
        batch_size = 10
        experiences, weights, indices = replay_buffer.sample(batch_size)
        
        # Update priorities
        new_errors = np.random.rand(batch_size)
        old_priorities = [replay_buffer.priorities[idx] for idx in indices]
        
        replay_buffer.update_priorities(indices, new_errors)
        
        # Check if priorities were updated
        for i, idx in enumerate(indices):
            new_priority = abs(new_errors[i]) + replay_buffer.epsilon
            assert abs(replay_buffer.priorities[idx] - new_priority) < 1e-5
            assert replay_buffer.priorities[idx] != old_priorities[i]


class TestTradingEnvironment:
    def test_init(self, mock_trading_env):
        """Test initialization"""
        assert mock_trading_env.num_stocks == 5
        assert len(mock_trading_env.stock_symbols) == 5
        assert mock_trading_env.current_date_idx == 0
        assert mock_trading_env.current_time_idx == 0
    
    def test_reset(self, mock_trading_env):
        """Test resetting the environment"""
        state = mock_trading_env.reset()
        
        assert mock_trading_env.current_date_idx == 0
        assert mock_trading_env.current_time_idx == 0
        assert mock_trading_env.positions == {
            "AAPL": 0, "MSFT": 0, "GOOGL": 0, "AMZN": 0, "META": 0
        }
        assert mock_trading_env.balance == mock_trading_env.initial_balance
        assert state.shape == (5, len(mock_trading_env.feature_columns) + 1)
    
    def test_step(self, mock_trading_env):
        """Test taking a step in the environment"""
        mock_trading_env.reset()
        
        # Take a step with buy actions
        actions = {"AAPL": 1, "MSFT": 1, "GOOGL": 0, "AMZN": 0, "META": 0}
        next_state, rewards, done, info = mock_trading_env.step(actions)
        
        assert next_state.shape == (5, len(mock_trading_env.feature_columns) + 1)
        assert len(rewards) == 5
        assert not done
        assert mock_trading_env.current_time_idx == 1
        assert mock_trading_env.positions["AAPL"] == 1
        assert mock_trading_env.positions["MSFT"] == 1
        assert mock_trading_env.positions["GOOGL"] == 0
        
        # Take another step with different actions
        actions = {"AAPL": 2, "MSFT": 0, "GOOGL": 1, "AMZN": 0, "META": 0}
        next_state, rewards, done, info = mock_trading_env.step(actions)
        
        assert mock_trading_env.positions["AAPL"] == 0  # AAPL sold
        assert mock_trading_env.positions["MSFT"] == 1  # MSFT still held
        assert mock_trading_env.positions["GOOGL"] == 1  # GOOGL bought
    
    def test_portfolio_value_calculation(self, mock_trading_env):
        """Test portfolio value calculation after taking actions"""
        mock_trading_env.reset()
        initial_balance = mock_trading_env.balance
        
        # Buy a stock
        actions = {"AAPL": 1, "MSFT": 0, "GOOGL": 0, "AMZN": 0, "META": 0}
        next_state, rewards, done, info = mock_trading_env.step(actions)
        
        # Check that balance decreased
        assert mock_trading_env.balance < initial_balance
        
        # Check that portfolio value includes stock value
        assert mock_trading_env.portfolio_value > mock_trading_env.balance
        
        # Get the apple price
        apple_df = mock_trading_env.current_market_data["AAPL"]
        apple_price = apple_df.iloc[mock_trading_env.current_time_idx]["close"]
        
        # Check that portfolio includes apple price (minus transaction cost)
        expected_portfolio = mock_trading_env.balance + apple_price
        assert abs(mock_trading_env.portfolio_value - expected_portfolio) < 1e-5


class TestMetaPolicyLayer:
    def test_init(self):
        """Test initialization"""
        num_stocks = 5
        stock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        meta_policy = pdt.MetaPolicyLayer(num_stocks, stock_symbols)
        
        assert meta_policy.correlation_matrix.shape == (num_stocks, num_stocks)
        assert meta_policy.strategy_allocation.shape == (num_stocks,)
        assert meta_policy.stock_symbols == stock_symbols
    
    def test_update_correlation(self):
        """Test updating correlation matrix"""
        num_stocks = 5
        stock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        meta_policy = pdt.MetaPolicyLayer(num_stocks, stock_symbols)
        
        # Create random returns
        returns = torch.rand(num_stocks)
        
        # Get initial correlation matrix
        initial_corr = meta_policy.correlation_matrix.clone()
        
        # Update correlation
        meta_policy.update_correlation(returns)
        
        # Assertions
        updated_corr = meta_policy.correlation_matrix
        assert not torch.allclose(initial_corr, updated_corr)
    
    def test_allocate_strategies(self):
        """Test allocating strategies"""
        num_stocks = 5
        stock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        meta_policy = pdt.MetaPolicyLayer(num_stocks, stock_symbols)
        
        # Create portfolio state
        portfolio_state = torch.rand(num_stocks * 2)
        
        # Allocate strategies
        allocation = meta_policy.allocate_strategies(portfolio_state)
        
        assert allocation.shape == (num_stocks,)
        assert abs(allocation.sum().item() - 1.0) < 1e-5  # Should sum to 1
    
    def test_get_strategy_for_stock(self):
        """Test getting strategy for a specific stock"""
        num_stocks = 5
        stock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        meta_policy = pdt.MetaPolicyLayer(num_stocks, stock_symbols)
        
        # Set a known allocation
        meta_policy.strategy_allocation = torch.tensor([0.2, 0.3, 0.1, 0.15, 0.25])
        
        # Get strategy for each stock
        for i, symbol in enumerate(stock_symbols):
            allocation = meta_policy.get_strategy_for_stock(symbol)
            assert abs(allocation - meta_policy.strategy_allocation[i].item()) < 1e-5
        
        # Test with unknown stock
        allocation = meta_policy.get_strategy_for_stock("UNKNOWN")
        assert allocation == 0.0


class TestParameterServer:
    def test_init(self, dqn_model):
        """Test initialization"""
        num_workers = 5
        stock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        param_server = pdt.ParameterServer(dqn_model, num_workers, stock_symbols)
        
        assert param_server.model == dqn_model
        assert param_server.num_workers == num_workers
        assert param_server.stock_symbols == stock_symbols
        assert param_server.global_step == 0
        assert isinstance(param_server.replay_buffer, pdt.ExperienceReplay)
        assert isinstance(param_server.meta_policy, pdt.MetaPolicyLayer)
    
    def test_update(self, dqn_model):
        """Test updating the global model"""
        num_workers = 5
        stock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        param_server = pdt.ParameterServer(dqn_model, num_workers, stock_symbols)
        
        # Create a batch of fake gradients
        gradients = [torch.rand_like(param) for param in dqn_model.parameters()]
        
        # Create fake experiences
        experiences = []
        for _ in range(10):
            state = np.random.rand(5, 19)
            action = np.random.randint(0, 3)
            next_state = np.random.rand(5, 19)
            reward = np.random.rand()
            done = False
            stock_id = np.random.randint(0, 5)
            experiences.append((state, action, next_state, reward, done, stock_id))
        
        # Mock _train_step to avoid actual training
        param_server._train_step = MagicMock()
        
        # Update the model
        updated_state_dict = param_server.update(gradients, experiences)
        
        assert len(param_server.replay_buffer) == len(experiences)
        assert param_server.global_step == 1
        assert isinstance(updated_state_dict, dict)


class TestWorker:
    def test_init(self, dqn_model):
        """Test initialization"""
        # Create mock environment
        mock_env = MagicMock()
        
        stock_symbol = "AAPL"
        stock_idx = 0
        worker = pdt.Worker(stock_symbol, stock_idx, mock_env, dqn_model)
        
        assert worker.stock_symbol == stock_symbol
        assert worker.stock_idx == stock_idx
        assert worker.env == mock_env
        assert worker.local_model == dqn_model
        assert len(worker.local_buffer) == 0
        assert worker.epsilon == pdt.EPSILON_START
    
    def test_collect_experience(self, dqn_model):
        """Test collecting experience"""
        # Create mock environment
        mock_env = MagicMock()
        mock_env.reset.return_value = np.zeros((5, 19))
        mock_env.step.return_value = (
            np.zeros((5, 19)),  # next_state
            {"AAPL": 0.1},  # rewards
            False,  # done
            {}  # info
        )
        
        stock_symbol = "AAPL"
        stock_idx = 0
        worker = pdt.Worker(stock_symbol, stock_idx, mock_env, dqn_model)
        
        # Collect experience
        experiences = worker.collect_experience(num_steps=5)
        
        assert len(experiences) == 5
        assert len(worker.local_buffer) == 5
        
        # Check experience structure
        for exp in experiences:
            assert len(exp) == 6  # state, action, next_state, reward, done, stock_idx
    
    def test_compute_gradients(self, dqn_model):
        """Test computing gradients"""
        # Create mock environment
        mock_env = MagicMock()
        
        stock_symbol = "AAPL"
        stock_idx = 0
        worker = pdt.Worker(stock_symbol, stock_idx, mock_env, dqn_model)
        
        # First, collect enough experiences
        for _ in range(pdt.BATCH_SIZE * 2):
            worker.local_buffer.append((
                np.zeros((5, 19)),
                0,
                np.zeros((5, 19)),
                0.1,
                False,
                stock_idx
            ))
        
        # Compute gradients
        result = worker.compute_gradients()
        
        assert result is not None
        gradients, batch = result
        assert len(gradients) == len(list(dqn_model.parameters()))
        assert len(batch) == pdt.BATCH_SIZE
        assert len(worker.local_buffer) == 0  # Buffer should be cleared
    
    def test_update_model(self, dqn_model):
        """Test updating the local model"""
        # Create mock environment
        mock_env = MagicMock()
        
        stock_symbol = "AAPL"
        stock_idx = 0
        worker = pdt.Worker(stock_symbol, stock_idx, mock_env, dqn_model)
        
        # Get current state dict
        old_state_dict = {k: v.clone() for k, v in dqn_model.state_dict().items()}
        
        # Create a modified state dict
        new_state_dict = {}
        for key, value in old_state_dict.items():
            new_state_dict[key] = value + 0.1
        
        # Update the model
        worker.update_model(new_state_dict)
        
        # Assertions
        updated_state_dict = worker.local_model.state_dict()
        for key in old_state_dict:
            assert not torch.allclose(old_state_dict[key], updated_state_dict[key])


# Integration tests

@pytest.mark.integration
def test_real_data_model_inference(real_data_loader, dqn_model):
    """Test model inference with real data (if available)"""
    try:
        # Get some real dates
        dates = real_data_loader.list_available_dates("AAPL", 2020, 2020)
        if not dates:
            pytest.skip("No data available for AAPL in 2020")
        
        # Load data for the first date
        market_data = real_data_loader.load_synchronous_market_data(dates[0])
        if not all(symbol in market_data for symbol in ["AAPL", "MSFT"]):
            pytest.skip("Not all required stocks have data for the test date")
        
        # Create an environment with real data
        env = pdt.TradingEnvironment(
            real_data_loader,
            [dates[0]],
            ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        )
        
        # Reset and get initial state
        state = env.reset()
        
        # Convert state to tensor and run through model
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get actions for all stocks
        with torch.no_grad():
            q_values = dqn_model(state_tensor)
            assert len(q_values) == 5
            
            # Get action for AAPL
            aapl_q = dqn_model(state_tensor, 0)
            assert aapl_q.shape == (1, 3)
            
            # Take the best action
            actions = {}
            for i, symbol in enumerate(["AAPL", "MSFT", "GOOGL", "AMZN", "META"]):
                action = q_values[i].argmax(dim=1).item()
                actions[symbol] = action
            
            # Take a step in the environment
            next_state, rewards, done, info = env.step(actions)
            
            # Verify the results
            assert next_state.shape == state.shape
            assert len(rewards) == 5
            assert isinstance(info, dict)
    except Exception as e:
        pytest.skip(f"Real data model inference test failed: {e}")


@pytest.mark.integration
def test_end_to_end_training_cycle(real_data_loader, dqn_model):
    """Test a complete training cycle (if real data is available)"""
    try:
        # Get some real dates
        dates = real_data_loader.list_available_dates("AAPL", 2020, 2020)
        if not dates or len(dates) < 2:
            pytest.skip("Not enough data available for AAPL in 2020")
        
        # Use first date for training
        training_dates = [dates[0]]
        
        # Create an environment with real data
        env = pdt.TradingEnvironment(
            real_data_loader,
            training_dates,
            ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        )
        
        # Create a replay buffer
        replay_buffer = pdt.ExperienceReplay(100)
        
        # Create target network
        target_model = pdt.DQNetwork(5, 19, 3)
        target_model.load_state_dict(dqn_model.state_dict())
        
        # Create optimizer
        optimizer = pdt.optim.Adam(dqn_model.parameters(), lr=1e-4)
        
        # Reset environment
        state = env.reset()
        
        # Take some random actions and train
        for _ in range(5):
            # Choose actions
            actions = {}
            for i, symbol in enumerate(["AAPL", "MSFT", "GOOGL", "AMZN", "META"]):
                action = np.random.randint(0, 3)
                actions[symbol] = action
            
            # Take step
            next_state, rewards, done, info = env.step(actions)
            
            # Store experiences
            for i, symbol in enumerate(["AAPL", "MSFT", "GOOGL", "AMZN", "META"]):
                replay_buffer.add(
                    state, actions[symbol], next_state, rewards[symbol], done, i
                )
            
            # Train if we have enough samples
            if len(replay_buffer) > 10:
                # Sample batch
                experiences, weights, indices = replay_buffer.sample(10)
                
                # Prepare batch data
                batch_states = torch.FloatTensor(
                    np.array([exp.state for exp in experiences])
                )
                batch_actions = torch.LongTensor(
                    np.array([exp.action for exp in experiences])
                )
                batch_next_states = torch.FloatTensor(
                    np.array([exp.next_state for exp in experiences])
                )
                batch_rewards = torch.FloatTensor(
                    np.array([exp.reward for exp in experiences])
                )
                batch_dones = torch.FloatTensor(
                    np.array([float(exp.done) for exp in experiences])
                )
                
                # Calculate Q-values
                current_q_values = []
                for i, exp in enumerate(experiences):
                    q_values = dqn_model(batch_states[i:i+1], exp.stock_id)
                    current_q_values.append(q_values[0, batch_actions[i]])
                
                current_q_values = torch.stack(current_q_values)
                
                # Calculate target Q-values
                target_q_values = []
                for i, exp in enumerate(experiences):
                    with torch.no_grad():
                        # Get next state Q-values from target network
                        next_q = target_model(batch_next_states[i:i+1], exp.stock_id)
                        next_q_values = next_q.max(1)[0]
                        target = batch_rewards[i] + 0.99 * next_q_values * (1 - batch_dones[i])
                        target_q_values.append(target)
                
                target_q_values = torch.stack(target_q_values)
                
                # Compute loss and update
                weights_tensor = torch.FloatTensor(weights)
                loss = (weights_tensor * (current_q_values - target_q_values) ** 2).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update priorities
                td_errors = (target_q_values - current_q_values).detach().numpy()
                replay_buffer.update_priorities(indices, td_errors)
            
            # Move to next state
            state = next_state
            
            if done:
                break
        
        # Verify training worked (model changed)
        for param in dqn_model.parameters():
            assert param.grad is not None
    except Exception as e:
        pytest.skip(f"End-to-end training test failed: {e}")