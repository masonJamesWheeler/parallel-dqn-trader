#!/usr/bin/env python
import argparse
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
import parallel_dqn_trading as pdt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run DQN Trading System')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'parallel'],
                        help='Training mode: single process or parallel MPI')
    parser.add_argument('--stocks', type=str, nargs='+', default=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                        help='Stocks to train on')
    parser.add_argument('--start_date', type=str, default='2018-01-01',
                        help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2019-12-31',
                        help='End date for training data (YYYY-MM-DD)')
    parser.add_argument('--eval_start', type=str, default='2020-01-01',
                        help='Start date for evaluation data (YYYY-MM-DD)')
    parser.add_argument('--eval_end', type=str, default='2020-12-31',
                        help='End date for evaluation data (YYYY-MM-DD)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of training episodes')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    
    return parser.parse_args()

def load_environment_variables():
    """Load environment variables from .cloudkeys/api_keys.env"""
    # Try to load from .cloudkeys directory
    if os.path.exists('.cloudkeys/api_keys.env'):
        load_dotenv('.cloudkeys/api_keys.env')
        logger.info("Loaded API keys from .cloudkeys/api_keys.env")
    else:
        logger.warning(".cloudkeys/api_keys.env not found. Make sure environment variables are set manually.")

def main():
    args = parse_args()
    
    # Load environment variables
    load_environment_variables()
    
    # Check if required environment variables are set
    if not os.environ.get('AZURE_STORAGE_CONNECTION_STRING'):
        logger.error("AZURE_STORAGE_CONNECTION_STRING environment variable is not set")
        logger.error("Please set this variable or add it to .cloudkeys/api_keys.env")
        return
    
    if not os.environ.get('ALPHAVANTAGE_API_KEY'):
        logger.error("ALPHAVANTAGE_API_KEY environment variable is not set")
        logger.error("Please set this variable or add it to .cloudkeys/api_keys.env")
        return
    
    # Update global constants
    pdt.TRAINING_STOCKS = args.stocks
    pdt.NUM_STOCKS = len(args.stocks)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Parse dates
    training_start = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    training_end = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    eval_start = datetime.strptime(args.eval_start, '%Y-%m-%d').date()
    eval_end = datetime.strptime(args.eval_end, '%Y-%m-%d').date()
    
    logger.info(f"Training mode: {args.mode}")
    logger.info(f"Training stocks: {args.stocks}")
    logger.info(f"Training period: {args.start_date} to {args.end_date}")
    logger.info(f"Evaluation period: {args.eval_start} to {args.eval_end}")
    
    if args.mode == 'single':
        # Create data loader for testing
        data_loader = pdt.AzureStockDataLoader(
            pdt.AZURE_STORAGE_CONNECTION_STRING,
            pdt.CONTAINER_NAME,
            pdt.BLOB_BASE_URL,
            pdt.TRAINING_STOCKS
        )
        
        # Get training dates
        logger.info("Getting training dates...")
        training_dates = data_loader.get_training_dates(
            start_date=training_start,
            end_date=training_end
        )
        logger.info(f"Found {len(training_dates)} training dates")
        
        if len(training_dates) == 0:
            logger.error("No training dates found. Exiting.")
            return
        
        # Modify number of episodes for training
        pdt.num_episodes = args.episodes
        
        # Run single-process training
        logger.info("Starting single-process training...")
        
        # Create model
        model = pdt.DQNetwork(
            pdt.NUM_STOCKS, pdt.MARKET_FEATURES, pdt.ACTION_SPACE
        ).to(pdt.device)
        
        # Create optimizer
        optimizer = pdt.optim.Adam(model.parameters(), lr=pdt.LEARNING_RATE)
        
        # Create target network
        target_model = pdt.DQNetwork(
            pdt.NUM_STOCKS, pdt.MARKET_FEATURES, pdt.ACTION_SPACE
        ).to(pdt.device)
        target_model.load_state_dict(model.state_dict())
        
        # Create replay buffer
        replay_buffer = pdt.ExperienceReplay(pdt.BUFFER_SIZE)
        
        # Create environment
        env = pdt.TradingEnvironment(data_loader, training_dates, pdt.TRAINING_STOCKS)
        
        # Training loop
        num_episodes = args.episodes
        epsilon = pdt.EPSILON_START
        global_step = 0
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select actions for all stocks
                actions = {}
                for i, symbol in enumerate(pdt.TRAINING_STOCKS):
                    # Epsilon-greedy action selection
                    if pdt.random.random() < epsilon:
                        action = pdt.random.randint(0, pdt.ACTION_SPACE - 1)
                    else:
                        with pdt.torch.no_grad():
                            state_tensor = pdt.torch.FloatTensor(state).unsqueeze(0).to(pdt.device)
                            q_values = model(state_tensor, i)
                            action = q_values.argmax(dim=1).item()
                    
                    actions[symbol] = action
                
                # Take step in environment
                next_state, rewards, done, info = env.step(actions)
                episode_reward += sum(rewards.values())
                
                # Store experiences in replay buffer
                for i, symbol in enumerate(pdt.TRAINING_STOCKS):
                    replay_buffer.add(
                        state, actions[symbol], next_state, rewards[symbol], done, i
                    )
                
                # Train if replay buffer has enough samples
                if len(replay_buffer) > pdt.MIN_REPLAY_SIZE:
                    # Sample from replay buffer
                    experiences, weights, indices = replay_buffer.sample(pdt.BATCH_SIZE)
                    
                    # Organize batch data
                    states = pdt.torch.FloatTensor(
                        pdt.np.array([exp.state for exp in experiences])
                    ).to(pdt.device)
                    
                    act = pdt.torch.LongTensor(
                        pdt.np.array([exp.action for exp in experiences])
                    ).to(pdt.device)
                    
                    next_states = pdt.torch.FloatTensor(
                        pdt.np.array([exp.next_state for exp in experiences])
                    ).to(pdt.device)
                    
                    rews = pdt.torch.FloatTensor(
                        pdt.np.array([exp.reward for exp in experiences])
                    ).to(pdt.device)
                    
                    dones = pdt.torch.FloatTensor(
                        pdt.np.array([float(exp.done) for exp in experiences])
                    ).to(pdt.device)
                    
                    stock_ids = [exp.stock_id for exp in experiences]
                    
                    # Calculate current Q-values
                    current_q_values = []
                    for i, exp in enumerate(experiences):
                        q_values = model(states[i:i+1], exp.stock_id)
                        current_q_values.append(q_values[0, act[i]])
                    
                    current_q_values = pdt.torch.stack(current_q_values)
                    
                    # Calculate target Q-values
                    target_q_values = []
                    for i, exp in enumerate(experiences):
                        with pdt.torch.no_grad():
                            # Get next state Q-values from target network
                            next_q = target_model(next_states[i:i+1], exp.stock_id)
                            
                            # Double DQN: use online network to select action
                            online_next_q = model(next_states[i:i+1], exp.stock_id)
                            best_action = online_next_q.argmax(dim=1)
                            
                            # Use target network to evaluate action
                            next_q_value = next_q[0, best_action]
                            
                            # Calculate target using Bellman equation
                            target = rews[i] + pdt.GAMMA * next_q_value * (1 - dones[i])
                            target_q_values.append(target)
                    
                    target_q_values = pdt.torch.stack(target_q_values)
                    
                    # Calculate TD errors
                    td_errors = target_q_values - current_q_values
                    
                    # Update priorities in replay buffer
                    replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
                    
                    # Calculate loss with importance sampling weights
                    weights_tensor = pdt.torch.FloatTensor(weights).to(pdt.device)
                    loss = (weights_tensor * pdt.F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
                    
                    # Perform optimization step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update target network periodically
                    if global_step % pdt.TARGET_UPDATE_FREQ == 0:
                        target_model.load_state_dict(model.state_dict())
                    
                    global_step += 1
                
                # Move to next state
                state = next_state
                
                # Update epsilon
                epsilon = max(
                    pdt.EPSILON_END,
                    pdt.EPSILON_START - (pdt.EPSILON_START - pdt.EPSILON_END) * global_step / pdt.EPSILON_DECAY
                )
            
            # Print episode results
            logger.info(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")
            
            # Evaluate model periodically
            if (episode + 1) % 10 == 0 or episode == num_episodes - 1:
                # Get evaluation dates
                eval_dates = data_loader.get_training_dates(
                    start_date=eval_start,
                    end_date=eval_end
                )[:10]  # Use first 10 days for evaluation
                
                if eval_dates:
                    logger.info(f"Evaluating model on {len(eval_dates)} days...")
                    portfolio_df = pdt.evaluate_model(model, data_loader, eval_dates, pdt.TRAINING_STOCKS)
                    
                    # Save portfolio performance
                    if portfolio_df is not None:
                        portfolio_df.to_csv(f"{args.output}/portfolio_ep{episode+1}.csv")
        
        # Save trained model
        model_path = f"{args.output}/trained_model.pth"
        pdt.torch.save(model.state_dict(), model_path)
        logger.info(f"Training complete. Model saved to {model_path}")
    
    else:  # Parallel mode
        # Run parallel training using MPI
        pdt.main()

if __name__ == "__main__":
    main()