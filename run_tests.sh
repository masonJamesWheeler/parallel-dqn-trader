#!/bin/bash

echo "Running DQN Trading System Tests"

# Run standard tests
echo "Running unit tests..."
pytest test_dqn_trading.py -v

# Run integration tests if requested
if [ "$1" == "--with-integration" ]; then
    echo "Running integration tests..."
    pytest test_dqn_trading.py -v -m integration
fi

# Run tests with coverage if requested
if [ "$1" == "--with-coverage" ]; then
    echo "Running tests with coverage..."
    pytest test_dqn_trading.py --cov=parallel_dqn_trading
    
    # Generate coverage report
    coverage report -m
    
    # Generate HTML report if available
    if command -v coverage >/dev/null 2>&1; then
        coverage html
        echo "Coverage report generated in htmlcov/"
    fi
fi

echo "Tests completed"