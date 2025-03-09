#\!/bin/bash

# Load environment variables
if [ -f .cloudkeys/api_keys.env ]; then
    export $(grep -v '^#' .cloudkeys/api_keys.env | xargs)
    echo "Loaded environment variables from .cloudkeys/api_keys.env"
else
    echo "Warning: .cloudkeys/api_keys.env not found"
fi

# Run the specified script with all arguments passed through
python "$@"
