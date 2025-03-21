#!/bin/bash
# This script sets up a Python 3.12.3 virtual environment in the current user's home directory,
# and installs all necessary packages to run image_regression.py.

# You will need to run "sudo chmod +x ./setup-venv.sh" prior to use.  Then, run script with ./setup-venv.sh
# OR IF NOT IN DIRECTORY: replace ./setup-venv.sh with /path/to/setup-venv.sh

# R. Tillman 3.21.25.  Contact me if there are any problems!

# Exit immediately if a command fails.
set -e

# Automatically detect the current user's home directory.
HOME_DIR="$HOME"
echo "Current user's home directory detected as: $HOME_DIR"

# Update the package list.
echo "Updating package list..."
sudo apt-get update

# Install Python 3.12, its venv module, and development tools.
echo "Installing Python 3.12 and required system packages..."
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential

# Change to the user's home directory.
cd "$HOME_DIR"

# Create a virtual environment named 'image_regression_env' in the user's home directory.
echo "Creating a Python 3.12.3 virtual environment in $HOME_DIR/image_regression_env..."
python3 -m venv image_regression_env

# Activate the virtual environment.
echo "Activating the virtual environment..."
source image_regression_env/bin/activate

# Upgrade pip to the latest version.
echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing required pip packages..."
pip install "tensorflow[and-cuda]" opencv-python matplotlib pyserial numpy pandas

echo "Setup complete! The virtual environment 'image_regression_env' is now ready."
echo "To activate the virtual environment in the future, run:"
echo "source $HOME_DIR/image_regression_env/bin/activate"
