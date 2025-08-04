#!/bin/bash

# Exit on any error
set -e

echo "Setting up Neural Memory Mapper development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p assets/temp

echo "Setup complete! Activate the virtual environment with: source venv/bin/activate"
