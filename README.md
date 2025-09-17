# TrafCast

A traffic speed prediction system for Los Angeles using LSTM neural networks.

## Overview

TrafCast predicts real-time traffic speeds across major Los Angeles highways and roads using deep learning. The system uses an LSTM (Long Short-Term Memory) model trained on historical traffic data to forecast speed patterns.

## Model Details

- **Architecture**: LSTM neural network with 2,191,617 parameters
- **Training Data**: 32+ million data points from LA traffic sensors
- **Performance**: Best validation loss of 6.6276, test loss of 6.0229
- **Features**: Weather data, road characteristics, time patterns, and historical speeds

## Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Install Git LFS:**
   ```bash
   # On macOS with Homebrew
   brew install git-lfs
   
   # On Ubuntu/Debian
   sudo apt install git-lfs
   
   # On Windows (with Git for Windows)
   git lfs install
   ```

2. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd TrafCast
   git lfs

3. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

The app will be available at `http://localhost:8501`

## Usage

1. Select roads from the available LA highways
2. Choose a date and time for prediction
3. Select visualization mode (Predicted, Real, or Comparison)
4. Click "Apply Prediction" to generate traffic speed maps

## Data

The model was trained on CSV files containing traffic sensor data from major LA roads including I-405, US-101, I-5, and state highways.
