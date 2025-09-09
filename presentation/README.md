# TrafCast: Traffic Flow Prediction System

## Project Overview

TrafCast is a deep learning-based traffic flow prediction system that uses LSTM neural networks to forecast traffic speeds on Los Angeles highways. The system processes real-time traffic sensor data and provides accurate speed predictions for better traffic management and route planning.

## Presentation Structure

This presentation folder contains comprehensive documentation about the TrafCast project:

### 📁 Files Overview

1. **`project_overview.md`** - Complete project description and objectives
2. **`data_collection.md`** - Detailed explanation of data gathering process
3. **`data_processing.md`** - Data preprocessing and feature engineering
4. **`model_architecture.md`** - LSTM model design and training scheme
5. **`challenges_solutions.md`** - Data imbalance issues and weighting solutions
6. **`roadmap_system.md`** - Road network mapping and coordinate matching
7. **`training_scheme.md`** - Training configuration and hyperparameters
8. **`results_performance.md`** - Model performance metrics and evaluation
9. **`technical_implementation.md`** - Code structure and implementation details
10. **`presentation_slides.md`** - Formatted presentation slides for presentation

### 🎯 Key Highlights

- **Data Scale**: 1 month of traffic data from ~20 major LA highways
- **Model**: Bidirectional LSTM with weighted Huber loss
- **Performance**: 4.18 mph MAE on test set
- **Innovation**: Addresses speed class imbalance with custom weighting
- **Real-world Application**: Interactive road network visualization

### 🚀 Quick Start

1. Review `project_overview.md` for project context
2. Check `model_architecture.md` for technical details
3. See `results_performance.md` for performance metrics
4. Use `presentation_slides.md` for formal presentation

### 📊 Current Training Status

The model is currently being trained with the following configuration:
- **Epochs**: 20
- **Batch Size**: 128
- **Hidden Size**: 256
- **Architecture**: Bidirectional LSTM
- **Loss Function**: Weighted Huber Loss
- **Data**: Full training dataset

Training is in progress and results will be available upon completion.

---

*Generated for TrafCast Deep Learning Project - University Course*
