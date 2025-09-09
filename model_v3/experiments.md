# 🏆 BEST PERFORMANCE MODEL (Lowest MAE)
python train_lstm.py --csv test1.csv --epochs 20 --batch_size 128 --hidden_size 256 --bidirectional --loss_type weighted_huber --model_out lstm_model_v3.pt
==================================================
EVALUATION METRICS
==================================================
MAE (Mean Absolute Error): 4.1815 mph
RMSE (Root Mean Square Error): 7.6657 mph
MAPE (Mean Absolute Percentage Error): 8.20%
R² (Coefficient of Determination): 0.4294

Speed Range Performance:
  Low ≤30: 11.5730 mph (n=1889)
  Medium 30-60: 7.1722 mph (n=15717)
  High ≥60: 3.2165 mph (n=63171)

Saving predictions to test_predictions.csv
Predictions saved with 80539 rows