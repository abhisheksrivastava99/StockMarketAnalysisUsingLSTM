# LSTM-Based Stock Price Prediction

## Overview
This project implements an LSTM (Long Short-Term Memory) neural network to predict stock prices using historical data. The model is trained on Google's stock price data obtained from Yahoo Finance and uses deep learning techniques to forecast future stock movements.

## Features
- Fetches historical stock data from Yahoo Finance.
- Preprocesses data using MinMax scaling.
- Splits data into training and test sets.
- Builds and trains an LSTM model with dropout layers to prevent overfitting.
- Evaluates model performance using RMSE.
- Visualizes predictions compared to actual stock prices.
- Forecasts future stock prices using the trained model.

## Dependencies
Ensure you have the following Python libraries installed:

```bash
pip install pandas pandas_datareader numpy matplotlib scikit-learn keras tensorflow
```

## Dataset
- The dataset consists of Google's historical stock prices from Yahoo Finance.
- The script loads data from a CSV file or fetches it online if not found.

## Usage
### 1. Load Data
The function `load_financial_data` reads the stock price data:
```python
df_APPL = load_financial_data(start_date='2001-01-01', end_date='2018-01-01', output_file='df_APPL.csv')
```

### 2. Data Preprocessing
- Extracts the closing prices.
- Normalizes values using MinMaxScaler.
- Splits the data into training and testing sets.

### 3. Model Training
The LSTM model is defined and trained:
```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=2)
```

### 4. Prediction & Evaluation
- The trained model makes predictions on the test data.
- RMSE is calculated to evaluate accuracy.
```python
math.sqrt(mean_squared_error(y_test, test_predict))
```

### 5. Visualization
- Plots actual vs. predicted stock prices.
- Shows forecasted values.
```python
plt.plot(y_test_inverse, 'g', label='Actual Prices')
plt.plot(test_predict, 'r', label='Predicted Prices')
plt.legend()
plt.show()
```

## Future Improvements
- Experiment with different LSTM architectures.
- Fine-tune hyperparameters for better accuracy.
- Use attention mechanisms to enhance performance.

## Author
Developed by **Abhishek Srivastava**.

## License
This project is licensed under the MIT License.
