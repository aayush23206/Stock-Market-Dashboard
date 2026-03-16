"""
Forecasting module with ARIMA and LSTM models for stock price prediction.
Includes model training, evaluation, and visualization utilities.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller, kpss
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ARIMAForecaster:
    """ARIMA model for time series forecasting."""
    
    def __init__(self, train_data: pd.Series, order: Tuple[int, int, int] = None):
        """
        Initialize ARIMA model.
        
        Args:
            train_data: Training time series data
            order: (p, d, q) parameters. If None, auto-detect.
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels not installed. Install via: pip install statsmodels")
        
        self.train_data = train_data
        self.model = None
        self.fitted_model = None
        self.predictions = None
        
        if order is None:
            self.order = self._auto_detect_order()
        else:
            self.order = order
    
    def _auto_detect_order(self) -> Tuple[int, int, int]:
        """Auto-detect ARIMA order using ACF/PACF analysis."""
        # Perform ADF test to determine d
        adf_result = adfuller(self.train_data, autolag='AIC')
        d = 0 if adf_result[1] < 0.05 else 1
        
        # Simple heuristic for p and q (can be improved with grid search)
        p, q = 1, 1
        return (p, d, q)
    
    def fit(self) -> Dict:
        """Fit ARIMA model."""
        try:
            self.fitted_model = ARIMA(self.train_data, order=self.order).fit()
            
            return {
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'order': self.order,
                'summary': str(self.fitted_model.summary())
            }
        except Exception as e:
            return {'error': str(e)}
    
    def forecast(self, steps: int) -> np.ndarray:
        """
        Generate forecast.
        
        Args:
            steps: Number of periods to forecast
        
        Returns:
            Array of forecasted values
        """
        if self.fitted_model is None:
            self.fit()
        
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        return forecast_result.predicted_mean.values
    
    def get_confidence_intervals(self, steps: int, confidence: float = 0.95):
        """Get forecast with confidence intervals."""
        if self.fitted_model is None:
            self.fit()
        
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        forecastci = forecast_result.get_forecast_view(mu=forecast_result.predicted_mean)
        conf_int = forecast_result.conf_int(alpha=1-confidence)
        
        return {
            'forecast': forecast_result.predicted_mean.values,
            'lower_bound': conf_int.iloc[:, 0].values,
            'upper_bound': conf_int.iloc[:, 1].values
        }
    
    def evaluate(self, test_data: pd.Series) -> Dict:
        """Evaluate model on test data."""
        predictions = self.forecast(steps=len(test_data))
        
        mse = mean_squared_error(test_data, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data, predictions)
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        r2 = r2_score(test_data, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2_score': r2,
            'predictions': predictions
        }


class LSTMForecaster:
    """LSTM neural network for time series forecasting."""
    
    def __init__(self, train_data: pd.Series, lookback: int = 60):
        """
        Initialize LSTM model.
        
        Args:
            train_data: Training time series data
            lookback: Number of previous timesteps to use for prediction
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("tensorflow not installed. Install via: pip install tensorflow")
        
        self.train_data = train_data.values.reshape(-1, 1)
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.scaled_data = self.scaler.fit_transform(self.train_data)
    
    def prepare_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM."""
        X, y = [], []
        for i in range(len(data) - self.lookback - 1):
            X.append(data[i:(i + self.lookback), 0])
            y.append(data[i + self.lookback, 0])
        return np.array(X), np.array(y)
    
    def build_model(self) -> Sequential:
        """Build LSTM architecture."""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    def fit(self, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2) -> Dict:
        """
        Train LSTM model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
        
        Returns:
            Training history
        """
        X, y = self.prepare_data(self.scaled_data)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        self.model = self.build_model()
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        return {
            'loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1],
            'epochs': len(history.history['loss'])
        }
    
    def forecast(self, steps: int) -> np.ndarray:
        """
        Generate forecast.
        
        Args:
            steps: Number of periods to forecast
        
        Returns:
            Array of forecasted values (unscaled)
        """
        if self.model is None:
            self.fit()
        
        last_sequence = self.scaled_data[-self.lookback:]
        predictions = []
        
        for _ in range(steps):
            X = last_sequence.reshape(1, self.lookback, 1)
            next_pred = self.model.predict(X, verbose=0)[0, 0]
            predictions.append(next_pred)
            last_sequence = np.append(last_sequence[1:], next_pred)
        
        # Inverse transform to get original scale
        predictions = np.array(predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(predictions).flatten()
    
    def evaluate(self, test_data: pd.Series) -> Dict:
        """Evaluate model on test data."""
        test_scaled = self.scaler.transform(test_data.values.reshape(-1, 1))
        
        # Concatenate train and test for context
        total_data = np.vstack([self.scaled_data, test_scaled])
        
        # Prepare test data with lookback
        X_test, y_test = self.prepare_data(total_data[-len(test_data) - self.lookback:])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        predictions_scaled = self.model.predict(X_test, verbose=0)
        predictions = self.scaler.inverse_transform(predictions_scaled).flatten()
        y_test_unscaled = test_data.values[self.lookback:]
        
        mse = mean_squared_error(y_test_unscaled, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_unscaled, predictions)
        mape = np.mean(np.abs((y_test_unscaled - predictions) / y_test_unscaled)) * 100
        r2 = r2_score(y_test_unscaled, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2_score': r2,
            'predictions': predictions
        }


class ForecastComparison:
    """Compare different forecasting models."""
    
    def __init__(self, train_data: pd.Series, test_data: pd.Series):
        """
        Initialize comparison.
        
        Args:
            train_data: Training data
            test_data: Test data
        """
        self.train_data = train_data
        self.test_data = test_data
        self.results = {}
    
    def run_arima(self) -> Dict:
        """Run ARIMA forecast."""
        try:
            forecaster = ARIMAForecaster(self.train_data)
            forecaster.fit()
            self.results['arima'] = forecaster.evaluate(self.test_data)
            return self.results['arima']
        except Exception as e:
            return {'error': str(e)}
    
    def run_lstm(self) -> Dict:
        """Run LSTM forecast."""
        try:
            forecaster = LSTMForecaster(self.train_data)
            forecaster.fit(epochs=30)  # Reduced for speed
            self.results['lstm'] = forecaster.evaluate(self.test_data)
            return self.results['lstm']
        except Exception as e:
            return {'error': str(e)}
    
    def run_naive_baseline(self) -> Dict:
        """Run naive baseline (last value repeated)."""
        predictions = np.full(len(self.test_data), self.train_data.iloc[-1])
        
        mse = mean_squared_error(self.test_data, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.test_data, predictions)
        mape = np.mean(np.abs((self.test_data - predictions) / self.test_data)) * 100
        r2 = r2_score(self.test_data, predictions)
        
        self.results['naive'] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2_score': r2,
            'predictions': predictions
        }
        return self.results['naive']
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """Get comparison summary of all models."""
        summary_data = []
        
        for model_name, metrics in self.results.items():
            if 'error' not in metrics:
                summary_data.append({
                    'Model': model_name.upper(),
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'MAPE (%)': metrics['mape'],
                    'R² Score': metrics['r2_score']
                })
        
        return pd.DataFrame(summary_data).sort_values('RMSE')
