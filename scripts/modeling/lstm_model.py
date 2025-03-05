import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

class LSTMModel:
    """
    Implements LSTM for time series forecasting.
    """

    def __init__(self, train, test, lookback=30):
        self.train = train
        self.test = test
        self.lookback = lookback
        self.model = None

    def create_sequences(self, data):
        """
        Generate LSTM sequences.
        """
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i: i + self.lookback])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)

    def build_model(self):
        """
        Build LSTM model.
        """
        self.model = Sequential([
            LSTM(50, activation="relu", return_sequences=True, input_shape=(self.lookback, 1)),
            LSTM(50, activation="relu"),
            Dense(1)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    def train_model(self, epochs=20, batch_size=16):
        """
        Train LSTM model.
        """
        X_train, y_train = self.create_sequences(self.train)
        X_test, y_test = self.create_sequences(self.test)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    def forecast(self, last_data):
        """
        Forecast next value.
        """
        last_sequence = np.array(last_data[-self.lookback:]).reshape(1, self.lookback, 1)
        return self.model.predict(last_sequence)[0, 0]

