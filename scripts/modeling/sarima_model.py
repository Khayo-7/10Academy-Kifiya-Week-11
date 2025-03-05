from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAModel:
    """
    Implements Seasonal ARIMA (SARIMA) for forecasting.
    """

    def __init__(self, train, test, order=(1,1,1), seasonal_order=(1,1,1,12)):
        self.train = train
        self.test = test
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None

    def train_model(self):
        """
        Train SARIMA model.
        """
        self.model = SARIMAX(self.train, order=self.order, seasonal_order=self.seasonal_order).fit()
        print(self.model.summary())

    def forecast(self, steps=30):
        """
        Forecast future values.
        """
        forecast = self.model.forecast(steps=steps)
        return forecast

    def evaluate(self):
        """
        Compute RMSE and MAE.
        """
        predictions = self.model.forecast(steps=len(self.test))
        rmse = np.sqrt(mean_squared_error(self.test, predictions))
        mae = mean_absolute_error(self.test, predictions)
        print(f"SARIMA RMSE: {rmse}, MAE: {mae}")
        return predictions
