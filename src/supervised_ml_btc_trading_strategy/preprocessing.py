import pandas as pd
import numpy as np
import talib as talib
from sklearn.model_selection import train_test_split

from src.supervised_ml_btc_trading_strategy.constants import B_WINDOW, F_WINDOW, ALPHA_PTH, BETA_PTH, PCT_PERIOD, MA_TIMEFRAME

class FeaturesCreator:
    def __init__(self):
        self.mean_volume = None
        self.std_volume = None
        self.ma_timeframe=None
        self.add_patterns=None

    def fit(self, X: pd.DataFrame, ma_timeframe: list, add_patterns=False):
        self.mean_volume = X['Volume'].mean()
        self.std_volume = X['Volume'].std()
        self.ma_timeframe = ma_timeframe
        self.add_patterns = add_patterns

    def create_features(self, X: pd.DataFrame):
        if isinstance(self.mean_volume, (float, int)) and isinstance(self.std_volume, (float, int)):
            log_return = np.log(X['Close']) - np.log(X['Close'].shift(1))
            X['Range'] = (X['High'] / X['Low']) - 1
            X['Z_score'] = (((log_return - log_return.rolling(20).mean()) / log_return.rolling(20).std()))
            X['RSI'] = ((talib.RSI(X['Close'])) / 100)
            upper_band, _, lower_band = talib.BBANDS(X['Close'], nbdevup=2, nbdevdn=2, matype=0)
            X['boll'] = ((X['Close'] - lower_band) / (upper_band - lower_band))
            X['ULTOSC'] = ((talib.ULTOSC(X['High'], X['Low'], X['Close'])) / 100)
            X['MACD_fast'], X['MACD_slow'], _ = talib.MACD(X['Close'])
            X['pct_change'] = (X['Close'].pct_change())
            X['zsVol'] = (X['Volume'] - self.mean_volume) / self.std_volume
            for tf in self.ma_timeframe:
                X[f'MA_{tf}'] = talib.MA(X['Close'], tf)
                X[f'EMA_{tf}'] = talib.EMA(X['Close'], 21)
            X['PR_MA_Ratio_short'] = \
                ((X['Close'] - X['MA_21']) / X['MA_21'])
            X['MA_Ratio_short'] = \
                ((X['MA_21'] - X['MA_50']) / X['MA_50'])
            X['MA_Ratio'] = (
                        (X['MA_50'] - X['MA_100']) / X['MA_100'])
            X['PR_MA_Ratio'] = ((X['Close'] - X['MA_50']) / X['MA_50'])
            
            # Add timely features
            df_day_of_week = pd.to_datetime(X['Date']).dt.dayofweek
            df_month = pd.to_datetime(X['Date']).dt.month
            df_hourly = pd.to_datetime(X['Date']).dt.hour

            X['day_of_week_sin'] = np.sin(2 * np.pi * df_day_of_week / 7)
            X['day_of_week_cos'] = np.cos(2 * np.pi * df_day_of_week / 7)

            X['month_sin'] = np.sin(2 * np.pi * df_month / 12)
            X['month_cos'] = np.cos(2 * np.pi * df_month / 12)

            X['hour_sin'] = np.sin(2 * np.pi * df_hourly / 24)
            X['hour_cos'] = np.cos(2 * np.pi * df_hourly / 24)

            if self.add_patterns:
                X['CDL2CROWS'] = talib.CDL2CROWS(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLBELTHOLD'] = talib.CDLBELTHOLD(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLENGULFING'] = talib.CDLENGULFING(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLMARUBOZU'] = talib.CDLMARUBOZU(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLPIERCING'] = talib.CDLPIERCING(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(X['Open'], X['High'], X['Low'], X['Close']) / 100
                X['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(X['Open'], X['High'], X['Low'], X['Close']) / 100
        else:
            print('Cannot create the features without fitting the FeatureCreator instance')
        
        return X

    def transform(self, X):
        return self.create_features(X)
    
    def fit_transform(self, X, ma_timeframe, add_patterns=False):
        self.fit(X, ma_timeframe, add_patterns)
        return self.transform(X)

class LabelCreator:
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.b_window = None
        self.f_window = None

    def fit_alpha_and_beta(self, X, b_window, f_window, alpha_pth, beta_pth, pct_period):
        self.b_window = b_window
        self.f_window = f_window
        pct_change = X['Close'].pct_change(periods=pct_period)
        alpha = pct_change.abs().quantile(alpha_pth)
        beta = pct_change.abs().quantile(beta_pth)
        self.alpha, self.beta = alpha, beta
        print(f"alpha={self.alpha} and beta={self.beta}")
    
    def create_label(self, X):
        x = X.copy()
        x['Close_MA'] = x['Close'].ewm(span=self.b_window).mean()
        x['s-1'] = x['Close'].shift(-1 * self.f_window)
        x['alpha'] = self.alpha
        x['beta'] = self.beta * (1 + (self.f_window * 0.1))
        x['signal'] = x.apply(LabelCreator.check_label, axis=1)
        return x[['Date', 'signal']]
    
    def fit_create_label(self, X, b_window, f_window, alpha_pth=ALPHA_PTH, beta_pth=BETA_PTH, pct_period=PCT_PERIOD):
        self.fit_alpha_and_beta(X, b_window, f_window, alpha_pth, beta_pth, pct_period)
        return self.create_label(X)
    
    def fit_create_label_light(self, X, b_window, f_window, alpha_pth=ALPHA_PTH, beta_pth=BETA_PTH, pct_period=PCT_PERIOD):
        df = self.fit_create_label(X, b_window, f_window, alpha_pth, beta_pth, pct_period)
        return df['signal']

    @staticmethod
    def check_label(z):
        if (abs((z['s-1'] - z['Close_MA']) / z['Close_MA']) > z['alpha']) and \
                (abs((z['s-1'] - z['Close_MA']) / z['Close_MA']) < (z['beta'])):
            if z['s-1'] > z['Close_MA']:
                return 1
            elif z['s-1'] < z['Close_MA']:
                return -1
            else:
                return 0
        else:
            return 0


def create_train_test_data(data: pd.DataFrame, **params) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ''' Initialise the train and test data using cleaned data, train_size and parameters to create features and labels'''
    features_creator = FeaturesCreator()
    label_creator = LabelCreator()
    X_train, X_test = train_test_split(data, train_size=params['train_size'], shuffle=False, stratify=None)

    features_train = features_creator.fit_transform(X=X_train, add_patterns=False, ma_timeframe=params['MA_TIMEFRAME'])
    features_test = features_creator.transform(X=X_test)

    labels_train = label_creator.fit_create_label(X=X_train, b_window=params['B_WINDOW'],
                                                  f_window=params['F_WINDOW'],
                                                  alpha_pth=params['ALPHA_PTH'],
                                                  beta_pth=params['BETA_PTH'],
                                                  pct_period=params['PCT_PERIOD']
                                                  )
    labels_test = label_creator.create_label(X=X_test)

    return features_train, labels_train, features_test, labels_test