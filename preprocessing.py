import pandas as pd

from technical_analysis_lib import TechnicalAnalysis
from constants import B_WINDOW, F_WINDOW, ALPHA_PTH, BETA_PTH, PCT_PERIOD


class LabelCreation:

    @staticmethod
    def get_alpha_and_beta(data, alpha_pth=ALPHA_PTH, beta_pth=BETA_PTH, pct_period=PCT_PERIOD):
        pct_change = data['Close'].pct_change()

        alpha = pct_change.abs().quantile(alpha_pth)
        beta = pct_change.abs().quantile(beta_pth)

        return alpha, beta
    
    @staticmethod
    def assign_labels(data, b_window, f_window, alpha, beta):
        x = data.copy()
        x['Close_MA'] = x['Close'].ewm(span=b_window).mean()
        x['s-1'] = x['Close'].shift(-1 * f_window)
        x['alpha'] = alpha
        x['beta'] = beta * (1 + (f_window * 0.1))
        x['label'] = x.apply(LabelCreation.check_label, axis=1)
        return x['label']

    @staticmethod
    def check_label(z):
        if (abs((z['s-1'] - z['Close_MA']) / z['Close_MA']) > z['alpha']) and \
                (abs((z['s-1'] - z['Close_MA']) / z['Close_MA']) < (z['beta'])):
            if z['s-1'] > z['Close_MA']:
                return -1
            elif z['s-1'] < z['Close_MA']:
                return 1
            else:
                return 0
        else:
            return 0


## preprocessed dataset creation
class Preprocessing:

    def __init__(self) -> None:
        self.alpha = None
        self.beta = None
        self.features = None
        self.labels = None

    @staticmethod
    def feature_engineering(data):
        transformed_data = TechnicalAnalysis.compute_oscillators(data)
        transformed_data = TechnicalAnalysis.add_timely_data(transformed_data)
        transformed_data = TechnicalAnalysis.find_patterns(transformed_data)

        return transformed_data
        
## label creation
    def label_creation(self, data):
        alpha, beta = LabelCreation.get_alpha_and_beta(data, alpha_pth=ALPHA_PTH, beta_pth=BETA_PTH, pct_period=PCT_PERIOD)
        print(f"Alpha= {alpha} & Beta= {beta}")
        self.alpha = alpha
        self.beta = beta
        return LabelCreation.assign_labels(data, B_WINDOW, F_WINDOW, alpha, beta)
    
    def create_features_and_labels(self, data):
        df_features = Preprocessing.feature_engineering(data)
        self.labels = self.label_creation(df_features)
        self.features = df_features.drop(['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Asset_name'], axis=1)
        print("Features and labels created")

    def get_features_and_labels(self, data):
        if self.features == None:
            print("Features haven't been created")
            return None
        elif self.labels == None:
            print("Labels haven't been created")
            return None
        else:
            return self.features, self.labels

    def save_features_and_labels(self, path_to_save):
        if self.features == None:
            print("Features haven't been created")
            return None
        elif self.labels == None:
            print("Labels haven't been created")
            return None

        df_final = pd.concat([self.features, self.labels], axis=1)
        df_final.to_pickle(path_to_save)
        print(f'Preprocessed data saved in path : {path_to_save}')