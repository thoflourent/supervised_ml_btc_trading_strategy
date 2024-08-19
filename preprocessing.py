import pandas as pd

from technical_analysis_lib import TechnicalAnalysis
from label_creation_utils import LabelCreation
from constants import B_WINDOW, F_WINDOW, ALPHA, BETA


## preprocessed dataset creation
class FeatureDataPipeline:

    @staticmethod
    def feature_engineering(data):
        transformed_data = TechnicalAnalysis.compute_oscillators(data)
        transformed_data = TechnicalAnalysis.add_timely_data(transformed_data)
        transformed_data = TechnicalAnalysis.find_patterns(transformed_data)

        return transformed_data
    
## label creation
    @staticmethod
    def label_creation(data):
        return LabelCreation.assign_labels(data, B_WINDOW, F_WINDOW, ALPHA, BETA)


    @staticmethod
    def get_features_and_labels(data):
        df_features = FeatureDataPipeline.feature_engineering(data)
        labels = FeatureDataPipeline.label_creation(df_features)

        return df_features.drop(['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Asset_name'], axis=1), labels
    
    @staticmethod
    def save_features_and_labels(data, path_to_save):
        df_features = FeatureDataPipeline.feature_engineering(data)
        labels = FeatureDataPipeline.label_creation(df_features)

        df_final = pd.concat([df_features.drop(['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Asset_name'], axis=1), labels], axis=1)

        df_final.to_pickle(path_to_save)
        print(f'Data saved in path : {path_to_save}')