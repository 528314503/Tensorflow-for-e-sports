import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self, file_paths):
        dfs = [pd.read_csv(path) for path in file_paths]
        return pd.concat(dfs, axis=0, ignore_index=True)

    def preprocess(self, df):
        df = self.handle_missing_values(df)
        df = self.create_features(df)
        df = self.scale_features(df)
        return df

    def handle_missing_values(self, df):
        return df.fillna(df.mean())

    def create_features(self, df):
        df['user_engagement_score'] = df['view_time'] * df['interaction_count']
        df['event_popularity'] = df.groupby('event_id')['user_id'].transform('count')
        return df

    def scale_features(self, df):
        numeric_features = ['user_engagement_score', 'event_popularity', 'ticket_price']
        df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
        return df