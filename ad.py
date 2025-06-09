import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class LogAnomalyDetector:
    """
    A comprehensive anomaly detection system for log data with mixed data types.
    Uses multiple techniques: Isolation Forest, DBSCAN, Statistical methods, and PCA.
    """
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize the anomaly detector.
        
        Parameters:
        contamination (float): Expected proportion of outliers in the data
        random_state (int): Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.preprocessor = None
        self.isolation_forest = None
        self.dbscan = None
        self.pca = None
        self.feature_names = None
        self.numerical_features = None
        self.categorical_features = None
        self.datetime_features = None
        self.fitted = False
        
    def _identify_feature_types(self, df):
        """Automatically identify numerical, categorical, and datetime features."""
        numerical_features = []
        categorical_features = []
        datetime_features = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numerical_features.append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_features.append(col)
            else:
                # Try to convert to datetime
                try:
                    pd.to_datetime(df[col].dropna().iloc[:100])
                    datetime_features.append(col)
                except:
                    categorical_features.append(col)
        
        return numerical_features, categorical_features, datetime_features
    
    def _extract_datetime_features(self, df, datetime_cols):
        """Extract useful features from datetime columns."""
        df_processed = df.copy()
        
        for col in datetime_cols:
            if not pd.api.types.is_datetime64_any_dtype(df_processed[col]):
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
            
            # Extract time-based features
            df_processed[f'{col}_hour'] = df_processed[col].dt.hour
            df_processed[f'{col}_day_of_week'] = df_processed[col].dt.dayofweek
            df_processed[f'{col}_month'] = df_processed[col].dt.month
            df_processed[f'{col}_is_weekend'] = (df_processed[col].dt.dayofweek >= 5).astype(int)
            
            # Time since epoch (for trend analysis)
            df_processed[f'{col}_timestamp'] = df_processed[col].astype('int64') // 10**9
            
        return df_processed
    
    def _create_preprocessor(self, numerical_features, categorical_features):
        """Create preprocessing pipeline for different feature types."""
        transformers = []
        
        if numerical_features:
            numerical_transformer = StandardScaler()
            transformers.append(('num', numerical_transformer, numerical_features))
        
        if categorical_features:
            # Use OneHotEncoder for categorical features with handle_unknown='ignore'
            categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        if not transformers:
            raise ValueError("No valid features found for preprocessing")
            
        return ColumnTransformer(transformers=transformers, remainder='drop')
    
    def fit(self, df, target_column=None):
        """
        Fit the anomaly detection models on the log data.
        
        Parameters:
        df (pd.DataFrame): Input log data
        target_column (str): Optional target column to exclude from features
        """
        print("Fitting anomaly detection models...")
        
        # Prepare data
        df_work = df.copy()
        if target_column and target_column in df_work.columns:
            df_work = df_work.drop(columns=[target_column])
        
        # Identify feature types
        self.numerical_features, self.categorical_features, self.datetime_features = \
            self._identify_feature_types(df_work)
        
        print(f"Identified features:")
        print(f"  Numerical: {len(self.numerical_features)} features")
        print(f"  Categorical: {len(self.categorical_features)} features") 
        print(f"  Datetime: {len(self.datetime_features)} features")
        
        # Process datetime features
        if self.datetime_features:
            df_work = self._extract_datetime_features(df_work, self.datetime_features)
            # Update feature lists after datetime processing
            new_num_features = [col for col in df_work.columns if col.endswith(('_hour', '_day_of_week', '_month', '_timestamp'))]
            new_cat_features = [col for col in df_work.columns if col.endswith('_is_weekend')]
            self.numerical_features.extend(new_num_features)
            self.categorical_features.extend(new_cat_features)
            df_work = df_work.drop(columns=self.datetime_features)
        
        # Handle missing values
        for col in self.numerical_features:
            if col in df_work.columns:
                df_work[col] = df_work[col].fillna(df_work[col].median())
        
        for col in self.categorical_features:
            if col in df_work.columns:
                df_work[col] = df_work[col].fillna('missing')
        
        # Create and fit preprocessor
        self.preprocessor = self._create_preprocessor(self.numerical_features, self.categorical_features)
        X_processed = self.preprocessor.fit_transform(df_work)
        
        # Store feature names for later use
        feature_names = []
        if self.numerical_features:
            feature_names.extend(self.numerical_features)
        if self.categorical_features:
            # Get feature names from OneHotEncoder
            cat_transformer = self.preprocessor.named_transformers_['cat']
            cat_feature_names = cat_transformer.get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_feature_names)
        
        self.feature_names = feature_names
        
        # Fit models
        print("Training Isolation Forest...")
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        self.isolation_forest.fit(X_processed)
        
        print("Training DBSCAN...")
        # Use smaller sample for DBSCAN if dataset is large
        if len(X_processed) > 10000:
            sample_idx = np.random.choice(len(X_processed), 10000, replace=False)
            X_sample = X_processed[sample_idx]
        else:
            X_sample = X_processed
            
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.dbscan.fit(X_sample)
        
        print("Training PCA...")
        self.pca = PCA(n_components=min(10, X_processed.shape[1]))
        self.pca.fit(X_processed)
        
        self.fitted = True
        print("Model training completed!")
        
        return self
    
    def predict_anomalies(self, df, target_column=None, method='ensemble'):
        """
        Predict anomalies in the log data.
        
        Parameters:
        df (pd.DataFrame): Input log data
        target_column (str): Optional target column to exclude from features
        method (str): 'isolation_forest', 'dbscan', 'statistical', 'pca', or 'ensemble'
        
        Returns:
        dict: Dictionary containing anomaly predictions and scores
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare data (same preprocessing as fit)
        df_work = df.copy()
        if target_column and target_column in df_work.columns:
            df_work = df_work.drop(columns=[target_column])
        
        # Process datetime features
        if self.datetime_features:
            df_work = self._extract_datetime_features(df_work, self.datetime_features)
            df_work = df_work.drop(columns=self.datetime_features)
        
        # Handle missing values
        for col in self.numerical_features:
            if col in df_work.columns:
                df_work[col] = df_work[col].fillna(df_work[col].median())
        
        for col in self.categorical_features:
            if col in df_work.columns:
                df_work[col] = df_work[col].fillna('missing')
        
        # Transform data
        X_processed = self.preprocessor.transform(df_work)
        
        results = {}
        
        if method in ['isolation_forest', 'ensemble']:
            # Isolation Forest predictions (-1 for anomaly, 1 for normal)
            if_predictions = self.isolation_forest.predict(X_processed)
            if_scores = self.isolation_forest.score_samples(X_processed)
            results['isolation_forest'] = {
                'predictions': (if_predictions == -1).astype(int),
                'scores': -if_scores  # Convert to positive scores
            }
        
        if method in ['statistical', 'ensemble']:
            # Statistical anomaly detection using Z-score
            statistical_anomalies = np.zeros(len(X_processed))
            z_scores = np.abs(stats.zscore(X_processed, axis=0, nan_policy='omit'))
            statistical_anomalies = (z_scores > 3).any(axis=1).astype(int)
            results['statistical'] = {
                'predictions': statistical_anomalies,
                'scores': np.max(z_scores, axis=1)
            }
        
        if method in ['pca', 'ensemble']:
            # PCA-based anomaly detection
            X_pca = self.pca.transform(X_processed)
            X_reconstructed = self.pca.inverse_transform(X_pca)
            reconstruction_errors = np.sum((X_processed - X_reconstructed) ** 2, axis=1)
            pca_threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
            results['pca'] = {
                'predictions': (reconstruction_errors > pca_threshold).astype(int),
                'scores': reconstruction_errors
            }
        
        if method == 'ensemble':
            # Ensemble method: combine multiple approaches
            ensemble_scores = np.zeros(len(X_processed))
            ensemble_predictions = np.zeros(len(X_processed))
            
            methods_used = []
            if 'isolation_forest' in results:
                ensemble_scores += results['isolation_forest']['scores']
                ensemble_predictions += results['isolation_forest']['predictions']
                methods_used.append('isolation_forest')
            
            if 'statistical' in results:
                # Normalize statistical scores to 0-1 range
                stat_scores_norm = (results['statistical']['scores'] - results['statistical']['scores'].min()) / \
                                 (results['statistical']['scores'].max() - results['statistical']['scores'].min() + 1e-8)
                ensemble_scores += stat_scores_norm
                ensemble_predictions += results['statistical']['predictions']
                methods_used.append('statistical')
            
            if 'pca' in results:
                # Normalize PCA scores to 0-1 range
                pca_scores_norm = (results['pca']['scores'] - results['pca']['scores'].min()) / \
                                (results['pca']['scores'].max() - results['pca']['scores'].min() + 1e-8)
                ensemble_scores += pca_scores_norm
                ensemble_predictions += results['pca']['predictions']
                methods_used.append('pca')
            
            # Average the scores and predictions
            ensemble_scores /= len(methods_used)
            ensemble_predictions = (ensemble_predictions >= len(methods_used) / 2).astype(int)
            
            results['ensemble'] = {
                'predictions': ensemble_predictions,
                'scores': ensemble_scores,
                'methods_used': methods_used
            }
        
        return results[method] if method != 'ensemble' else results['ensemble']
    
    def analyze_anomalies(self, df, anomaly_results, top_n=10):
        """
        Analyze and interpret the detected anomalies.
        
        Parameters:
        df (pd.DataFrame): Original data
        anomaly_results (dict): Results from predict_anomalies
        top_n (int): Number of top anomalies to analyze
        
        Returns:
        pd.DataFrame: DataFrame with anomaly analysis
        """
        anomaly_indices = np.where(anomaly_results['predictions'] == 1)[0]
        
        if len(anomaly_indices) == 0:
            print("No anomalies detected!")
            return pd.DataFrame()
        
        # Get top anomalies by score
        top_anomaly_indices = anomaly_indices[np.argsort(anomaly_results['scores'][anomaly_indices])[-top_n:]]
        
        # Create analysis DataFrame
        analysis_df = df.iloc[top_anomaly_indices].copy()
        analysis_df['anomaly_score'] = anomaly_results['scores'][top_anomaly_indices]
        analysis_df['anomaly_rank'] = range(len(top_anomaly_indices), 0, -1)
        
        return analysis_df.sort_values('anomaly_score', ascending=False)
    
    def plot_anomaly_analysis(self, df, anomaly_results, figsize=(15, 10)):
        """
        Create visualizations for anomaly analysis.
        
        Parameters:
        df (pd.DataFrame): Original data
        anomaly_results (dict): Results from predict_anomalies
        figsize (tuple): Figure size for plots
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Log Anomaly Detection Analysis', fontsize=16)
        
        # 1. Anomaly score distribution
        axes[0, 0].hist(anomaly_results['scores'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(anomaly_results['scores']), color='red', linestyle='--', label='Mean')
        axes[0, 0].set_title('Distribution of Anomaly Scores')
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 2. Anomaly detection results
        normal_indices = np.where(anomaly_results['predictions'] == 0)[0]
        anomaly_indices = np.where(anomaly_results['predictions'] == 1)[0]
        
        axes[0, 1].scatter(normal_indices, anomaly_results['scores'][normal_indices], 
                          alpha=0.6, c='blue', label=f'Normal ({len(normal_indices)})', s=20)
        axes[0, 1].scatter(anomaly_indices, anomaly_results['scores'][anomaly_indices], 
                          alpha=0.8, c='red', label=f'Anomaly ({len(anomaly_indices)})', s=30)
        axes[0, 1].set_title('Anomaly Detection Results')
        axes[0, 1].set_xlabel('Data Point Index')
        axes[0, 1].set_ylabel('Anomaly Score')
        axes[0, 1].legend()
        
        # 3. Top anomalies over time (if timestamp available)
        time_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                time_col = col
                break
        
        if time_col and len(anomaly_indices) > 0:
            try:
                df_temp = df.copy()
                df_temp[time_col] = pd.to_datetime(df_temp[time_col], errors='coerce')
                
                axes[1, 0].plot(df_temp[time_col], anomaly_results['scores'], alpha=0.5, color='gray', linewidth=1)
                axes[1, 0].scatter(df_temp[time_col].iloc[anomaly_indices], 
                                  anomaly_results['scores'][anomaly_indices], 
                                  color='red', s=50, alpha=0.8, label='Anomalies')
                axes[1, 0].set_title('Anomalies Over Time')
                axes[1, 0].set_xlabel('Time')
                axes[1, 0].set_ylabel('Anomaly Score')
                axes[1, 0].legend()
                plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
            except:
                axes[1, 0].text(0.5, 0.5, 'Time series plot not available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Time Series Analysis')
        else:
            axes[1, 0].text(0.5, 0.5, 'No time column found', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Time Series Analysis')
        
        # 4. Feature importance (top features contributing to anomalies)
        if hasattr(self, 'feature_names') and self.feature_names:
            # Calculate feature importance based on variance in anomalous samples
            df_work = df.copy()
            if self.datetime_features:
                df_work = self._extract_datetime_features(df_work, self.datetime_features)
                df_work = df_work.drop(columns=self.datetime_features)
            
            # Handle missing values
            for col in self.numerical_features:
                if col in df_work.columns:
                    df_work[col] = df_work[col].fillna(df_work[col].median())
            
            X_processed = self.preprocessor.transform(df_work)
            
            if len(anomaly_indices) > 0:
                feature_importance = np.var(X_processed[anomaly_indices], axis=0)
                top_features_idx = np.argsort(feature_importance)[-10:]
                
                axes[1, 1].barh(range(len(top_features_idx)), feature_importance[top_features_idx])
                axes[1, 1].set_yticks(range(len(top_features_idx)))
                axes[1, 1].set_yticklabels([self.feature_names[i] if i < len(self.feature_names) 
                                           else f'Feature_{i}' for i in top_features_idx])
                axes[1, 1].set_title('Top Features in Anomalies')
                axes[1, 1].set_xlabel('Variance')
            else:
                axes[1, 1].text(0.5, 0.5, 'No anomalies to analyze', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature analysis not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Analysis')
        
        plt.tight_layout()
        plt.show()

# Example usage and testing
def demo_log_anomaly_detection():
    """
    Demonstrate the log anomaly detection system with sample data.
    """
    print("=== Log Anomaly Detection Demo ===\n")
    
    # Create sample log data
    np.random.seed(42)
    n_samples = 1000
    n_anomalies = 50
    
    # Normal log data
    normal_data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples-n_anomalies, freq='1min'),
        'user_id': np.random.choice(['user_1', 'user_2', 'user_3', 'user_4'], n_samples-n_anomalies),
        'response_time': np.random.normal(200, 50, n_samples-n_anomalies),
        'status_code': np.random.choice([200, 201, 404], n_samples-n_anomalies, p=[0.8, 0.15, 0.05]),
        'request_size': np.random.exponential(1000, n_samples-n_anomalies),
        'endpoint': np.random.choice(['/api/login', '/api/data', '/api/logout'], n_samples-n_anomalies)
    }
    
    # Anomalous log data
    anomaly_data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_anomalies, freq='1min'),
        'user_id': np.random.choice(['suspicious_user', 'user_1'], n_anomalies),
        'response_time': np.random.normal(2000, 200, n_anomalies),  # Much higher response time
        'status_code': np.random.choice([500, 503, 404], n_anomalies),  # More error codes
        'request_size': np.random.exponential(10000, n_anomalies),  # Larger requests
        'endpoint': np.random.choice(['/api/admin', '/api/data'], n_anomalies)
    }
    
    # Combine normal and anomalous data
    df_normal = pd.DataFrame(normal_data)
    df_anomaly = pd.DataFrame(anomaly_data)
    df = pd.concat([df_normal, df_anomaly], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    print(f"Created sample log data with {len(df)} records")
    print(f"Data shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    
    # Initialize and fit the detector
    detector = LogAnomalyDetector(contamination=0.05)  # Expect 5% anomalies
    detector.fit(df)
    
    # Predict anomalies
    print("\n=== Making Predictions ===")
    results = detector.predict_anomalies(df, method='ensemble')
    
    print(f"Detected {np.sum(results['predictions'])} anomalies out of {len(df)} records")
    print(f"Anomaly rate: {np.sum(results['predictions'])/len(df)*100:.2f}%")
    
    # Analyze top anomalies
    print("\n=== Top Anomalies Analysis ===")
    analysis = detector.analyze_anomalies(df, results, top_n=5)
    print(analysis)
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    detector.plot_anomaly_analysis(df, results)
    
    return detector, df, results

if __name__ == "__main__":
    # Run the demo
    detector, sample_df, sample_results = demo_log_anomaly_detection()
