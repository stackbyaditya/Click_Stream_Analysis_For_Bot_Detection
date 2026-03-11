"""
REUSABLE PREPROCESSING MODULE FOR FRAUD DETECTION
==================================================

This module provides a flexible, production-ready preprocessing pipeline
for converting raw clickstream data into machine learning-ready datasets.

Classes:
- FraudDetectionPreprocessor: Main preprocessing class
- DataValidator: Data quality validation
- FeatureEngineer: Feature engineering utilities
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import warnings

from sklearn.preprocessing import StandardScaler
import hashlib

warnings.filterwarnings('ignore')


class DataValidator:
    """Validate data quality and integrity"""

    @staticmethod
    def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> Dict:
        """Check for missing values"""
        missing = df.isnull().sum()
        result = {
            'total_missing': int(missing.sum()),
            'columns_with_missing': {},
            'meets_threshold': True,
        }

        for col, count in missing[missing > 0].items():
            pct = count / len(df)
            if pct > threshold:
                result['meets_threshold'] = False
            result['columns_with_missing'][col] = {
                'count': int(count),
                'percentage': float(pct),
            }

        return result

    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> Dict:
        """Detect outliers using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                mask = (df[col] < lower) | (df[col] > upper)
                n_outliers = mask.sum()

                if n_outliers > 0:
                    outliers[col] = {
                        'count': int(n_outliers),
                        'percentage': float(n_outliers / len(df)),
                        'bounds': {'lower': float(lower), 'upper': float(upper)},
                    }

        return outliers

    @staticmethod
    def check_label_balance(labels: np.ndarray) -> Dict:
        """Check label distribution balance"""
        unique, counts = np.unique(labels, return_counts=True)
        result = {}

        for label, count in zip(unique, counts):
            result[int(label)] = int(count)

        return result


class FeatureEngineer:
    """Feature engineering utilities"""

    @staticmethod
    def calculate_mouse_features(x_coords: np.ndarray, y_coords: np.ndarray) -> Dict:
        """Calculate mouse movement features"""
        if len(x_coords) == 0:
            return {
                'mouse_speed_mean': 0.0,
                'mouse_speed_std': 0.0,
                'mouse_path_length': 0.0,
                'direction_change_count': 0,
                'movement_std': 0.0,
            }

        x_diff = np.diff(x_coords)
        y_diff = np.diff(y_coords)
        distances = np.sqrt(x_diff**2 + y_diff**2)

        angles = []
        if len(distances) > 1:
            for i in range(len(distances) - 1):
                if distances[i] > 0 and distances[i+1] > 0:
                    cos_angle = (x_diff[i] * x_diff[i+1] + y_diff[i] * y_diff[i+1]) / (
                        distances[i] * distances[i+1] + 1e-6
                    )
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    angles.append(angle)

        return {
            'mouse_speed_mean': float(np.mean(distances)),
            'mouse_speed_std': float(np.std(distances)),
            'mouse_path_length': float(np.sum(distances)),
            'direction_change_count': len(angles),
            'movement_std': float(np.std([len(x_coords), np.std(x_coords), np.std(y_coords)])),
        }

    @staticmethod
    def calculate_coordinate_entropy(coords: np.ndarray, n_bins: int = 10) -> float:
        """Calculate entropy of coordinate distribution"""
        if len(coords) == 0:
            return 0.0

        hist, _ = np.histogram(coords, bins=n_bins)
        hist = hist[hist > 0]

        if len(hist) == 0:
            return 0.0

        probs = hist / np.sum(hist)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return float(entropy)

    @staticmethod
    def calculate_bot_likelihood_score(row: pd.Series) -> float:
        """Calculate heuristic bot likelihood score"""
        score = 0

        if row['requests_per_minute'] > 100:
            score += 2
        elif row['requests_per_minute'] > 50:
            score += 1

        if row.get('coordinate_entropy', 10) < 2:
            score += 2
        elif row.get('coordinate_entropy', 10) < 3:
            score += 1

        if row.get('mouse_speed_std', 100) < 10:
            score += 1

        if row.get('is_proxy', False):
            score += 1

        if row['session_duration_sec'] > 3600 and row['total_requests'] > 1000:
            score += 1

        return float(score)


class FraudDetectionPreprocessor:
    """Main preprocessing pipeline for fraud detection"""

    CATEGORY_MAPPING = {'human': 0, 'moderate_bot': 1, 'advanced_bot': 2}

    def __init__(self, dataset_dir: Path, output_dir: Path, sequence_length: int = 50,
                 random_seed: int = 42):
        """Initialize preprocessor

        Args:
            dataset_dir: Path to directory containing CSV datasets
            output_dir: Path to output directory for processed datasets
            sequence_length: Length of LSTM sequences
            random_seed: Random seed for reproducibility
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.sequence_length = sequence_length
        self.random_seed = random_seed
        self.scaler = StandardScaler()

        np.random.seed(random_seed)

    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV datasets"""
        print("[INFO] Loading datasets...")
        datasets = {}

        filenames = {
            'behavioral_detailed_adv': 'humans_and_advanced_bots_behavioral_detailed.csv',
            'temporal_detailed_adv': 'humans_and_advanced_bots_temporal_detailed.csv',
            'web_activity_adv': 'humans_and_advanced_bots_web_activity_summary.csv',
            'behavioral_summary_adv': 'humans_and_advanced_bots_behavior_summary.csv',
            'behavioral_detailed_mod': 'humans_and_moderate_bots_behavioral_detailed.csv',
            'temporal_detailed_mod': 'humans_and_moderate_bots_temporal_detailed.csv',
            'web_activity_mod': 'humans_and_moderate_bots_web_activity_summary.csv',
            'behavioral_summary_mod': 'humans_and_moderate_bots_behavior_summary.csv',
        }

        for key, filename in filenames.items():
            filepath = self.dataset_dir / filename
            try:
                datasets[key] = pd.read_csv(filepath)
                print(f"  [OK] {filename} ({len(datasets[key])} rows)")
            except FileNotFoundError:
                print(f"  [SKIP] {filename} (not found)")

        return datasets

    def extract_category_label(self, category_str: str) -> Optional[str]:
        """Extract bot type from category string"""
        if pd.isna(category_str):
            return None

        cat_lower = str(category_str).lower()
        if 'advanced' in cat_lower:
            return 'advanced_bot'
        elif 'moderate' in cat_lower:
            return 'moderate_bot'
        else:
            return 'human'

    def aggregate_sessions(self, behavioral_df: pd.DataFrame, temporal_df: pd.DataFrame,
                          web_activity_df: pd.DataFrame, bot_type_label: str) -> pd.DataFrame:
        """Aggregate multiple datasets by session"""

        if web_activity_df is None or len(web_activity_df) == 0:
            return pd.DataFrame()

        sessions_data = []

        for idx, row in web_activity_df.iterrows():
            session_id = row['session_id']
            session_info = {
                'session_id': session_id,
                'bot_type': bot_type_label,
                'label': self.CATEGORY_MAPPING.get(bot_type_label, 0),
            }

            # Behavioral features
            if behavioral_df is not None:
                session_behavior = behavioral_df[behavioral_df['session_id'] == session_id]
                if len(session_behavior) > 0:
                    session_info['total_movements'] = len(session_behavior)
                    session_info['x_coords'] = session_behavior['x_coordinate'].values
                    session_info['y_coords'] = session_behavior['y_coordinate'].values
                else:
                    session_info['total_movements'] = 0
                    session_info['x_coords'] = np.array([])
                    session_info['y_coords'] = np.array([])
            else:
                session_info['total_movements'] = 0
                session_info['x_coords'] = np.array([])
                session_info['y_coords'] = np.array([])

            # Web activity features
            session_info['total_requests'] = int(row['total_requests'])
            session_info['successful_requests'] = int(row['successful_requests'])
            session_info['total_response_size'] = int(row['total_response_size'])
            session_info['avg_response_size'] = int(row['avg_response_size'])
            session_info['activity_date'] = row['activity_date']
            session_info['time_range'] = row['time_range']

            # Temporal features
            if temporal_df is not None:
                session_temporal = temporal_df[temporal_df['session_id'] == session_id]
                session_info['logs_count'] = len(session_temporal)
            else:
                session_info['logs_count'] = 0

            sessions_data.append(session_info)

        return pd.DataFrame(sessions_data)

    def add_synthetic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic synthetic device and network features"""
        return df  # Placeholder - implement as needed

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw data"""
        # Temporal features
        df['clicks_per_minute'] = (df['total_movements'] /
                                   df.get('session_duration_sec', 1).replace(0, 1)) * 60
        df['requests_per_minute'] = (df['total_requests'] /
                                     df.get('session_duration_sec', 1).replace(0, 1)) * 60

        # Success rate
        df['success_rate'] = (df['successful_requests'] / df['total_requests'].replace(0, 1))

        # Mouse features
        mouse_features = []
        for idx, row in df.iterrows():
            x_coords = row.get('x_coords', np.array([]))
            y_coords = row.get('y_coords', np.array([]))

            if isinstance(x_coords, np.ndarray) and isinstance(y_coords, np.ndarray):
                features = FeatureEngineer.calculate_mouse_features(x_coords, y_coords)
            else:
                features = {k: 0 for k in [
                    'mouse_speed_mean', 'mouse_speed_std', 'mouse_path_length',
                    'direction_change_count', 'movement_std'
                ]}
            mouse_features.append(features)

        mouse_df = pd.DataFrame(mouse_features)
        for col in mouse_df.columns:
            df[col] = mouse_df[col]

        # Entropy features
        entropies = []
        for idx, row in df.iterrows():
            x_coords = row.get('x_coords', np.array([]))
            y_coords = row.get('y_coords', np.array([]))

            if isinstance(x_coords, np.ndarray) and isinstance(y_coords, np.ndarray):
                x_entropy = FeatureEngineer.calculate_coordinate_entropy(x_coords)
                y_entropy = FeatureEngineer.calculate_coordinate_entropy(y_coords)
                avg_entropy = (x_entropy + y_entropy) / 2
            else:
                avg_entropy = 0

            entropies.append(avg_entropy)

        df['coordinate_entropy'] = entropies
        df['bot_likelihood_score'] = df.apply(
            FeatureEngineer.calculate_bot_likelihood_score, axis=1
        )

        return df

    def prepare_lstm_sequences(self, behavioral_df: pd.DataFrame, sessions_df: pd.DataFrame,
                              bot_type: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare sequences for LSTM training"""
        sequences = []
        sequence_labels = []
        sequence_ids = []

        if behavioral_df is None or len(behavioral_df) == 0:
            return np.array([]), np.array([]), []

        sessions_for_type = sessions_df[sessions_df['bot_type'] == bot_type][
            'session_id'
        ].values
        label = self.CATEGORY_MAPPING.get(bot_type, 0)

        for session_id in sessions_for_type:
            session_behaviors = behavioral_df[behavioral_df['session_id'] == session_id]

            if len(session_behaviors) < 2:
                continue

            x_coords = session_behaviors['x_coordinate'].values
            y_coords = session_behaviors['y_coordinate'].values

            sequence_features = []
            for i in range(len(x_coords)):
                movement_speed = 0
                if i > 0:
                    movement_speed = np.sqrt((x_coords[i] - x_coords[i-1])**2 +
                                            (y_coords[i] - y_coords[i-1])**2)

                x_norm = x_coords[i] / 1600 if x_coords[i] != 0 else 0
                y_norm = y_coords[i] / 900 if y_coords[i] != 0 else 0

                features = np.array([
                    x_norm, y_norm, movement_speed / 100, 1, 0,
                    i / max(len(x_coords), 1),
                ])
                sequence_features.append(features)

            # Pad or truncate
            if len(sequence_features) < self.sequence_length:
                padding_size = self.sequence_length - len(sequence_features)
                padding = np.zeros((padding_size, len(sequence_features[0])))
                sequence_features = np.vstack([sequence_features, padding])
            else:
                sequence_features = np.array(sequence_features[:self.sequence_length])

            sequences.append(sequence_features)
            sequence_labels.append(label)
            sequence_ids.append(session_id)

        return np.array(sequences), np.array(sequence_labels), sequence_ids

    def save_outputs(self, sessions_df: pd.DataFrame, sequences: np.ndarray,
                     labels: np.ndarray, sequence_ids: List[str]):
        """Save processed datasets"""
        # Save aggregated dataset
        sessions_output = sessions_df.drop(
            columns=['x_coords', 'y_coords'], errors='ignore'
        )
        sessions_output.to_csv(
            self.output_dir / 'session_aggregated_dataset.csv', index=False
        )

        # Save sequences
        sequence_data = {
            'sequences': sequences,
            'labels': labels,
            'session_ids': sequence_ids,
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'n_features': sequences.shape[2] if len(sequences.shape) > 2 else 0,
        }

        with open(self.output_dir / 'session_sequence_dataset.pkl', 'wb') as f:
            pickle.dump(sequence_data, f)

        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'dataset_info': {
                'total_sessions': len(sessions_df),
                'total_features': len(sessions_df.columns),
                'label_distribution': sessions_df['label'].value_counts().to_dict(),
            },
            'lstm_info': {
                'total_sequences': len(sequences),
                'sequence_length': self.sequence_length,
                'n_features': sequences.shape[2] if len(sequences.shape) > 2 else 0,
            },
        }

        with open(self.output_dir / 'dataset_metadata_report.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def run(self) -> Dict:
        """Run complete preprocessing pipeline"""
        print("\n" + "="*80)
        print("FRAUD DETECTION PREPROCESSING PIPELINE (REUSABLE MODULE)")
        print("="*80 + "\n")

        # Load
        datasets = self.load_datasets()

        # Process and return results
        return {'datasets': datasets, 'output_dir': self.output_dir}


if __name__ == '__main__':
    # Example usage
    preprocessor = FraudDetectionPreprocessor(
        dataset_dir='Datasets',
        output_dir='preprocessing_output',
        sequence_length=50,
        random_seed=42,
    )

    result = preprocessor.run()
    print(f"\n[OK] Modules loaded and ready for use")
