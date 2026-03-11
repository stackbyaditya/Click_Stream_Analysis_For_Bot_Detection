"""
QUICK START GUIDE - USING PREPROCESSED DATASETS
================================================

This guide shows how to load and use the preprocessed datasets
for various machine learning tasks.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
PREPROCESSING_DIR = Path('preprocessing_output')

print("="*80)
print("PREPROCESSED DATASETS - QUICK START GUIDE")
print("="*80 + "\n")

# ============================================================================
# 1. LOAD AGGREGATED DATASET (for tree-based models)
# ============================================================================
print("[1] Loading Session-Aggregated Dataset")
print("-" * 80)

aggregated_df = pd.read_csv(PREPROCESSING_DIR / 'session_aggregated_dataset.csv')

print(f"Shape: {aggregated_df.shape}")
print(f"Columns: {list(aggregated_df.columns)}")
print(f"\nFirst few rows:")
print(aggregated_df[['session_id', 'label', 'bot_type', 'total_movements',
                      'total_requests', 'requests_per_minute']].head())

# ============================================================================
# 2. LOAD LSTM SEQUENCE DATASET
# ============================================================================
print("\n[2] Loading LSTM Sequence Dataset")
print("-" * 80)

with open(PREPROCESSING_DIR / 'session_sequence_dataset.pkl', 'rb') as f:
    sequence_data = pickle.load(f)

print(f"Sequences shape: {sequence_data['sequences'].shape}")
print(f"Labels shape: {sequence_data['labels'].shape}")
print(f"Sequence length: {sequence_data['sequence_length']}")
print(f"Features per timestep: {sequence_data['n_features']}")
print(f"Total sequences: {len(sequence_data['session_ids'])}")

# ============================================================================
# 3. PREPARE DATA FOR SKLEARN MODELS (Random Forest, XGBoost, SVM)
# ============================================================================
print("\n[3] Preparing Data for Sklearn Models")
print("-" * 80)

# Select only numeric features
numeric_df = aggregated_df.select_dtypes(include=[np.number])

# Separate features and target
X_numeric = numeric_df.drop('label', axis=1)
y = numeric_df['label']

print(f"Features: {X_numeric.shape[1]}")
print(f"Samples: {X_numeric.shape[0]}")
print(f"Target distribution:\n{y.value_counts()}")

# Example: Train/test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_numeric, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ============================================================================
# 4. PREPARE DATA FOR DEEP LEARNING (TensorFlow/PyTorch)
# ============================================================================
print("\n[4] Preparing Data for Deep Learning")
print("-" * 80)

sequences = sequence_data['sequences']
labels = sequence_data['labels']
scaler = sequence_data['scaler']

print(f"Sequences: {sequences.shape}")
print(f"Labels: {labels.shape}")
print(f"Label distribution: {np.bincount(labels.astype(int))}")

# Convert to one-hot encoding for classification
from tensorflow.keras.utils import to_categorical

y_onehot = to_categorical(labels, num_classes=3)
print(f"One-hot encoded shape: {y_onehot.shape}")

# ============================================================================
# 5. FEATURE ANALYSIS & VISUALIZATION
# ============================================================================
print("\n[5] Feature Analysis")
print("-" * 80)

# Top features by variance
numeric_cols = aggregated_df.select_dtypes(include=[np.number]).columns
numeric_cols = [c for c in numeric_cols if c != 'label']

variances_dict = dict(zip(numeric_cols, aggregated_df[numeric_cols].var()))
top_features = sorted(variances_dict.items(), key=lambda x: x[1], reverse=True)[:5]

print("Top 5 features by variance:")
for feature, var in top_features:
    print(f"  {feature:35s}: {var:12,.2f}")

# Label distribution
print(f"\nLabel distribution:")
label_counts = aggregated_df['label'].value_counts().sort_index()
for label_val, count in label_counts.items():
    label_name = {0: 'Human', 1: 'Moderate Bot', 2: 'Advanced Bot'}.get(label_val, 'Unknown')
    pct = (count / len(aggregated_df)) * 100
    print(f"  {label_name:15s} (label={label_val}): {count:4d} ({pct:5.1f}%)")

# ============================================================================
# 6. EXAMPLE: RANDOM FOREST MODEL
# ============================================================================
print("\n[6] Example: Train Random Forest Model")
print("-" * 80)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    # Train model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)
    print("RandomForest model trained!")

    # Evaluate
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 5 important features:")
    print(feature_importance.head())

except ImportError:
    print("[WARN] scikit-learn not installed")

# ============================================================================
# 7. EXAMPLE: VIEW SAMPLE SEQUENCE
# ============================================================================
print("\n[7] Sample Sequence Data")
print("-" * 80)

sample_idx = 0
sample_sequence = sequences[sample_idx]
sample_label = labels[sample_idx]
sample_id = sequence_data['session_ids'][sample_idx]

print(f"Session ID: {sample_id}")
print(f"Label: {sample_label} ({['Human', 'Moderate Bot', 'Advanced Bot'][int(sample_label)]})")
print(f"Sequence shape: {sample_sequence.shape}")
print(f"\nFirst 5 timesteps:")
print(f"{'Time':>5} {'X':>10} {'Y':>10} {'Speed':>10} {'Click':>8} {'Scroll':>8} {'Pos':>8}")
print("-" * 60)

for t in range(min(5, len(sample_sequence))):
    features = sample_sequence[t]
    print(f"{t:5d} {features[0]:10.4f} {features[1]:10.4f} "
          f"{features[2]:10.4f} {features[3]:8.1f} {features[4]:8.1f} {features[5]:8.4f}")

# ============================================================================
# 8. EXAMPLE: LSTM MODEL DEFINITION
# ============================================================================
print("\n[8] Example: LSTM Model Definition")
print("-" * 80)

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # Build LSTM model
    lstm_model = models.Sequential([
        layers.LSTM(64, input_shape=(50, 6), return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    lstm_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("LSTM model architecture:")
    lstm_model.summary()

    print("\nTo train the model:")
    print("  history = lstm_model.fit(")
    print("      sequences, y_onehot,")
    print("      epochs=50, batch_size=8, validation_split=0.2)")

except ImportError:
    print("[WARN] TensorFlow not installed")

# ============================================================================
# 9. FEATURE ENGINEERING TIPS
# ============================================================================
print("\n[9] Feature Engineering Tips for Your Models")
print("-" * 80)

tips = [
    "1. Use StandardScaler for linear models (SVM, Logistic Regression)",
    "2. Tree-based models (RF, XGBoost) don't need scaling",
    "3. Consider PCA for dimensionality reduction if using many features",
    "4. Bot likelihood score may be too informative - consider removing for fairness",
    "5. Categorical features (browser, device) may need encoding for some models",
    "6. Check for high correlation between features (drop redundant ones)",
    "7. Use class weights in loss if dealing with imbalanced data",
    "8. Normalize sequences are already handled - use as-is for LSTM",
    "9. Monitor for data drift by comparing new predictions with training distribution",
    "10. Perform cross-validation to assess generalization performance"
]

for tip in tips:
    print(f"  {tip}")

# ============================================================================
# 10. METADATA INFORMATION
# ============================================================================
print("\n[10] Dataset Metadata")
print("-" * 80)

import json

with open(PREPROCESSING_DIR / 'dataset_metadata_report.json') as f:
    metadata = json.load(f)

print(f"Total sessions: {metadata['dataset_info']['total_sessions']}")
print(f"Total features: {metadata['dataset_info']['total_features']}")
print(f"LSTM sequences: {metadata['lstm_sequences']['total_sequences']}")
print(f"Sequence length: {metadata['lstm_sequences']['sequence_length']}")
print(f"Features per timestep: {metadata['lstm_sequences']['n_features_per_step']}")

print("\n" + "="*80)
print("Ready to build your fraud detection model!")
print("="*80 + "\n")
