import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, TimeDistributed

# ==========================================
# 1. LOAD DATA FROM DISK
# ==========================================
# Make sure the CSV file is in the same directory as your Python script/notebook
df = pd.read_csv('click_fraud_dataset.csv')

# ==========================================
# 2. DATA PREPROCESSING & SORTING
# ==========================================
print("Starting Preprocessing...")
# Sort chronologically by IP
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by=['ip_address', 'timestamp'])

# Define feature categories based on your schema
categorical_cols = ['device_type', 'browser', 'operating_system', 'ad_position', 'device_ip_reputation']
numerical_cols = ['click_duration', 'scroll_depth', 'mouse_movement', 'keystrokes_detected', 
                  'click_frequency', 'time_since_last_click', 'VPN_usage', 'proxy_usage', 'bot_likelihood_score']

# Encode Categoricals
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Convert to string to prevent errors with mixed types or nulls
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Scale Numericals (Crucial for Neural Networks)
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

feature_cols = categorical_cols + numerical_cols
num_features = len(feature_cols)

# ==========================================
# 3. SEQUENCE CREATION & PADDING
# ==========================================
sequences = []
labels = []

# Group by IP address
grouped = df.groupby('ip_address')

for ip, group in grouped:
    seq_features = group[feature_cols].values
    seq_labels = group['is_fraudulent'].values 
    
    sequences.append(seq_features)
    labels.append(seq_labels)

# Pad sequences to exactly 10 timesteps (Adjust this based on your actual data's average)
MAX_TIMESTEPS = 10 

# Pad features (X)
X_padded = pad_sequences(sequences, maxlen=MAX_TIMESTEPS, padding='post', dtype='float32')

# Pad labels (y) to match the feature length
y_padded = pad_sequences(labels, maxlen=MAX_TIMESTEPS, padding='post', dtype='float32')

# Keras LSTMs require 3D targets for sequence output: (Samples, Timesteps, 1)
y_padded = np.expand_dims(y_padded, axis=-1)

print(f"Total Unique IPs (Sequences): {len(sequences)}")
print(f"X shape: {X_padded.shape} -> (Samples, Timesteps, Features)")
print(f"y shape: {y_padded.shape} -> (Samples, Timesteps, 1)")

# ==========================================
# 4. LSTM MODEL ARCHITECTURE
# ==========================================
print("\nBuilding Model...")
model = Sequential([
    # Masking layer ignores the 0.0 padding values during training
    Masking(mask_value=0.0, input_shape=(MAX_TIMESTEPS, num_features)),
    
    # LSTM layer with return_sequences=True to output at every timestep
    LSTM(64, return_sequences=True, dropout=0.2),
    
    # TimeDistributed applies the final Dense calculation to every timestep independently
    TimeDistributed(Dense(1, activation='sigmoid'))
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

# ==========================================
# 5. TRAINING
# ==========================================
print("\nStarting Training...")
# Using a validation split to monitor overfitting
history = model.fit(
    X_padded, 
    y_padded, 
    epochs=15,          
    batch_size=32, 
    validation_split=0.2,
    verbose=1
)