"""
COMPREHENSIVE FRAUD DETECTION PREPROCESSING PIPELINE
====================================================

Steps:
1. Data Inspection
2. Label Linkage Validation
3. Record Linkage & Classification
4. Synthetic Device & Network Features
5. Feature Engineering
6. LSTM Sequence Preparation
7. Output Datasets
8. Data Quality Validation
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hashlib

# Configuration
DATASET_DIR = Path("Datasets")
OUTPUT_DIR = Path("preprocessing_output")
OUTPUT_DIR.mkdir(exist_ok=True)

SEQUENCE_LENGTH = 50  # For LSTM input
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# Category to label mapping
CATEGORY_MAPPING = {
    'humans': 0,
    'human': 0,
    'moderate_bot': 1,
    'advanced_bot': 2,
}

print("\n" + "="*80)
print("FRAUD DETECTION PREPROCESSING PIPELINE")
print("="*80)

# ============================================================================
# STEP 1 & 2: DATA INSPECTION + LABEL LINKAGE VALIDATION
# ============================================================================
print("\n[STEP 1-2] Data Inspection & Label Linkage Validation")
print("-" * 80)

def load_full_dataset(filepath, dtype_spec=None):
    """Load full dataset with specified dtypes"""
    print(f"  Loading: {filepath.name}...", end=" ")
    try:
        df = pd.read_csv(filepath, dtype=dtype_spec)
        print(f"OK ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return None

# Load all datasets
behavioral_detailed_adv = load_full_dataset(
    DATASET_DIR / "humans_and_advanced_bots_behavioral_detailed.csv"
)
temporal_detailed_adv = load_full_dataset(
    DATASET_DIR / "humans_and_advanced_bots_temporal_detailed.csv"
)
web_activity_adv = load_full_dataset(
    DATASET_DIR / "humans_and_advanced_bots_web_activity_summary.csv"
)
behavior_summary_adv = load_full_dataset(
    DATASET_DIR / "humans_and_advanced_bots_behavior_summary.csv"
)

behavioral_detailed_mod = load_full_dataset(
    DATASET_DIR / "humans_and_moderate_bots_behavioral_detailed.csv"
)
temporal_detailed_mod = load_full_dataset(
    DATASET_DIR / "humans_and_moderate_bots_temporal_detailed.csv"
)
web_activity_mod = load_full_dataset(
    DATASET_DIR / "humans_and_moderate_bots_web_activity_summary.csv"
)
behavior_summary_mod = load_full_dataset(
    DATASET_DIR / "humans_and_moderate_bots_behavior_summary.csv"
)

# Combine category datasets
print("\n  Combining datasets by category...")

# Extract label from category field
def extract_category_label(category_str):
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

# Add labels to temporal data
if temporal_detailed_adv is not None:
    temporal_detailed_adv['bot_type'] = temporal_detailed_adv['category'].apply(
        extract_category_label
    )
    print(f"    Advanced bots temporal: {len(temporal_detailed_adv)} rows")

if temporal_detailed_mod is not None:
    temporal_detailed_mod['bot_type'] = temporal_detailed_mod['category'].apply(
        extract_category_label
    )
    print(f"    Moderate bots temporal: {len(temporal_detailed_mod)} rows")

if behavioral_detailed_adv is not None:
    behavioral_detailed_adv['bot_type'] = behavioral_detailed_adv['category'].apply(
        extract_category_label
    )
    print(f"    Advanced bots behavioral: {len(behavioral_detailed_adv)} rows")

if behavioral_detailed_mod is not None:
    behavioral_detailed_mod['bot_type'] = behavioral_detailed_mod['category'].apply(
        extract_category_label
    )
    print(f"    Moderate bots behavioral: {len(behavioral_detailed_mod)} rows")

# ============================================================================
# STEP 3: RECORD LINKAGE & SESSION AGGREGATION
# ============================================================================
print("\n[STEP 3] Record Linkage & Session Aggregation")
print("-" * 80)

def aggregate_by_session(behavioral_df, temporal_df, web_activity_df, bot_type_label):
    """Aggregate multiple datasets by session"""

    print(f"  Processing: {bot_type_label}")
    sessions_data = []

    if web_activity_df is None or len(web_activity_df) == 0:
        print(f"    [SKIP] No web activity data")
        return pd.DataFrame()

    for idx, row in web_activity_df.iterrows():
        session_id = row['session_id']

        session_info = {
            'session_id': session_id,
            'bot_type': bot_type_label,
            'label': CATEGORY_MAPPING.get(bot_type_label, 0),
        }

        # Get behavioral features (mouse movements) for this session
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

        # Get web activity features
        session_info['total_requests'] = int(row['total_requests'])
        session_info['successful_requests'] = int(row['successful_requests'])
        session_info['total_response_size'] = int(row['total_response_size'])
        session_info['avg_response_size'] = int(row['avg_response_size'])
        session_info['activity_date'] = row['activity_date']
        session_info['time_range'] = row['time_range']

        # Get temporal features (web logs) for this session
        if temporal_df is not None:
            session_temporal = temporal_df[temporal_df['session_id'] == session_id]
            session_info['logs_count'] = len(session_temporal)

            if len(session_temporal) > 0:
                # Extract timestamps
                timestamps = []
                for ts_str in session_temporal['timestamp'].values:
                    try:
                        # Parse timestamp like: [29/Oct/2019:09:17:51 +0000]
                        ts_clean = ts_str.strip('[]').split('+')[0].strip()
                        dt = datetime.strptime(ts_clean, "%d/%b/%Y:%H:%M:%S")
                        timestamps.append(dt)
                    except:
                        pass

                if len(timestamps) > 1:
                    timestamps = sorted(timestamps)
                    session_info['session_duration_sec'] = (
                        timestamps[-1] - timestamps[0]
                    ).total_seconds()

                    # Calculate request intervals
                    intervals = [
                        (timestamps[i+1] - timestamps[i]).total_seconds()
                        for i in range(len(timestamps)-1)
                    ]
                    session_info['request_interval_mean'] = np.mean(intervals)
                    session_info['request_interval_std'] = np.std(intervals)
                else:
                    session_info['session_duration_sec'] = 0
                    session_info['request_interval_mean'] = 0
                    session_info['request_interval_std'] = 0
            else:
                session_info['session_duration_sec'] = 0
                session_info['request_interval_mean'] = 0
                session_info['request_interval_std'] = 0
        else:
            session_info['logs_count'] = 0
            session_info['session_duration_sec'] = 0
            session_info['request_interval_mean'] = 0
            session_info['request_interval_std'] = 0

        sessions_data.append(session_info)

    df_result = pd.DataFrame(sessions_data)
    print(f"    Generated: {len(df_result)} session records")
    return df_result

# Aggregate advanced bots
sessions_adv = aggregate_by_session(
    behavioral_detailed_adv,
    temporal_detailed_adv,
    web_activity_adv,
    'advanced_bot'
)

# Aggregate moderate bots
sessions_mod = aggregate_by_session(
    behavioral_detailed_mod,
    temporal_detailed_mod,
    web_activity_mod,
    'moderate_bot'
)

# Combine both
if len(sessions_adv) > 0 and len(sessions_mod) > 0:
    sessions_combined = pd.concat([sessions_adv, sessions_mod], ignore_index=True)
    print(f"\n  Total sessions: {len(sessions_combined)}")
elif len(sessions_adv) > 0:
    sessions_combined = sessions_adv
    print(f"\n  Total sessions (advanced only): {len(sessions_combined)}")
else:
    sessions_combined = sessions_mod
    print(f"\n  Total sessions (moderate only): {len(sessions_combined)}")

# ============================================================================
# STEP 4: SYNTHETIC DEVICE & NETWORK FEATURES
# ============================================================================
print("\n[STEP 4] Synthetic Device & Network Features")
print("-" * 80)

BROWSERS = {
    'Chrome': 0.65,
    'Safari': 0.18,
    'Firefox': 0.10,
    'Edge': 0.07,
}

OPERATING_SYSTEMS = {
    'Windows': 0.70,
    'MacOS': 0.15,
    'Linux': 0.10,
    'iOS': 0.03,
    'Android': 0.02,
}

DEVICE_TYPES = {
    'desktop': 0.75,
    'mobile': 0.20,
    'tablet': 0.05,
}

def generate_user_agent(browser, os):
    """Generate realistic user agent string"""
    ua_templates = {
        ('Chrome', 'Windows'): f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        ('Chrome', 'MacOS'): f'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        ('Chrome', 'Linux'): f'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        ('Safari', 'MacOS'): f'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        ('Safari', 'iOS'): f'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1',
        ('Firefox', 'Windows'): f'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        ('Firefox', 'Linux'): f'Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
        ('Edge', 'Windows'): f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59',
    }
    return ua_templates.get((browser, os), f'Mozilla/5.0 ({os}) Default')

def generate_datacenter_ip():
    """Generate datacenter IP range (more likely for bots)"""
    # Common datacenter IP ranges
    ranges = [
        (8, 8),      # Google
        (1, 1),      # APNIC
        (64, 64),    # ARIN
        (128, 128),  # Cogent
        (207, 207),  # Limelight
    ]
    octet1 = np.random.choice([r[0] for r in ranges])
    octet2 = np.random.randint(0, 256)
    octet3 = np.random.randint(0, 256)
    octet4 = np.random.randint(1, 255)
    return f"{octet1}.{octet2}.{octet3}.{octet4}"

def generate_residential_ip():
    """Generate typical residential IP"""
    octet1 = np.random.randint(1, 255)
    octet2 = np.random.randint(0, 256)
    octet3 = np.random.randint(0, 256)
    octet4 = np.random.randint(1, 255)
    return f"{octet1}.{octet2}.{octet3}.{octet4}"

COUNTRIES = ['US', 'GB', 'DE', 'FR', 'CA', 'AU', 'CN', 'IN', 'BR', 'JP']
REGIONS_BY_COUNTRY = {
    'US': ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'],
    'GB': ['London', 'Manchester', 'Birmingham', 'Leeds', 'Glasgow', 'Liverpool'],
    'DE': ['Berlin', 'Munich', 'Frankfurt', 'Cologne', 'Hamburg', 'Dresden'],
    'FR': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice', 'Nantes'],
    'CA': ['ON', 'QC', 'BC', 'AB', 'MB', 'SK', 'NS', 'NB', 'PE', 'NL'],
    'AU': ['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'ACT', 'NT'],
}

def add_device_features(df):
    """Add synthetic device and network features"""
    n_rows = len(df)

    # Browser distribution
    browsers = np.random.choice(
        list(BROWSERS.keys()),
        size=n_rows,
        p=list(BROWSERS.values())
    )
    df['browser'] = browsers

    # OS distribution
    os_list = np.random.choice(
        list(OPERATING_SYSTEMS.keys()),
        size=n_rows,
        p=list(OPERATING_SYSTEMS.values())
    )
    df['operating_system'] = os_list

    # Device type
    device_types = np.random.choice(
        list(DEVICE_TYPES.keys()),
        size=n_rows,
        p=list(DEVICE_TYPES.values())
    )
    df['device_type'] = device_types

    # User agent
    df['user_agent'] = [
        generate_user_agent(b, o)
        for b, o in zip(browsers, os_list)
    ]

    # IP address - bots more likely to use datacenter IPs
    ips = []
    for i, label in enumerate(df['label']):
        if label > 0:  # Bot
            # 70% chance for datacenter IP for bots
            if np.random.random() < 0.7:
                ips.append(generate_datacenter_ip())
            else:
                ips.append(generate_residential_ip())
        else:  # Human
            # 95% chance for residential IP for humans
            if np.random.random() < 0.95:
                ips.append(generate_residential_ip())
            else:
                ips.append(generate_datacenter_ip())
    df['ip_address'] = ips

    # Detect datacenter proxy
    def is_datacenter_proxy(ip):
        try:
            first_octet = int(ip.split('.')[0])
            return first_octet in [8, 1, 64, 128, 207]
        except:
            return False

    df['is_proxy'] = df['ip_address'].apply(is_datacenter_proxy)

    # Country and region
    countries = np.random.choice(COUNTRIES, size=n_rows)
    df['country'] = countries

    regions = [
        np.random.choice(REGIONS_BY_COUNTRY.get(c, ['Region']))
        for c in countries
    ]
    df['region'] = regions

    print(f"  Added device features to {n_rows} sessions")
    print(f"    - Browser distribution: {dict(zip(*np.unique(df['browser'], return_counts=True)))}")
    print(f"    - Datacenter IPs: {df['is_proxy'].sum()}/{n_rows}")

    return df

sessions_combined = add_device_features(sessions_combined)

# ============================================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================================
print("\n[STEP 5] Feature Engineering")
print("-" * 80)

def engineer_features(df):
    """Create derived behavioral and temporal features"""

    # Temporal features
    df['clicks_per_minute'] = (df['total_movements'] /
                               df['session_duration_sec'].replace(0, 1)) * 60
    df['requests_per_minute'] = (df['total_requests'] /
                                 df['session_duration_sec'].replace(0, 1)) * 60

    # Success rate
    df['success_rate'] = (df['successful_requests'] /
                         df['total_requests'].replace(0, 1))

    # Mouse movement features
    def calc_mouse_features(x_coords, y_coords):
        """Calculate mouse movement statistics"""
        if len(x_coords) == 0:
            return {
                'mouse_speed_mean': 0,
                'mouse_speed_std': 0,
                'mouse_path_length': 0,
                'direction_change_count': 0,
                'movement_std': 0,
            }

        # Calculate Euclidean distances between consecutive points
        x_diff = np.diff(x_coords)
        y_diff = np.diff(y_coords)
        distances = np.sqrt(x_diff**2 + y_diff**2)

        # Calculate angles (direction changes)
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
            'mouse_speed_mean': np.mean(distances),
            'mouse_speed_std': np.std(distances),
            'mouse_path_length': np.sum(distances),
            'direction_change_count': len(angles),
            'movement_std': np.std([len(x_coords), np.std(x_coords), np.std(y_coords)]),
        }

    mouse_features = []
    for idx, row in df.iterrows():
        x_coords = row.get('x_coords', np.array([]))
        y_coords = row.get('y_coords', np.array([]))

        if isinstance(x_coords, np.ndarray) and isinstance(y_coords, np.ndarray):
            features = calc_mouse_features(x_coords, y_coords)
        else:
            features = {
                'mouse_speed_mean': 0,
                'mouse_speed_std': 0,
                'mouse_path_length': 0,
                'direction_change_count': 0,
                'movement_std': 0,
            }
        mouse_features.append(features)

    mouse_df = pd.DataFrame(mouse_features)
    for col in mouse_df.columns:
        df[col] = mouse_df[col]

    # Entropy features (for randomness)
    def calc_coordinate_entropy(coords):
        """Calculate entropy of coordinate distribution"""
        if len(coords) == 0:
            return 0

        # Bin coordinates into buckets
        n_bins = max(10, len(np.unique(coords)) // 4)
        hist, _ = np.histogram(coords, bins=n_bins)
        hist = hist[hist > 0]  # Remove zero bins
        probs = hist / np.sum(hist)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy

    entropies = []
    for idx, row in df.iterrows():
        x_coords = row.get('x_coords', np.array([]))
        y_coords = row.get('y_coords', np.array([]))

        if isinstance(x_coords, np.ndarray) and isinstance(y_coords, np.ndarray):
            x_entropy = calc_coordinate_entropy(x_coords)
            y_entropy = calc_coordinate_entropy(y_coords)
            avg_entropy = (x_entropy + y_entropy) / 2
        else:
            avg_entropy = 0

        entropies.append(avg_entropy)

    df['coordinate_entropy'] = entropies

    # Bot likelihood score (heuristic)
    def calc_bot_score(row):
        """Heuristic score for bot-like behavior"""
        score = 0

        # High request frequency
        if row['requests_per_minute'] > 100:
            score += 2
        elif row['requests_per_minute'] > 50:
            score += 1

        # Low coordinate entropy (repetitive patterns)
        if row['coordinate_entropy'] < 2:
            score += 2
        elif row['coordinate_entropy'] < 3:
            score += 1

        # High consistency in mouse movement
        if row['mouse_speed_std'] < 10:
            score += 1

        # Datacenter IP
        if row['is_proxy']:
            score += 1

        # Very long sessions with many requests
        if row['session_duration_sec'] > 3600 and row['total_requests'] > 1000:
            score += 1

        return score

    df['bot_likelihood_score'] = df.apply(calc_bot_score, axis=1)

    print(f"  Engineered {len(mouse_df.columns)} mouse features")
    print(f"  Engineered temporal and entropy features")
    print(f"  Calculated bot likelihood scores")

    return df

sessions_combined = engineer_features(sessions_combined)

# Remove coordinate arrays before saving (not needed for aggregated dataset)
sessions_combined = sessions_combined.drop(columns=['x_coords', 'y_coords'], errors='ignore')

print(f"\n  Feature summary:")
print(f"    Rows: {len(sessions_combined)}")
print(f"    Columns: {len(sessions_combined.columns)}")
print(f"    Label distribution:\n{sessions_combined['label'].value_counts().to_string()}")

# ============================================================================
# STEP 6: TEMPORAL DATA PREPARATION FOR LSTM
# ============================================================================
print("\n[STEP 6] Temporal Data Preparation for LSTM")
print("-" * 80)

def prepare_lstm_sequences(behavioral_df, temporal_df, web_activity_df, sessions_df, bot_type):
    """Prepare sequence data for LSTM training"""

    sequences = []
    sequence_labels = []
    sequence_ids = []

    if behavioral_df is None or len(behavioral_df) == 0:
        return sequences, sequence_labels, sequence_ids

    # Get all sessions for this bot type
    sessions_for_type = sessions_df[sessions_df['bot_type'] == bot_type]['session_id'].values

    for session_id in sessions_for_type:
        # Get behavioral data for this session
        session_behaviors = behavioral_df[behavioral_df['session_id'] == session_id]

        if len(session_behaviors) < 2:
            continue

        # Get label for this session
        label = CATEGORY_MAPPING.get(bot_type, 0)

        # Create sequence of movements
        sequence_features = []
        x_coords = session_behaviors['x_coordinate'].values
        y_coords = session_behaviors['y_coordinate'].values

        for i in range(len(x_coords)):
            # Calculate movement speed from previous position
            if i > 0:
                movement_speed = np.sqrt((x_coords[i] - x_coords[i-1])**2 +
                                        (y_coords[i] - y_coords[i-1])**2)
            else:
                movement_speed = 0

            # Normalize coordinates to [0, 1]
            x_norm = x_coords[i] / 1600 if x_coords[i] != 0 else 0
            y_norm = y_coords[i] / 900 if y_coords[i] != 0 else 0

            # Create feature vector
            features = np.array([
                x_norm,
                y_norm,
                movement_speed / 100,  # Normalize speed
                1,  # Click flag (presence of movement)
                0,  # Scroll flag (not available in this data)
                i / max(len(x_coords), 1),  # Temporal position (0-1)
            ])

            sequence_features.append(features)

        # Pad or truncate to SEQUENCE_LENGTH
        if len(sequence_features) < SEQUENCE_LENGTH:
            # Pad with zeros
            padding_size = SEQUENCE_LENGTH - len(sequence_features)
            padding = np.zeros((padding_size, len(sequence_features[0])))
            sequence_features = np.vstack([sequence_features, padding])
        else:
            # Truncate
            sequence_features = np.array(sequence_features[:SEQUENCE_LENGTH])

        sequences.append(sequence_features)
        sequence_labels.append(label)
        sequence_ids.append(session_id)

    return np.array(sequences), np.array(sequence_labels), sequence_ids

print("  Preparing LSTM sequences for advanced bots...")
seq_adv, labels_adv, ids_adv = prepare_lstm_sequences(
    behavioral_detailed_adv, temporal_detailed_adv, web_activity_adv,
    sessions_combined, 'advanced_bot'
)

print("  Preparing LSTM sequences for moderate bots...")
seq_mod, labels_mod, ids_mod = prepare_lstm_sequences(
    behavioral_detailed_mod, temporal_detailed_mod, web_activity_mod,
    sessions_combined, 'moderate_bot'
)

# Combine sequences
if len(seq_adv) > 0 and len(seq_mod) > 0:
    sequences_all = np.vstack([seq_adv, seq_mod])
    labels_all = np.concatenate([labels_adv, labels_mod])
    ids_all = ids_adv + ids_mod
elif len(seq_adv) > 0:
    sequences_all = seq_adv
    labels_all = labels_adv
    ids_all = ids_adv
else:
    sequences_all = seq_mod
    labels_all = labels_mod
    ids_all = ids_mod

print(f"  Total sequences prepared: {len(sequences_all)}")
print(f"  Sequence shape: {sequences_all.shape}")
print(f"  Sequences: {len(sequences_all)}, Sequence length: {SEQUENCE_LENGTH}, Features: 6")

# Normalize sequences
scaler = StandardScaler()
sequences_reshaped = sequences_all.reshape(-1, sequences_all.shape[-1])
sequences_normalized = scaler.fit_transform(sequences_reshaped)
sequences_all = sequences_normalized.reshape(sequences_all.shape)

print(f"  Sequences normalized using StandardScaler")

# ============================================================================
# STEP 7: OUTPUT DATASETS
# ============================================================================
print("\n[STEP 7] Output Datasets")
print("-" * 80)

# Save aggregated session dataset
output_file = OUTPUT_DIR / "session_aggregated_dataset.csv"
sessions_combined.to_csv(output_file, index=False)
print(f"  [OK] Saved: {output_file}")

# Save LSTM sequence dataset
sequence_data = {
    'sequences': sequences_all,
    'labels': labels_all,
    'session_ids': ids_all,
    'scaler': scaler,
    'sequence_length': SEQUENCE_LENGTH,
    'n_features': sequences_all.shape[2] if len(sequences_all.shape) > 2 else 0,
}

output_file = OUTPUT_DIR / "session_sequence_dataset.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(sequence_data, f)
print(f"  [OK] Saved: {output_file}")

# ============================================================================
# STEP 8: DATA QUALITY VALIDATION
# ============================================================================
print("\n[STEP 8] Data Quality Validation")
print("-" * 80)

# Missing values
print("  Missing values:")
missing_counts = sessions_combined.isnull().sum()
if missing_counts.sum() > 0:
    print(f"    {missing_counts[missing_counts > 0].to_string()}")
else:
    print("    None found (Good!)")

# Outlier detection using IQR
numeric_cols = sessions_combined.select_dtypes(include=[np.number]).columns
outliers_dict = {}

for col in numeric_cols:
    Q1 = sessions_combined[col].quantile(0.25)
    Q3 = sessions_combined[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    n_outliers = ((sessions_combined[col] < lower_bound) |
                  (sessions_combined[col] > upper_bound)).sum()

    if n_outliers > 0:
        outliers_dict[col] = n_outliers

print(f"  Outliers detected (IQR method):")
if len(outliers_dict) > 0:
    for col, count in sorted(outliers_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
        pct = (count / len(sessions_combined)) * 100
        print(f"    - {col}: {count} ({pct:.1f}%)")
else:
    print("    None found")

# Label balance
print(f"\n  Label balance:")
label_dist = sessions_combined['label'].value_counts().sort_index()
for label_val, count in label_dist.items():
    label_name = {0: 'Human', 1: 'Moderate Bot', 2: 'Advanced Bot'}.get(label_val, 'Unknown')
    pct = (count / len(sessions_combined)) * 100
    print(f"    {label_name} (label={label_val}): {count} ({pct:.1f}%)")

# Feature statistics
print(f"\n  Feature statistics (numeric columns):")
print(f"    Mean values:")
for col in numeric_cols[:5]:
    print(f"      {col}: {sessions_combined[col].mean():.4f}")
print(f"    ... and {len(numeric_cols) - 5} more features")

# ============================================================================
# METADATA REPORT
# ============================================================================
print("\n  Generating metadata report...")

metadata = {
    'created_at': datetime.now().isoformat(),
    'dataset_info': {
        'total_sessions': len(sessions_combined),
        'total_features': len(sessions_combined.columns),
        'label_distribution': sessions_combined['label'].value_counts().to_dict(),
        'label_mapping': {'0': 'human', '1': 'moderate_bot', '2': 'advanced_bot'},
    },
    'lstm_sequences': {
        'total_sequences': len(sequences_all),
        'sequence_length': SEQUENCE_LENGTH,
        'n_features_per_step': sequences_all.shape[2] if len(sequences_all.shape) > 2 else 0,
        'features': ['x_norm', 'y_norm', 'movement_speed', 'click_flag', 'scroll_flag', 'temporal_pos'],
    },
    'feature_columns': list(sessions_combined.columns),
    'categorical_features': [
        'session_id', 'bot_type', 'browser', 'operating_system', 'device_type',
        'ip_address', 'country', 'region',
    ],
    'numeric_features': list(numeric_cols),
    'synthetic_features': [
        'user_agent', 'browser', 'operating_system', 'device_type',
        'ip_address', 'country', 'region', 'is_proxy',
    ],
    'engineered_features': [
        'clicks_per_minute', 'requests_per_minute', 'success_rate',
        'mouse_speed_mean', 'mouse_speed_std', 'mouse_path_length',
        'direction_change_count', 'movement_std', 'coordinate_entropy',
        'bot_likelihood_score',
    ],
    'data_quality': {
        'missing_values_count': int(missing_counts.sum()),
        'outliers_detected': len(outliers_dict),
        'label_balance': 'Moderate' if (len(label_dist) > 1 and max(label_dist) / min(label_dist) < 3) else ('Balanced' if len(label_dist) == 1 else 'Imbalanced'),
    },
    'preprocessing_config': {
        'random_seed': RANDOM_SEED,
        'sequence_length': SEQUENCE_LENGTH,
        'source_datasets': list(DATASET_DIR.glob('*.csv')),
    },
}

metadata_file = OUTPUT_DIR / "dataset_metadata_report.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"  [OK] Saved: {metadata_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PREPROCESSING COMPLETE")
print("="*80)
print(f"\nOutput files generated in: {OUTPUT_DIR.absolute()}")
print(f"  1. session_aggregated_dataset.csv ({len(sessions_combined)} sessions)")
print(f"  2. session_sequence_dataset.pkl ({len(sequences_all)} sequences)")
print(f"  3. dataset_metadata_report.json")
print(f"\nReady for model training!")
print("="*80 + "\n")
