import sqlite3
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss, classification_report
import numpy as np
from typing import Optional, Tuple, List
import os
from tqdm import tqdm
import random
# --- Keras/TensorFlow Imports ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib # Can be removed if all plot labels are in English

# --- Seed Fixing (for reproducibility) ---
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1' # Enhance determinism of TensorFlow operations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Turn off oneDNN custom operations (based on info message)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Set seeds for execution
set_seeds()

# Pandas display settings
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)

"""## 2: Data Loading and Preprocessing Function Definition (load_and_process_pbp)"""

def load_and_process_pbp(db_path: str, limit_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Loads play-by-play data from an SQLite database, processes data types,
    calculates home team win information, and merges it with the original data.
    (Supports chunk loading and progress bar display)
    """
    print(f"--- Starting data loading and processing ---")
    print(f"Database path: {db_path}")
    if limit_rows:
        print(f"Row limit: {limit_rows}")
    else:
        print("Row limit: None (loading all rows)")

    try:
        print("Connecting to database...")
        conn = sqlite3.connect(db_path)
        print("Database connection successful.")

        total_rows_to_load = limit_rows
        if total_rows_to_load is None:
            print("Determining total row count for progress bar...")
            count_query = "SELECT COUNT(*) FROM play_by_play;"
            total_rows_to_load = pd.read_sql_query(count_query, conn).iloc[0, 0]
            print(f"Total rows in table: {total_rows_to_load}")

        query = """
        SELECT
            game_id, eventnum, eventmsgtype, eventmsgactiontype,
            period, pctimestring,
            homedescription, neutraldescription, visitordescription,
            score, scoremargin
        FROM
            play_by_play
        """
        if limit_rows:
            query += f" LIMIT {limit_rows};"
        else:
            query += ";"

        chunk_size = 100000
        print(f"Loading data in chunks of {chunk_size} rows...")
        iterator = pd.read_sql_query(query, conn, chunksize=chunk_size)
        num_chunks = (total_rows_to_load + chunk_size - 1) // chunk_size

        list_of_dfs = []
        for chunk_df in tqdm(iterator, total=num_chunks, desc="Loading data"):
            list_of_dfs.append(chunk_df)

        print("\nConcatenating all loaded chunks...")
        df = pd.concat(list_of_dfs, ignore_index=True)
        print(f"Successfully loaded {len(df)} rows into DataFrame.")

        conn.close()
        print("Database connection closed.")

        print("Processing data types...")
        df['game_id'] = df['game_id'].astype(str)
        df['pctimestring'] = df['pctimestring'].astype(str)
        df['score'] = df['score'].astype(str)
        df['scoremargin'] = df['scoremargin'].astype(str)
        df['eventnum'] = pd.to_numeric(df['eventnum'], errors='coerce')
        df['eventmsgtype'] = pd.to_numeric(df['eventmsgtype'], errors='coerce')
        df['eventmsgactiontype'] = pd.to_numeric(df['eventmsgactiontype'], errors='coerce')
        df['period'] = pd.to_numeric(df['period'], errors='coerce')
        desc_cols = ['homedescription', 'neutraldescription', 'visitordescription']
        for col in desc_cols:
            df[col] = df[col].fillna('')
        print("Data type processing complete.")

        print("\nAttempting to determine game outcomes...")
        game_outcomes = pd.Series(dtype=int)
        end_game_events = df[(df['eventmsgtype'] == 13) & (df['score'].str.contains(' - ', na=False))].copy()

        if not end_game_events.empty:
            end_game_events = end_game_events.dropna(subset=['period'])
            if not end_game_events.empty:
                end_game_events['period'] = end_game_events['period'].astype(int)
                final_events = end_game_events.sort_values('period').groupby('game_id').last()

                if 'score' in final_events.columns:
                    scores_split = final_events['score'].str.split(' - ', expand=True)
                    scores_split.columns = ['home_score', 'visitor_score']
                    scores_split['home_score'] = pd.to_numeric(scores_split['home_score'], errors='coerce')
                    scores_split['visitor_score'] = pd.to_numeric(scores_split['visitor_score'], errors='coerce')
                    scores_split = scores_split.dropna(subset=['home_score', 'visitor_score'])

                    if not scores_split.empty:
                        scores_split['home_win'] = (scores_split['home_score'] > scores_split['visitor_score']).astype(int)
                        game_outcomes = scores_split['home_win']
                        print(f"Determined outcomes for {len(game_outcomes)} games.")
                        if limit_rows:
                            print("[Warning] Game outcomes may be incomplete due to the row limit.")

        print("Merging game outcomes back to the main DataFrame...")
        if not game_outcomes.empty:
             df_with_outcome = df.merge(game_outcomes.rename('home_win'), on='game_id', how='left')
        else:
             print("No game outcomes determined, adding 'home_win' column with NaN.")
             df['home_win'] = pd.NA
             df_with_outcome = df

        print("--- Data loading and processing finished ---")
        return df_with_outcome

    except sqlite3.Error as e:
        print(f"\n--- Database Error ---")
        print(f"An error occurred while interacting with the database: {e}")
        return None
    except FileNotFoundError:
        print(f"\n--- File Not Found Error ---")
        print(f"Error: The database file was not found at the specified path: {db_path}")
        return None
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error details: {e}")
        return None

"""## 3: Feature Engineering Function Definitions"""

def parse_time_to_seconds(time_str: str) -> Optional[int]:
    if isinstance(time_str, str) and ':' in time_str:
        try:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds
        except ValueError:
            return None
    return None

def calculate_seconds_elapsed(row: pd.Series) -> Optional[float]:
    period = row['period']
    pctimestring = row['pctimestring']
    if pd.isna(period) or period < 1:
        return None
    seconds_in_period = parse_time_to_seconds(pctimestring)
    if seconds_in_period is None:
        return None
    seconds_per_period = 720 if period <= 4 else 300 # 12 min for regulation, 5 min for OT
    seconds_elapsed_in_current_period = seconds_per_period - seconds_in_period
    if seconds_elapsed_in_current_period < 0 or seconds_elapsed_in_current_period > seconds_per_period:
         return None # Should not happen with valid pctimestring
    if period <= 4: # Regulation periods 1-4
        total_seconds_elapsed = (period - 1) * 720 + seconds_elapsed_in_current_period
    else: # Overtime periods (period 5 is OT1, period 6 is OT2, etc.)
        total_seconds_elapsed = 4 * 720 + (period - 5) * 300 + seconds_elapsed_in_current_period
    return total_seconds_elapsed

def process_score_margin(margin_str: str) -> Optional[int]:
    if margin_str == 'TIE':
        return 0
    elif pd.isna(margin_str) or margin_str == '':
         return None
    else:
        try:
            return int(str(margin_str).replace('+', ''))
        except ValueError:
            return None

"""## 4: Configuration and Data Loading Execution"""
# Please modify the path according to your environment
db_file = r'C:\Users\amilu\Projects\vsCodeFile\PBL\nba.sqlite' # Path for lab PC
# db_file = r'C:\Programing\PBL\nba.sqlite' # Path for laptop PC

# Set the number of rows to process (None for all data)
# limit_rows = 500000 # About 500,000 rows are recommended for operational checks
#limit_rows = 45616 # Original code value
limit_rows = 225109
# limit_rows = None

df_processed = load_and_process_pbp(db_file, limit_rows=limit_rows)

if df_processed is not None:
    print("\nShape of loaded data:", df_processed.shape)
else:
    print("Data loading failed.")

"""## 5: Feature Engineering Execution"""
if df_processed is not None:
    print("\n--- Feature Engineering ---")
    print("Calculating total seconds elapsed...")
    df_processed['seconds_elapsed'] = df_processed.apply(calculate_seconds_elapsed, axis=1)
    print("Processing score margin...")
    df_processed['numeric_score_margin'] = df_processed['scoremargin'].apply(process_score_margin)
    print("Forward filling missing 'numeric_score_margin' within each game...")
    df_processed = df_processed.sort_values(by=['game_id', 'eventnum'])
    df_processed['numeric_score_margin'] = df_processed.groupby('game_id')['numeric_score_margin'].ffill()
    print("Generating composite event ID...")
    df_processed['composite_event_id'] = (df_processed['eventmsgtype'] * 1000 + df_processed['eventmsgactiontype'])
    print("Feature Engineering complete.")

"""## 6: Data Preparation for Model (Filtering)"""
if df_processed is not None:
    print("\n--- Data Preparation for Modeling ---")
    initial_rows = len(df_processed)
    model_df = df_processed.dropna(subset=['home_win', 'seconds_elapsed', 'numeric_score_margin', 'period', 'composite_event_id']).copy()
    model_df = model_df[model_df['period'] > 0]
    model_df = model_df[model_df['eventmsgtype'] != 12] # Exclude "Start Period" events
    filtered_rows = len(model_df)
    print(f"Rows before filtering: {initial_rows}")
    print(f"Rows after filtering invalid/unnecessary entries: {filtered_rows}")
    if filtered_rows == 0:
        print("No data left after filtering. Exiting.")
        model_df = pd.DataFrame() # Make it empty and skip subsequent processing

"""## 7: Feature and Target Selection, Train/Test Split"""
if 'model_df' in locals() and not model_df.empty:
    print("Applying Label Encoding to 'composite_event_id'...")
    le = LabelEncoder()
    model_df['composite_event_id'] = le.fit_transform(model_df['composite_event_id'])
    
    features = ['numeric_score_margin', 'seconds_elapsed', 'composite_event_id']
    target = 'home_win'
    print(f"\nSelected features: {features}")

    X = model_df[features]
    y = model_df[target].astype(int)
    groups = model_df['game_id']

    print("\nSplitting data into training and testing sets (game-aware)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    if groups.nunique() < 2:
        print("Warning: Not enough unique games for GroupShuffleSplit. Using regular split.")
        train_idx, test_idx = train_test_split(range(len(X)), test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)
    else:
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]
    
    # Keep the original DataFrame of the test data to merge prediction results later
    model_df_test = model_df.iloc[test_idx].copy()

    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    print(f"Training games: {groups_train.nunique()}")
    print(f"Testing games: {groups_test.nunique()}")
else:
    print("Skipping feature selection and train/test split as model_df is not available or empty.")

"""## 8: Data Transformation for LSTM (Scaling, Padding)"""
if 'X_train' in locals() and not X_train.empty:
    print("\n--- Preparing Data for LSTM ---")
    
    # 1. Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled data back to DataFrame to merge with game_id
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=features)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=features)
    
    # 2. Group by game and create sequences
    def create_padded_sequences(X_scaled_df: pd.DataFrame, y_series: pd.Series, groups_series: pd.Series) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Group by game, pad sequences, and return NumPy arrays"""
        game_ids = groups_series.unique()
        X_sequences = [X_scaled_df[groups_series == game_id].values for game_id in game_ids]
        y_sequences = [y_series[groups_series == game_id].values for game_id in game_ids]
        
        # Keep original sequence lengths (to exclude padding during evaluation)
        original_lengths = [len(seq) for seq in X_sequences]
        
        # Padding
        X_padded = pad_sequences(X_sequences, padding='pre', dtype='float32')
        y_padded = pad_sequences(y_sequences, padding='pre', value=-1) # Pad target with -1
        
        # Reshape target to (samples, timesteps, 1)
        y_padded = np.expand_dims(y_padded, -1)
        
        return X_padded, y_padded, original_lengths

    print("Creating padded sequences for train and test sets...")
    X_train_scaled_lstm, y_train_lstm, _ = create_padded_sequences(X_train_scaled_df, y_train, groups_train)
    X_test_scaled_lstm, y_test_lstm, test_original_lengths = create_padded_sequences(X_test_scaled_df, y_test, groups_test)
    
    print(f"Shape of X_train for LSTM: {X_train_scaled_lstm.shape}")
    print(f"Shape of y_train for LSTM: {y_train_lstm.shape}")
    print(f"Shape of X_test for LSTM: {X_test_scaled_lstm.shape}")
    print(f"Shape of y_test for LSTM: {y_test_lstm.shape}")
else:
    print("Skipping LSTM data preparation as training data is not available.")


"""## 9: Model Definition and Compilation"""
if 'X_train_scaled_lstm' in locals():
    print("\n--- Defining and Compiling Many-to-Many LSTM Model ---")
    model = Sequential([
        LSTM(32, input_shape=(X_train_scaled_lstm.shape[1], X_train_scaled_lstm.shape[2]), return_sequences=True),
        Dropout(0.3),
        TimeDistributed(Dense(16, activation='relu')),
        Dropout(0.3),
        TimeDistributed(Dense(1, activation='sigmoid'))
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
else:
    print("Skipping model definition as LSTM training data is not available.")

"""## 10: Model Training"""
if 'model' in locals():
    print("\n--- Training LSTM Model ---")
    # Set shuffle argument of model.fit to False
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train_scaled_lstm, y_train_lstm,
                        epochs=50,
                        batch_size=16,
                        validation_split=0.2,
                        shuffle=False, # Disable shuffling for each epoch
                        callbacks=[early_stopping])
    print("Model training complete.")
else:
    print("Skipping model training as the model is not defined.")

"""## 11: Prediction and Evaluation"""
if 'model' in locals() and 'X_test_scaled_lstm' in locals() and X_test_scaled_lstm.size > 0:
    print("\n--- Prediction and Evaluation with Many-to-Many LSTM ---")
    y_pred_proba_3d = model.predict(X_test_scaled_lstm)

    # Exclude padding and convert predicted and true values to flat lists
    y_pred_flat = []
    y_true_flat = []
    for i, length in enumerate(test_original_lengths):
        # Get only the original length from the end of the sequence
        valid_preds = y_pred_proba_3d[i, -length:, 0]
        valid_true = y_test_lstm[i, -length:, 0]
        y_pred_flat.extend(valid_preds)
        y_true_flat.extend(valid_true)

    if y_true_flat and y_pred_flat:
        y_pred_class_flat = (np.array(y_pred_flat) > 0.5).astype(int)
        accuracy = accuracy_score(y_true_flat, y_pred_class_flat)
        auc = roc_auc_score(y_true_flat, y_pred_flat)
        logloss = log_loss(y_true_flat, y_pred_flat)
        brier = brier_score_loss(y_true_flat, y_pred_flat)

        print(f"\nLSTM Model Evaluation Results (Per Play):")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  ROC AUC:     {auc:.4f}")
        print(f"  Log Loss:    {logloss:.4f}")
        print(f"  Brier Score: {brier:.4f}")

        # Add prediction results to the original test DataFrame
        if len(y_pred_flat) == len(model_df_test):
            model_df_test['win_probability_pred'] = y_pred_flat
            print("\nFirst 15 rows of test data with time-varying win probability:")
            display_cols = ['game_id', 'eventnum', 'period', 'pctimestring', 'numeric_score_margin', 'home_win', 'win_probability_pred']
            print(model_df_test[display_cols].head(15))
        else:
            print(f"Warning: Length mismatch when adding predictions. Predictions: {len(y_pred_flat)}, Test set: {len(model_df_test)}")
    else:
        print("No data to evaluate.")
else:
    print("Skipping prediction and evaluation as model or test data is not available.")
    model_df_test = pd.DataFrame() # Define as empty for subsequent processing

"""## 12: CSV Output"""
if 'model_df_test' in locals() and not model_df_test.empty:
    num_games_to_export = 3
    unique_test_games = model_df_test['game_id'].unique()
    if len(unique_test_games) > 0:
        games_to_export = unique_test_games[:num_games_to_export]
        df_for_export = model_df_test[model_df_test['game_id'].isin(games_to_export)]
        print(f"\nExporting data for {len(games_to_export)} games...")
        output_csv_path = f'lstm_predictions_{len(games_to_export)}_games.csv'
        try:
            df_for_export.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"Successfully exported test predictions to: {output_csv_path}")
        except Exception as e:
            print(f"An error occurred while exporting to CSV: {e}")
    else:
        print("\nNo games found in the test set to export.")
else:
    print("\nSkipping CSV export because the final test DataFrame ('model_df_test') is not available or is empty.")

"""## 13: Extracting Momentum (WPA Analysis)"""
if 'model_df_test' in locals() and not model_df_test.empty and 'le' in locals():
    print("\n--- Calculating Win Probability Added (WPA) for each event ---")
    
    event_id_to_name_map = {
        # --- Base Categories (Fallback for IDs not in detailed data) ---
        1000: 'Made FG (Other)', 
        2000: 'Missed FG (Other)', 
        3000: 'Free Throw (Other)', 
        4000: 'Rebound',
        5000: 'Turnover (Other)', 
        6000: 'Foul (Other)', 
        7000: 'Violation', 
        8000: 'Substitution',
        9000: 'Timeout', 
        10000: 'Jump Ball', 
        11000: 'Ejection', 
        12000: 'Start of Period',
        13000: 'End of Period', 
        18000: 'Other',

        # --- Detailed Actions (Reflecting current analysis) ---
        # Made Shot (Type 1)
        1001: 'Made Jump Shot',
        1002: 'Made Running/Floater Shot',
        1003: 'Made Slam Dunk',
        1005: 'Made Layup',
        1006: 'Made Hook Shot',
        1007: 'Made Tip-in/Alley-oop',

        # Missed Shot (Type 2)
        2001: 'Missed Jump Shot',
        2002: 'Missed Running Shot',
        2005: 'Missed Layup',
        2006: 'Missed Hook Shot',

        # Free Throw (Type 3)
        3010: 'Made Free Throw (1st/2nd)',
        3011: 'Made Free Throw (2nd/3rd)',
        3012: 'Missed Free Throw',
        3013: 'Made Technical Free Throw',

        # Turnover (Type 5)
        5001: 'Steal',
        5002: 'Lost Ball/Bad Pass',
        5004: 'Offensive Foul/Violation',

        # Foul (Type 6)
        6001: 'Personal Foul',
        6002: 'Shooting Foul',
        6003: 'Offensive Charge Foul'
    }

    # Get original composite ID (1001, 1005...) from LabelEncoder's encoded values (0,1,2...)
    encoded_label_to_original_id_map = {i: original_id for i, original_id in enumerate(le.classes_)}
    # Create a mapping dictionary from encoded values to event names
    encoded_label_to_name_map = {
        encoded_label: event_id_to_name_map.get(original_id, f'Unknown ID ({original_id})')
        for encoded_label, original_id in encoded_label_to_original_id_map.items()
    }

    wpa_df = model_df_test.copy()
    wpa_df['win_prob_before'] = wpa_df.groupby('game_id')['win_probability_pred'].shift(1)
    wpa_df['wpa_impact'] = wpa_df['win_probability_pred'] - wpa_df['win_prob_before']
    wpa_df_final = wpa_df.dropna(subset=['wpa_impact', 'composite_event_id'])

    event_impact = wpa_df_final.groupby('composite_event_id')['wpa_impact'].agg(['mean', 'count'])
    event_impact['event_name'] = event_impact.index.map(encoded_label_to_name_map)
    event_impact = event_impact.sort_values(by='mean', ascending=False)

    print("\n--- Detailed WPA Impact per Event (Top 15) ---")
    print("\n[Events that most increased win probability (Home Team)]")
    print(event_impact.head(15))
    print("\n[Events that most decreased win probability (Home Team)]")
    print(event_impact.tail(15).sort_values(by='mean'))

    event_impact_filtered = event_impact[event_impact['count'] >= 5]
    if not event_impact_filtered.empty:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(1, 2, figsize=(20, 12))
        fig.suptitle('Average WPA Impact per Detailed Event (Occurrences >= 5)', fontsize=18)
        top_events = event_impact_filtered.head(15)
        sns.barplot(ax=axes[0], x=top_events['mean'], y=top_events['event_name'], hue=top_events['event_name'], palette='Greens_r', legend=False)
        axes[0].set_title('Top 15 Events Increasing Win Probability')
        axes[0].set_xlabel('Average WPA Impact (Win Probability Increase)')
        axes[0].set_ylabel('Event Name')
        bottom_events = event_impact_filtered.tail(15).sort_values(by='mean', ascending=True)
        sns.barplot(ax=axes[1], x=bottom_events['mean'], y=bottom_events['event_name'], hue=bottom_events['event_name'], palette='Reds', legend=False)
        axes[1].set_title('Top 15 Events Decreasing Win Probability')
        axes[1].set_xlabel('Average WPA Impact (Win Probability Decrease)')
        axes[1].set_ylabel('') # No y-axis label for the second plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    else:
        print("\nNo events found with count >= 5 for plotting.")
else:
    print("\nSkipping WPA analysis because the final test DataFrame ('model_df_test') is not available or is empty.")

print("\n--- Win probability prediction script execution finished ---")