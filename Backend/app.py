# app.py - MODIFIED VERSION 2
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Initialize global variables
model = None
model_barber = None
model_sonoma = None
model_road_america = None
model_circuit_of_the_americas = None
model_virginia_international_raceway = None
db = None
db_barber = None
db_sonoma = None
db_road_america = None
db_circuit_of_the_americas = None
db_virginia_international_raceway = None

# --- LOAD BOTH THE MODEL AND THE DATA ---
try:
    # 1. Load the trained model pipeline
    model_barber = joblib.load('barber_overall_model.pkl') 

    # 2. Load the NEW engineered data
    db_barber = pd.read_parquet('barber_features_engineered.parquet')
    
    # 3. Handle duplicates
    db_barber = db_barber.drop_duplicates(subset=['meta_session', 'vehicle_id', 'lap'], keep='first')

    model_sonoma = joblib.load('sonoma_overall_model.pkl') 
    db_sonoma = pd.read_parquet('sonoma_features_engineered.parquet')
    db_sonoma = db_sonoma.drop_duplicates(subset=['meta_session', 'vehicle_id', 'lap'], keep='first')

    model_road_america = joblib.load('road_america_overall_model.pkl')
    db_road_america = pd.read_parquet('road_america_features_engineered.parquet')
    db_road_america = db_road_america.drop_duplicates(subset=['meta_session', 'vehicle_id', 'lap'], keep='first')

    model_circuit_of_the_americas = joblib.load('circuit_of_the_americas_overall_model.pkl')
    db_circuit_of_the_americas = pd.read_parquet('circuit_of_the_americas_features_engineered.parquet')
    db_circuit_of_the_americas = db_circuit_of_the_americas.drop_duplicates(subset=['meta_session', 'vehicle_id', 'lap'], keep='first')

    model_virginia_international_raceway = joblib.load('model_virginia_international_raceway_overall.pkl')
    db_virginia_international_raceway = pd.read_parquet('virginia_international_raceway_features_engineered.parquet')
    db_virginia_international_raceway = db_virginia_international_raceway.drop_duplicates(subset=['meta_session', 'vehicle_id', 'lap'], keep='first')

    #print('Vehicles in R1, db_circuit_of_the_americas:', db_circuit_of_the_americas[db_circuit_of_the_americas['meta_session'] == 'R1']['vehicle_id'].unique())

    db = db_barber
    model = model_barber
    print("Model and engineered database loaded successfully.")
    print(f"Database has {len(db)} rows ready for lookup.")
    
except Exception as e:
    print(f"FATAL ERROR: Could not load model or database: {e}")
    model = None
    db = None

# --- HELPER FUNCTION (Unchanged) ---
def get_features_for_lap(race_session, vehicle_id, lap):
    """
    Looks up the feature vector using boolean masking.
    Returns a 1-row DataFrame.
    """
    try:
        #print(f"Looking up features for: {(race_session, vehicle_id, lap)}")
        # Use boolean masking to find the exact row
        mask = (db['meta_session'] == race_session) & \
               (db['vehicle_id'] == vehicle_id) & \
               (db['lap'] == lap)
               
        feature_df = db[mask].copy()

        #print("Unique laps for this vehicle/session:", db[(db['meta_session'] == race_session) & (db['vehicle_id'] == vehicle_id)]['lap'].unique())
        
        if len(feature_df) == 0:
            print(f"LOOKUP FAILED for: {(race_session, vehicle_id, lap)}")
            return None
            
        # Return the 1-row DataFrame
        return feature_df
        
    except Exception as e:
        print(f"Error in get_features_for_lap: {e}")
        return None

# --- HELPER FUNCTION 2 (Unchanged) ---
def predict_single_lap(feature_df):
    """Calls the model with a 1-row feature DataFrame."""
    pred_log = model.predict(feature_df)
    pred_sec = np.expm1(pred_log)
    return pred_sec[0]

PIT_STOP_TIME = {
    'Barber': 34.0,
    'Circuit of the Americas': 36.0,
    'Road America': 52.0,
    'Sonoma': 45.0,
    'Virginia International Raceway': 25.0
}

# --- API ENDPOINT (SMARTER SIMULATION) ---
@app.route('/api/simulate-stint', methods=['POST'])
def simulate_stint():
    global db, model, model_barber, model_sonoma, db_barber, db_sonoma, db_road_america, model_road_america, db_circuit_of_the_americas, model_circuit_of_the_americas, model_virginia_international_raceway, db_virginia_international_raceway
    data = request.get_json()
    
    track_name = data.get('track_name')
    race_session = data.get('race_session')
    vehicle_id = data.get('vehicle_id')
    start_lap = int(data.get('start_lap'))
    stint_length = int(data.get('stint_length'))
    is_pit_stop = data.get('is_pit_stop', False)

    default_value = 100.92

    if track_name == 'Barber':
        db, model, default_value = db_barber, model_barber, 100.92
    elif track_name == 'Sonoma':
        db, model, default_value = db_sonoma, model_sonoma, 120.53
    elif track_name == 'Road America':
        db, model, default_value = db_road_america, model_road_america, 158.46
    elif track_name == 'Circuit of the Americas':
        db, model, default_value = db_circuit_of_the_americas, model_circuit_of_the_americas, 167.83
    elif track_name == 'Virginia International Raceway':
        db, model, default_value = db_virginia_international_raceway, model_virginia_international_raceway, 133.60
    else:
        return jsonify({"error": "Invalid track name"}), 400

    if not all([race_session, vehicle_id, start_lap, stint_length is not None]):
        return jsonify({"error": "Missing required fields."}), 400

    # --- Base features ---
    base_features = get_features_for_lap(race_session, vehicle_id, start_lap)
    
    if base_features is None or base_features.empty:
        if start_lap == 1:
            template_row = db[
                (db['meta_session'] == race_session) &
                (db['vehicle_id'] == vehicle_id)
            ].head(1).copy()

            if template_row.empty:
                return jsonify({"error": f"No data found AT ALL for vehicle {vehicle_id} in {race_session}."}), 404

            base_features = template_row
            base_features.loc[:, 'lap'] = 1
            base_features.loc[:, 'is_out_lap'] = 1
            base_features.loc[:, 'is_normal_lap'] = 0
            base_features.loc[:, 'laps_on_tires'] = 1
            base_features.loc[:, 'last_normal_lap_time'] = np.nan
            base_features.loc[:, 'rolling_3_normal_lap_avg'] = np.nan
        else:
            return jsonify({"error": f"No historical data for {(race_session, vehicle_id, start_lap)}"}), 404

    lap_times, true_times = [], []
    current_features = base_features.copy()
    row_index = current_features.index[0] 
    lap_history = [
        current_features.loc[row_index].get('last_normal_lap_time', default_value)
    ]
    lap_history = [t for t in lap_history if pd.notna(t) and t > 0]

    original_fuel_proxy = current_features.loc[row_index].get('fuel_load_proxy', np.nan)
    original_laps_on_tires = current_features.loc[row_index].get('laps_on_tires', 0)

    # --- Apply realistic pit stop modifiers ---
    tire_reset_value = 1
    out_lap_penalty = 3.0   # seconds slower after pit stop
    fresh_tire_bonus = -1.5 # faster pace after first lap on new tires
    fuel_reset_adjustment = -8 if is_pit_stop else 0  # simulate refuel (less weight if 0 fuel policy)

    for i in range(stint_length):
        current_lap_num_to_predict = start_lap + i
        current_features.loc[row_index, 'lap'] = current_lap_num_to_predict

        # --- Handle pit stop start ---
        if is_pit_stop and i == 0:
            current_features.loc[row_index, 'is_out_lap'] = 1
            current_features.loc[row_index, 'is_normal_lap'] = 0
            current_features.loc[row_index, 'laps_on_tires'] = tire_reset_value
        else:
            current_features.loc[row_index, 'is_out_lap'] = 0
            current_features.loc[row_index, 'is_normal_lap'] = 1
            current_features.loc[row_index, 'laps_on_tires'] = original_laps_on_tires + i + (1 if is_pit_stop else 0)

            if i > 0:
                prev_true = true_times[-1]
                prev_pred = lap_times[-1]
                last_time = prev_true if prev_true > 0 else prev_pred
                current_features.loc[row_index, 'last_normal_lap_time'] = last_time

                lap_history.append(last_time)
                if len(lap_history) > 3:
                    lap_history.pop(0)
                current_features.loc[row_index, 'rolling_3_normal_lap_avg'] = np.mean(lap_history)

        # --- Update fuel proxy dynamically ---
        if not pd.isna(original_fuel_proxy):
            current_features.loc[row_index, 'fuel_load_proxy'] = (
                original_fuel_proxy - i + fuel_reset_adjustment
            )

        # --- Predict lap time ---
        predicted_time = predict_single_lap(current_features)

        # --- Apply human-style modifiers for realism ---
        # if is_pit_stop:
        #     if i == 0:
        #         predicted_time += out_lap_penalty  # slow out lap
        #     elif i == 1:
        #         predicted_time += fresh_tire_bonus  # quick recovery next lap
        #     elif i < 5:
        #         # gradually converge to normal over next few laps
        #         decay_factor = np.exp(-i / 3.0)
        #         predicted_time += (fresh_tire_bonus * decay_factor)

        if is_pit_stop and i == 0:
            pit_time = PIT_STOP_TIME.get(track_name, 0)
            predicted_time += pit_time  # add pit lane duration

        lap_times.append(predicted_time)

        true_row = db[
            (db['meta_session'] == race_session) &
            (db['vehicle_id'] == vehicle_id) &
            (db['lap'] == current_lap_num_to_predict)
        ]
        true_time = float(true_row.iloc[0]['lap_time_seconds']) if len(true_row) > 0 else 0.0
        true_times.append(true_time)

    return jsonify({
        "race_session": race_session,
        "vehicle_id": vehicle_id,
        "start_lap": start_lap,
        "predicted_lap_times": lap_times,
        "true_lap_times": true_times
    })

# --- API ENDPOINT: Get unique vehicle IDs for a given race session ---
@app.route('/api/get-vehicles', methods=['POST'])
def get_unique_vehicles():
    data = request.get_json()
    track_name = data.get('track_name')
    race_session = data.get('race_session')

    if not race_session:
        return jsonify({"error": "Missing 'race_session' in request body."}), 400

    current_db = db_barber # Default
    if track_name == 'Barber':
        current_db = db_barber
    elif track_name == 'Sonoma':
        current_db = db_sonoma
    elif track_name == 'Road America':
        current_db = db_road_america
    elif track_name == 'Circuit of the Americas':
        current_db = db_circuit_of_the_americas
    elif track_name == 'Virginia International Raceway':
        current_db = db_virginia_international_raceway

    # Filter and extract unique vehicle IDs for this race session
    try:
        mask = current_db['meta_session'] == race_session
        unique_vehicles = current_db.loc[mask, 'vehicle_id'].dropna().unique().tolist()

        if not unique_vehicles:
            return jsonify({
                "race_session": race_session,
                "vehicle_ids": [],
                "message": f"No vehicles found for race_session '{race_session}'."
            }), 404

        return jsonify({
            "race_session": race_session,
            "vehicle_ids": sorted(unique_vehicles), # Sort them for consistency
            "count": len(unique_vehicles)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve vehicles: {e}"}), 500

# --- API ENDPOINT: Get lap numbers
@app.route('/api/get-laps', methods=['POST'])
def get_unique_laps():
    data = request.get_json()
    track_name = data.get('track_name')
    race_session = data.get('race_session')
    vehicle_id = data.get('vehicle_id')

    if not race_session or not vehicle_id:
        return jsonify({"error": "Missing 'race_session' or 'vehicle_id' in request body."}), 400

    current_db = db_barber # Default
    if track_name == 'Barber':
        current_db = db_barber
    elif track_name == 'Sonoma':
        current_db = db_sonoma
    elif track_name == 'Road America':
        current_db = db_road_america
    elif track_name == 'Circuit of the Americas':
        current_db = db_circuit_of_the_americas
    elif track_name == 'Virginia International Raceway':
        current_db = db_virginia_international_raceway

    # Filter and extract unique vehicle IDs for this race session
    try:
        available_laps = current_db[
            (current_db['meta_session'] == race_session) &
            (current_db['vehicle_id'] == vehicle_id)
        ]['lap'].dropna().unique()

        try:
            lap_numbers = sorted([int(x) for x in available_laps])
        except Exception:
            lap_numbers = sorted(list(available_laps))

        return jsonify({
            "lap_numbers": lap_numbers
        })
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve vehicles: {e}"}), 500

@app.route('/api/get-final-results', methods=['POST'])
def get_final_results():
    """
    Returns final race results for a given track/session using true lap times.
    Applies a minimum lap time threshold so no lap below MIN_LAP_TIME is counted.
    Includes best lap, completed laps, total time, DNF status, and invalid lap reporting.
    """
    data = request.get_json()
    track_name = data.get('track_name')
    race_session = data.get('race_session')
    MIN_LAP_TIME = int(data.get('min_lap_time_enforced', 60))

    if not track_name or not race_session:
        return jsonify({"error": "Missing 'track_name' or 'race_session'"}), 400

    # --- Select DB ---
    if track_name == 'Barber':
        current_db = db_barber
    elif track_name == 'Sonoma':
        current_db = db_sonoma
    elif track_name == 'Road America':
        current_db = db_road_america
    elif track_name == 'Circuit of the Americas':
        current_db = db_circuit_of_the_americas
    elif track_name == 'Virginia International Raceway':
        current_db = db_virginia_international_raceway
    else:
        return jsonify({"error": f"Unknown track '{track_name}'"}), 400

    # Filter for session
    session_df = current_db[current_db['meta_session'] == race_session].copy()

    if session_df.empty:
        return jsonify({"error": f"No results found for session '{race_session}'"}), 404

    # --- Apply MIN_LAP_TIME Filtering ---
    session_df['is_valid_lap'] = session_df['lap_time_seconds'] >= MIN_LAP_TIME

    # We consider only valid laps for timing
    valid_df = session_df[session_df['is_valid_lap']]

    # Determine race max lap using valid laps only
    max_lap = int(valid_df['lap'].max())

    results = []

    # Group by vehicle
    for vehicle_id, group in session_df.groupby('vehicle_id'):

        valid_group = group[group['is_valid_lap']]

        completed_laps = int(valid_group['lap'].max()) if not valid_group.empty else 0

        total_time = float(valid_group['lap_time_seconds'].sum()) if completed_laps > 0 else None

        best_lap_time = float(valid_group['lap_time_seconds'].min()) if completed_laps > 0 else None

        did_finish = (completed_laps == max_lap)

        # Identify invalid laps (anomalies)
        invalid_laps = group[~group['is_valid_lap']]['lap'].tolist()

        results.append({
            "vehicle_id": vehicle_id,
            "completed_laps": completed_laps,
            "total_time": total_time,
            "best_lap_time": best_lap_time,
            "invalid_laps": invalid_laps,
            "status": "Finished" if did_finish else "DNF"
        })

    # Sort finished first (lowest total time) then DNF
    sorted_results = sorted(
        results,
        key=lambda x: (
            x['status'] == "DNF",
            float('inf') if x['total_time'] is None else x['total_time']
        )
    )

    return jsonify({
        "track_name": track_name,
        "race_session": race_session,
        "max_lap": max_lap,
        "min_lap_time_enforced": MIN_LAP_TIME,
        "results": sorted_results
    })

# oh boy
def get_model_features(db_dataframe):
    """Gets the X features (all columns except target)"""
    if 'lap_time_seconds' in db_dataframe.columns:
        return db_dataframe.drop(columns=['lap_time_seconds']).columns
    return db_dataframe.columns

@app.route('/api/predict-new-session', methods=['POST'])
def predict_new_session():
    """
    Predicts a new race session based on minimal user input,
    iteratively predicting lap by lap.
    """
    MIN_LAP_TIME_ENFORCE = 25.0  # Min realistic lap time
    DEFAULT_SESSION_FOR_TEMPLATE = "R2" # Use R2 data to find default values
    
    # --- 1. Load Globals and Parse Request ---
    global model_barber, model_sonoma, model_road_america, model_circuit_of_the_americas, model_virginia_international_raceway
    global db_barber, db_sonoma, db_road_america, db_circuit_of_the_americas, db_virginia_international_raceway

    body = request.get_json() or {}
    track_name = body.get('track_name')
    vehicle_id = body.get('vehicle_id') # This is a new, user-defined ID
    total_laps_to_predict = int(body.get('total_laps_to_predict', 0))
    previous_laps = body.get('previous_laps', []) # list of dicts

    if not track_name or not vehicle_id or total_laps_to_predict <= 0:
        return jsonify({"error": "track_name, vehicle_id and total_laps_to_predict are required"}), 400

    # --- 2. Select Model and Get Feature List ---
    model_map = {
        'Barber': (model_barber, db_barber),
        'Sonoma': (model_sonoma, db_sonoma),
        'Road America': (model_road_america, db_road_america),
        'Circuit of the Americas': (model_circuit_of_the_americas, db_circuit_of_the_americas),
        'Virginia International Raceway': (model_virginia_international_raceway, db_virginia_international_raceway)
    }

    if track_name not in model_map:
        return jsonify({"error": f"Unknown track_name '{track_name}'"}), 400
        
    model, db_used = model_map[track_name]
    
    # Get the exact list of columns the model pipeline expects
    MODEL_FEATURES = get_model_features(db_used)

    # --- 3. Get Template Row for Defaults ---
    # Try to get any row from the default session to fill categorical/static features
    template = db_used[
        (db_used['meta_session'] == DEFAULT_SESSION_FOR_TEMPLATE) & (db_used['vehicle_id'] == vehicle_id)
    ].sort_values('lap').head(1)
    
    if template.empty:
         # Fallback: get ANY row from that track
        template = db_used[db_used['meta_session'] == DEFAULT_SESSION_FOR_TEMPLATE].head(1)
    if template.empty:
        # Fallback: empty dict, we'll use NaNs
        base_row = {}
    else:
        base_row = template.iloc[0].to_dict()

    # --- 4. Initialize State from Previous Laps ---
    results_rows = []
    lap_history_sec = [] # For calculating rolling averages
    
    start_lap = 1
    # State variables to be updated in the loop
    current_last_lap_time = np.nan
    current_laps_on_tires = 1
    current_fuel_load = None # Will be set below
    
    # Use user-provided temps if available, else fall back to template
    current_air_temp = base_row.get('session_air_temp', np.nan)
    current_track_temp = base_row.get('session_track_temp', np.nan)

    if previous_laps:
        prev = sorted(previous_laps, key=lambda x: int(x.get('lap', 0)))
        
        # Add provided laps to results
        for p in prev:
            lap_time = float(p.get('lap_time_seconds')) if p.get('lap_time_seconds') is not None else None
            results_rows.append({
                'lap': int(p.get('lap')),
                'lap_time_seconds': lap_time,
                'provided': True
            })
            if lap_time is not None and lap_time >= MIN_LAP_TIME_ENFORCE:
                lap_history_sec.append(lap_time)

        # Get the *last* provided lap to set the initial state
        last_prev_lap = prev[-1]
        start_lap = int(last_prev_lap.get('lap', 0)) + 1
        
        # Override state with user's last provided values
        current_last_lap_time = float(last_prev_lap.get('lap_time_seconds', np.nan))
        current_laps_on_tires = int(last_prev_lap.get('laps_on_tires', 1)) + 1 # Start next lap
        current_fuel_load = float(last_prev_lap.get('fuel_load_proxy', np.nan))
        current_air_temp = float(last_prev_lap.get('session_air_temp', current_air_temp))
        current_track_temp = float(last_prev_lap.get('session_track_temp', current_track_temp))

    # --- 5. Iterative Prediction Loop ---
    for lap_num in range(start_lap, total_laps_to_predict + 1):
        # Create a feature dict with all expected columns, defaulting to NaN
        feat = {col: np.nan for col in MODEL_FEATURES}
        
        # --- Overwrite with known values ---
        
        # 1. Categorical/Static Features
        feat['track'] = track_name
        feat['race_session'] = base_row.get('race_session', DEFAULT_SESSION_FOR_TEMPLATE) # Use template default
        feat['meta_session'] = base_row.get('meta_session', DEFAULT_SESSION_FOR_TEMPLATE) # Use template default
        feat['vehicle_id'] = vehicle_id # The new, user-provided ID
        feat['original_vehicle_id'] = vehicle_id # Assume same as vehicle_id
        feat['vehicle_number'] = base_row.get('vehicle_number', -1) # Use template's, or -1 as 'unknown'
        
        # 2. Key Engineered Features
        feat['last_lap_time'] = current_last_lap_time
        feat['laps_on_tires'] = current_laps_on_tires
        
        # Handle fuel proxy: either decrement user's value or use formula
        if current_fuel_load is not None and not np.isnan(current_fuel_load):
            current_fuel_load -= 1 # Decrement from last known value
        else:
            # Fallback to formula if not provided
            current_fuel_load = float(max(0, total_laps_to_predict - lap_num))
        feat['fuel_load_proxy'] = current_fuel_load
        
        # 3. Session Features
        feat['session_air_temp'] = current_air_temp
        feat['session_track_temp'] = current_track_temp
        
        # 4. Other Features (assuming not a pit lap)
        feat['CROSSING_FINISH_LINE_IN_PIT'] = 0 
        feat['pit_flag'] = 0
        feat['is_new_stint'] = 0
        
        # Create the 1-row DataFrame
        # All other 70+ columns (mean_speed, S1_SECONDS, etc.) remain np.nan
        # The model's SimpleImputer will fill these with the training data median
        feature_df = pd.DataFrame([feat], columns=MODEL_FEATURES)

        # --- 6. Predict ---
        try:
            pred_log = model.predict(feature_df)
            pred_sec = float(np.expm1(pred_log)[0])
        except Exception as e:
            return jsonify({"error": f"Prediction failed on lap {lap_num}: {e}. Features: {feat}"}), 500

        if pred_sec < MIN_LAP_TIME_ENFORCE:
            pred_sec = MIN_LAP_TIME_ENFORCE
            
        # --- 7. Append Results and Update State for Next Loop ---
        results_rows.append({
            'lap': lap_num,
            'lap_time_seconds': pred_sec,
            'provided': False
        })
        
        # Update state for the *next* iteration
        current_last_lap_time = pred_sec
        current_laps_on_tires += 1
        # current_fuel_load was already updated
        
    # --- 8. Format Final Response ---
    lap_times_all = [r['lap_time_seconds'] for r in results_rows if r['lap_time_seconds'] is not None]
    lap_times_predicted = [float(r['lap_time_seconds']) for r in results_rows if not r['provided'] and r['lap_time_seconds'] is not None]
    total_race_time = sum(lap_times_all)
    best_lap = min(lap_times_all) if lap_times_all else 0

    return jsonify({
        "track_name": track_name,
        "vehicle_id": vehicle_id,
        "total_laps_to_predict": total_laps_to_predict,
        "start_lap_predicted_from": start_lap,
        "predicted_laps": results_rows, # Contains both provided and predicted
        "predicted_lap_times": lap_times_predicted,
        "total_predicted_time": total_race_time,
        "best_lap_time": best_lap
    })
if __name__ == '__main__':
    app.run(debug=True, port=5000)