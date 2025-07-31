import pandas as pd
import numpy as np
from datetime import time, date, datetime

from src.common.feature_engineering import FeatureEngineer

engineer = FeatureEngineer(logging_off=True)

def generate_sessions_from_profile(df_sessions: pd.DataFrame, 
                                   df_agg: pd.DataFrame, 
                                   curr_time: time,
                                   min_enegy_differnece: float = 1, 
                                   max_power_kW: float = 13
                                   ) -> pd.DataFrame:

    df_sessions_padded = df_sessions.copy()

    target_date = pd.to_datetime(df_agg['timestamp'].iloc[0]).date()

    time_bins = pd.date_range("00:00", "23:45", freq="15min").time
    time_to_idx = {t: i for i, t in enumerate(time_bins)}

    psudo_seesion_counter = 0
    while df_sessions_padded.shape[0] < 200:

        if df_sessions_padded.shape[0] == 0:
            df_timeseries_padded = df_agg.copy()
            df_timeseries_padded['total_energy'] = 0
            df_timeseries_padded['cum_ev_count'] = 0
        else:    
            engineer.set_strategy('SessionsToEvents')
            df_events_padded = engineer.apply_transformation(df_sessions_padded)

            engineer.set_strategy('EventsToTimeseries')
            df_timeseries_padded = engineer.apply_transformation(df_events_padded)

        remaining_energy_to_fill = df_agg['total_energy'] - df_timeseries_padded['total_energy']
        remaining_ev_to_fill = df_agg['cum_ev_count'] - df_timeseries_padded['cum_ev_count']

        if remaining_energy_to_fill.iloc[-1] < min_enegy_differnece or pd.isna(remaining_energy_to_fill.iloc[-1]): 
            break

        diff_energy = np.diff(remaining_energy_to_fill)
        diff_ev = np.diff(remaining_ev_to_fill)

        psudo_departures = -1 * diff_ev
        psudo_departures[psudo_departures < 1] = 1

        energy_per_departure = (diff_energy/psudo_departures)
        departure_slot_to_fill: int = int(np.argmax(energy_per_departure) + 1)
        target_energy_per_ev = np.max(energy_per_departure)
        psudo_end_datetime = pd.to_datetime(f"{target_date} {time_bins[departure_slot_to_fill]}")

        for ins_i in range(0,int(psudo_departures[departure_slot_to_fill-1])):
            idx = np.argmax(remaining_ev_to_fill)
            if (remaining_ev_to_fill > 0).any() and idx < departure_slot_to_fill:
                pass
            else:
                idx = time_to_idx[curr_time]
            
            psudo_arrival_datetime = pd.to_datetime(f"{target_date} {time_bins[idx]}")
            target_energy_per_ev = min(target_energy_per_ev, (max_power_kW/4)*(departure_slot_to_fill-idx))
            row = pd.DataFrame([{
                'EV_id_x': 'EV0000', 
                'session_id': f"UEV0000-{target_date.strftime('%Y%m%d')}-{psudo_seesion_counter+1}",           
                'start_datetime': psudo_arrival_datetime,            
                'end_datetime': psudo_end_datetime,
                'total_energy': target_energy_per_ev
                # Add any other required columns with default values here
            }])
            df_sessions_padded = pd.concat([df_sessions_padded, row], ignore_index=True)
            psudo_seesion_counter += 1

    return df_sessions_padded


def sanity_check(df_sessions: pd.DataFrame, max_power_kW: float = 13):
    required_columns = ['EV_id_x', 'start_datetime', 'end_datetime', 'total_energy']
    missing_cols = [col for col in required_columns if col not in df_sessions.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    errors = []

    # Check for null values
    for col in required_columns:
        if df_sessions[col].isnull().any():
            errors.append(f"Column '{col}' contains null values.")

    # Check that end_datetime > start_datetime
    if not (df_sessions['end_datetime'] > df_sessions['start_datetime']).all():
        invalid_rows = df_sessions[df_sessions['end_datetime'] <= df_sessions['start_datetime']]
        errors.append(f"{len(invalid_rows)} rows have end_datetime <= start_datetime.")

    # Check total_energy is positive
    if (df_sessions['total_energy'] <= 0).any():
        invalid_energy = df_sessions[df_sessions['total_energy'] <= 0]
        errors.append(f"{len(invalid_energy)} rows have non-positive total_energy.")

    # Check power ≤ max_power_kW
    duration_hours = (df_sessions['end_datetime'] - df_sessions['start_datetime']).dt.total_seconds() / 3600
    power_kW = df_sessions['total_energy'] / duration_hours

    if (power_kW > max_power_kW).any():
        too_high = df_sessions[power_kW > max_power_kW]
        errors.append(f"{len(too_high)} rows exceed {max_power_kW} kW average power: {too_high}")


    if errors:
        error_msg = "\n".join(errors)
        raise ValueError(f"Session DataFrame check failed:\n{error_msg}")
    else:
        print("✅ Session DataFrame passed all checks.")
