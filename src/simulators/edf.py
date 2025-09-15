from typing import Tuple, Optional
from dataclasses import dataclass
from datetime import time
import pandas as pd
import numpy as np
import os
from src.simulators.base_simulator import Simulator

from src.common.session_generator import generate_sessions_from_profile
import logging

logger = logging.getLogger(__name__)

class EDF(Simulator):
    def __init__(self) -> None:
        self.time_bins = pd.date_range("00:00", "23:45", freq="15min").time
        self.time_to_idx = {t: i for i, t in enumerate(self.time_bins)}

        self.max_power = 13.0 #kW
        self.min_target_power: float = 13.0 #kW
        

    def initilize(self) -> None:
        self.system_state = pd.DataFrame({
            'session_id': pd.Series(dtype='str'),
            'EV_id_x': pd.Series(dtype='str'),
            'start_datetime': pd.Series(dtype='datetime64[ns]'),
            'end_datetime': pd.Series(dtype='datetime64[ns]'),
            'total_energy_demand': pd.Series(dtype='float'),
            'total_energy_supplied': pd.Series(dtype='float'),
            'status': pd.Series(dtype='str')
        })

        
    def step(self, curr_time: time, df_agg_timeseries: pd.DataFrame, df_usr_sessions: pd.DataFrame, active_session_info: pd.DataFrame) -> None:
        curr_date = pd.to_datetime(df_agg_timeseries['timestamp'].iloc[0]).date()
        curr_datetime = pd.to_datetime(f"{curr_date} {curr_time}")

        active_session_info['start_datetime'] = pd.to_datetime(active_session_info['date'].astype(str) + ' ' + active_session_info['start_time'].astype(str))
        active_session_info['end_datetime'] = pd.to_datetime(active_session_info['date'].astype(str) + ' ' + active_session_info['end_time'].astype(str), format='mixed')
                

        #Determine the schedule
        agg_supplied_energy = self.system_state['total_energy_supplied'].sum()
        mask = df_agg_timeseries['timestamp'] > curr_datetime
        _rem_energy_kWh = (df_agg_timeseries.loc[mask,'total_energy'] - agg_supplied_energy)
        _rem_energy_kWh[_rem_energy_kWh < 0] = 0
        _rem_time_h = ((df_agg_timeseries.loc[mask,'timestamp'] - curr_datetime).dt.total_seconds() / 3600)
        _rem_time_h[_rem_energy_kWh < 0.25] = 0.25
        target_power = np.max(_rem_energy_kWh / _rem_time_h)
        target_power = max(self.min_target_power, target_power)
        mask = self.system_state['status'] == 'charging'
        
        _merged = pd.merge(self.system_state.loc[mask, ['session_id', 'total_energy_supplied']], df_usr_sessions[['session_id', 'total_energy', 'end_datetime']],  on='session_id', how='left')
        if _merged.isna().any().any():
            raise ValueError("⚠️ There are missing values in the merged DataFrame.")
        required_energy = _merged['total_energy'] - _merged['total_energy_supplied']
        required_energy[required_energy < 0] = 0
        remaining_time_h = (_merged['end_datetime'] - curr_datetime).dt.total_seconds() / 3600
        remaining_time_h[remaining_time_h < 0.25] = 0.25
        required_power = required_energy / remaining_time_h
        _merged['required_power'] = required_power
        _merged = _merged.sort_values(by='required_power', ascending=False)

        num_of_evs_to_be_charged = np.ceil(target_power / self.max_power).astype(int)

        scheduled_sessions = _merged['session_id'].head(num_of_evs_to_be_charged).tolist()

        # Apply the schedule considering departing EVs  
        for target_id in scheduled_sessions:
            _index_system_state = self.system_state[(self.system_state['session_id'] == target_id) & (self.system_state['status'] == 'charging')].index
            if len(_index_system_state) != 1:
                raise ValueError(f"Expected exactly one match for EV_id {target_id}, but found {len(_index_system_state)}: {self.system_state.loc[_index_system_state]}")

            _rem_energy_demand = self.system_state.loc[_index_system_state, 'total_energy_demand'].iloc[0] - self.system_state.loc[_index_system_state, 'total_energy_supplied'].iloc[0]            
            _energy = np.minimum(_rem_energy_demand, self.max_power*0.25)
            
            if self.system_state.loc[_index_system_state, 'end_datetime'].iloc[0] < curr_datetime + pd.Timedelta(minutes=15):
                _rem_time_energy = self.max_power * (self.system_state.loc[_index_system_state, 'end_datetime'].iloc[0] - curr_datetime).total_seconds() / 3600
                _energy = np.minimum(_rem_time_energy, _energy)

            self.system_state.loc[_index_system_state, 'total_energy_supplied'] = self.system_state.loc[_index_system_state, 'total_energy_supplied'] + _energy

        # Determine new arrivals and start charging until the next scheduling decision
        arriving_evs = active_session_info[active_session_info['start_datetime'] >= curr_datetime]

        for index, row in arriving_evs.iterrows():
            if row['session_id'] in self.system_state['session_id']:
                raise ValueError(f"Arriving EV is already connected: {row}")     
            _energy = self.max_power * ((curr_datetime + pd.Timedelta(minutes=15)) - row['start_datetime']).total_seconds() / 3600
            if _energy > self.max_power / 4:
                raise ValueError(f"Supplied energy {_energy} is not feasible since max power is {self.max_power}")
            new_row = pd.DataFrame([{
                'session_id': row['session_id'],
                'EV_id_x': row['EV_id_x'],
                'start_datetime': row['start_datetime'],
                'end_datetime': row['end_datetime'],
                'total_energy_demand': row['total_energy'],
                'total_energy_supplied': min(_energy, row['total_energy']),
                'status': 'charging'
            }])
            self.system_state = pd.concat([self.system_state, new_row], ignore_index=True)

        # Set the status 
        mask = self.system_state['end_datetime'] < curr_datetime + pd.Timedelta(minutes=15)
        self.system_state.loc[mask, 'status'] = 'departed'

        mask = (self.system_state['total_energy_supplied'] >= self.system_state['total_energy_demand']) &  (self.system_state['status'] == 'charging')
        self.system_state.loc[mask, 'total_energy_supplied'] = self.system_state.loc[mask, 'total_energy_demand']
        self.system_state.loc[mask, 'status'] = 'fully_charged'

        generate_sessions_from_profile(df_usr_sessions, df_agg_timeseries, curr_time)


    def publish_results(self, output_dir: str, prefix: Optional[str] = None) -> None:
        os.makedirs(output_dir, exist_ok=True)
        f_name = "simulation_system_state.csv" if prefix == None else f"{prefix}_simulation_system_state.parquet"
        file_path = os.path.join(output_dir, f_name)
        self.system_state.to_csv(file_path)
