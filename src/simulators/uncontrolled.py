# perfect prediction scenario. 
# for benchmarking and evaluation only!
# every time a new EV arrives, we assume all its features are known. 

# use OA!! Filter out EV0000 cars. --> only use active sessions. 

from typing import Tuple, Optional
from datetime import time
import datetime as dt
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import math
import statistics
import networkx as nx
import numpy as np
import os
from src.simulators.base_simulator import Simulator

# focs
from src.common.FlowbasedOfflineChargingScheduler.FOCS import FOCSinstance, FlowNet, FlowOperations, FOCS, Schedule
from src.common.FlowbasedOfflineChargingScheduler.OA import OA

# Baver's code
from src.common.session_generator import generate_sessions_from_profile

import logging
logger = logging.getLogger(__name__)
 
logging.basicConfig( # write logger.info for messages and warnings, if for debugging, write logger.debug. Saved to file.
    filename= os.path.join('outputs', 'focs.log'),  
    filemode='a',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



class Uncontrolled(Simulator):
    def __init__(self, learning_prefix = "") -> None: 
        self.timeStep = 1
        self.timeBase = 3600
        self.tau = self.timeStep/self.timeBase
        self.power_default_kW = 13

        self.time_bins = pd.date_range("00:00", "23:45", freq="15min").time
        self.time_to_idx = {t: i for i, t in enumerate(self.time_bins)}

        self.identifier = 'uncontrolled'
        self.learning_prefix = learning_prefix

    def initilize(self) -> None:
        logger.info('---------------------------------------')
        logger.info('---------------------------------------')
        logger.info('initialization of optimizer - new day')
        # placeholder network and flow
        self.len_i = []
        self.supplied_energy = {}
        self.sim_profiles = {}
        self.breaks = []
        self.offset = 0 # used to increase robustness to steps with non-consecutive intervals

    def preprocessing(self,input, upcoming, curr_time=None):
        # update start times to curr_time (if further in the past).
        input['start_time_original'] = [x.time() for x in input['start_datetime']]
        if curr_time is not None:
            input['start_time'] = input['start_time_original'].mask(input['start_time_original'] < curr_time, curr_time)
        else: 
            input['start_time'] = input['start_time_original']
        input['start'] = [dt.datetime.combine(dt.datetime(1,1,1),input['start_time'].iloc[x]) for x in range(0,len(input))]
        input['start_original'] = [dt.datetime.combine(dt.datetime(1,1,1),input['start_time_original'].iloc[x]) for x in range(0,len(input))]
        # update end times.
        input['end_time'] = [x.time() for x in input['end_datetime']]
        input['end'] = [dt.datetime.combine(dt.datetime(1,1,1),input['end_time'].iloc[x]) for x in range(0,len(input))]

        # add new ids to dataframe
        self.new_ids = [key for key in upcoming['session_id'].unique() if key not in self.supplied_energy.keys()] # no dataleakage. Just bookkeeping.
        # logger.debug("new ids this step: {}".format(self.new_ids))
        for id in self.new_ids:
            self.supplied_energy[str(id)] = 0
            self.sim_profiles[str(id)] = [0 for x in range(0,len(self.len_i))]

        # update power
        input['maxPower'] = [self.power_default_kW*1000 for x in range(0,len(input))]
        input['average_power_W'] = 1000*input['total_energy']/((input['end'] - input['start_original'])/ np.timedelta64(1, 'h'))
        input['power'] = [22000 if input["average_power_W"].iloc[j]>input["maxPower"].iloc[j] else input["maxPower"].iloc[j] for j in range(0,len(input))] # in W

        # update energy based on previously scheduled
        input['total_energy_Wh'] = [max(0,input['total_energy'].iloc[x]-self.supplied_energy[str(input['session_id'].iloc[x])])*1000 if input['session_id'].iloc[x] in self.supplied_energy.keys() else input['total_energy'].iloc[x]*1000 for x in range(0,len(input))] 
        # check whether feasible solution still exists. Else, reduce 
        input['energy_uncontrolled_Wh'] = input['power']*((input['end'] - input['start'])/ np.timedelta64(1, 'h'))
        input['energy_deficit'] = [max(0, input['total_energy_Wh'].iloc[j] - input['energy_uncontrolled_Wh'].iloc[j]) for j in range(0, len(input))]
        if len(input)>0:
            if max(input['energy_deficit']) > 0:
                logger.debug('[WARNING]: EVs {} cannot be feasibly scheduled anymore based on their max power. We reduce their energy demand for the scheduler.'.format(input[input['energy_deficit']>0]['session_id'].to_list()))
                input['total_energy_Wh'] = input['total_energy_Wh'] - input['energy_deficit']

        # discretize 
        input['t0_' + str(900)] = input['start'].apply(lambda x: math.floor((x.hour*3600 + x.minute*60 + x.second)/900))
        input['t1_' + str(900)] = input['end'].apply(lambda x: math.ceil((x.hour*3600 + x.minute*60 + x.second)/900))
        input['t0_' + str(1)] = input['start'].apply(lambda x: math.floor((x.hour*3600 + x.minute*60 + x.second)/1))
        input['t1_' + str(1)] = input['end'].apply(lambda x: math.ceil((x.hour*3600 + x.minute*60 + x.second)/1))

        return input, upcoming


    def flow_to_dataframe(self, f):
        # flow to dataframe - format schedule
        s = [self.oa.instance.I_a] + [self.oa.instance.intervals_start] + [self.breaks[:-1]] + [self.len_i] + [[f['i'+str(i)]['t']/self.oa.instance.tau /self.oa.instance.len_i[i] for i in self.oa.instance.I_a]] + [[0]*len(self.oa.instance.I_a) for j in self.oa.instance.jobs]
        for jid, j in enumerate(self.oa.instance.jobs):
            for i in self.oa.instance.J_inverse['j'+str(j)]:
                s[5+jid][i] = f['j'+str(j)]['i'+str(i)] / self.oa.instance.tau /self.oa.instance.len_i[i]
        s = pd.DataFrame(s).T
        s.columns = ['time','breakpoints','breaks','len_i','agg'] + ['j'+str(j) for j in self.oa.instance.jobs]
        return s

    def step(self, curr_time: time, df_agg_timeseries: pd.DataFrame, df_usr_sessions: pd.DataFrame, active_session_info: pd.DataFrame) -> None:
        self.sessions = df_usr_sessions
        energy_agg = df_agg_timeseries
        upcoming = active_session_info
        # logger.debug(self.sessions.to_string())
        
        if curr_time != dt.time(hour=23, minute=45):
            logger.debug('skipped curr_time {}'.format(curr_time))
        else:

            # preprocess in second granularity
            logger.info('start optimizer.step() at curr_time {}'.format(curr_time))
            input = self.sessions.loc[self.sessions['end_datetime'].notnull()]
            input = input.reset_index()

            '''--------------update instance based on previous timesteps--------------'''
            # converts start and end times (in new columns)
            # updates self.new_ids
            # applies padding to self.supplied_energy and self.sim_profiles (for new_ids)
            # adds column with input energy demand - energy already served [in Wh]
            # adds power columns
            # adds columns with 15-minute indexed time stamps (0,1,2) instead of (00:00,00:15,00:30)
            # adds columns with second indexed time stamps (0,1,2) instead of (00:00:00,00:00:01,00:00:02)
            input, upcoming = self.preprocessing(input, upcoming)
            self.input = input

            '''-------------save schedule ----------------'''
            # based on write_power() in Bookkeeping.py

            # extract breakpoints within planningsinterval (full day)
            breaks = list(set(input['start_time'].to_list() + input['end_time'].to_list()))
            breaks = [dt.datetime.combine(dt.datetime(1,1,1,0,0,0),b) for b in breaks]
            self.breaks = list(set(breaks))
            self.breaks.sort()

            # add breaks by which jobs complete charging
            bp_per_id = []
            for id in input['session_id'].unique():
                # seconds to deliver demand at full power
                delta_secs = math.ceil(self.timeBase*input[input['session_id']==id]['total_energy'].iloc[0]/self.power_default_kW)
                bp_per_id += [dt.datetime.combine(dt.datetime(1,1,1,0,0,0),input[input['session_id']==id]['start_time'].iloc[0])+dt.timedelta(seconds=delta_secs)]

            self.breaks = self.breaks + bp_per_id
            self.breaks = list(set(self.breaks))
            self.breaks.sort()

            # log the lenghts of the intervals
            self.len_i = [(self.breaks[i] - self.breaks[i-1]).total_seconds() for i in range(1,len(self.breaks))] + [1]

            # save sim_profiles
            for idx, id in enumerate(input['session_id'].unique()):
                self.sim_profiles[id] = [0 for x in range(0,len(self.len_i))]
                self.sim_profiles[id][self.breaks.index(dt.datetime.combine(dt.datetime(1,1,1),input[input['session_id']==id]['start_time'].iloc[0])):self.breaks.index(bp_per_id[idx])+1] = [13 for i in range(self.breaks.index(dt.datetime.combine(dt.datetime(1,1,1),input[input['session_id']==id]['start_time'].iloc[0])),self.breaks.index(bp_per_id[idx])+1)]

            '''--------------bookkeeping--------------'''
            # update supplied energy
            taus = np.array([x*self.tau/self.timeStep for x in self.len_i]) # conversion factors

            for key in input['session_id'].unique(): 
                # except 'EV0000' key
                if key[1:7] != 'EV0000':
                    self.supplied_energy[key] = sum(np.array(self.sim_profiles[key])*taus)

            logger.debug('end of optimizer.step at tick {}'.format(curr_time))
        return

    def evaluate(self,real_data):# based on Schedule.objective() from FlowbasedOfflineChargingScheduler

        if len(real_data) != len(self.sim_profiles):
            logger.debug('[WARNING]: data likely contains non-unique EV-ids. Evaluation indices are messed up.')

        logger.debug('start evaluation step')

        # sorting breaks (again....) cause QOS2 broke...
        self.breaks.sort()

        # supplied energy to dataframe
        self.state = pd.DataFrame.from_dict(self.supplied_energy, orient='index',columns=['supplied_energy'])
        # drop 'EV0000' line
        # self.state.drop(self.state[self.state.index.map(str) == 'EV0000'].index, inplace=True)
        self.state['idx'] = [i for i in range(0,len(self.state))]
        ids = self.state.index.tolist()
        self.supplied_energy
        ## Metrics per job

        # ENS exact
        self.jobs_ens_abs_exact = [real_data['total_energy'].loc[real_data['session_id']==id].iloc[0] - self.state['supplied_energy'].loc[id] for id in ids]
        self.jobs_ens_rel_exact = [self.jobs_ens_abs_exact[self.state['idx'].loc[id]]/real_data['total_energy'].loc[real_data['session_id']==id].iloc[0] for id in ids]
        
        # ENS rounded and >= 0
        self.jobs_ens_abs = [round(max(0,x),6) for x in self.jobs_ens_abs_exact]
        self.jobs_ens_rel = [round(max(0,x),6) for x in self.jobs_ens_rel_exact]

        # QOS1 : relative energy served
        self.jobs_es_rel = [1 - x for x in self.jobs_ens_rel]

        # QOS2 : relative wait till charging starts
        self.jobs_qos2_waiting_real = self.qos_2(real_data)

        # QOS3 : variation of charging power over time (relative to max power)
        self.jobs_qos3_powervar_real = self.qos_3(real_data)

        # QOE : 
        self.jobs_qoe_real = self.qoe(real_data)

        ## Global metrics

        # #QOE total
        self.qoe_total_exact = sum(self.jobs_qoe_real)
        self.qoe_total_rel = self.qoe_total_exact/len(real_data)

        # ENS max
        self.jobs_ens_abs_max = max(self.jobs_ens_abs)
        self.jobs_ens_rel_max = max(self.jobs_ens_rel)

        # QOS1 min
        self.qos_1_min = min(self.jobs_es_rel)
        # QOS2 min
        self.qos_2_real_min = min(self.jobs_qos2_waiting_real)
        # QOS3 min
        self.qos_3_real_min = min(self.jobs_qos3_powervar_real)
        
        # Jain's fairness index based on relative ENS
        self.jain_ens_rel = self.Jain(self.jobs_ens_rel)
        self.jain_ens_rel_exact = self.Jain(self.jobs_ens_rel_exact)

        # Jain's fairness index based on qos and qoe
        self.jain_qos_1 = self.Jain(self.jobs_es_rel)
        self.jain_qos_2_real = self.Jain(self.jobs_qos2_waiting_real)
        self.jain_qos_3_real = self.Jain(self.jobs_qos3_powervar_real)

        # Hoßfeld's fairness index based on relative ENS
        self.hossfeld_ens_rel = self.Hossfeld(self.jobs_ens_rel)
        self.hossfeld_ens_rel_exact = self.Hossfeld(self.jobs_ens_rel_exact)

        # Hoßfeld's fairness index based on qos and qoe
        self.hossfeld_qos_1 = self.Hossfeld(self.jobs_es_rel)
        self.hossfeld_qos_2_real = self.Hossfeld(self.jobs_qos2_waiting_real)
        self.hossfeld_qos_3_real = self.Hossfeld(self.jobs_qos3_powervar_real)

        # cycle switches TODO

        # energy served
        self.es_exact = sum(real_data['total_energy']) - sum(self.jobs_ens_abs_exact)
        self.es = round(sum(real_data['total_energy']) - sum(self.jobs_ens_abs),6)
        
        # energy not served
        self.ens_abs_exact = sum(self.jobs_ens_abs_exact)
        self.ens_abs = sum(self.jobs_ens_abs)
        self.ens_rel_exact_avg = sum(self.jobs_ens_rel_exact)/len(self.jobs_ens_rel_exact)
        self.ens_rel_avg = sum(self.jobs_ens_rel)/len(self.jobs_ens_rel)

        # power peak
        self.peak = self.peak_power()

        # flatness objective value (2-norm)
        self.flat_value = self.flatness()

        return
    
    def Jain(self, x):
        if len(x) * sum([x_i**2 for x_i in x]) == 0:
            return 1 # all equally bad, therefore fair. 
        return sum(x)**2/(len(x) * sum([x_i**2 for x_i in x]))
    
    def Hossfeld(self, x, H = 1, h = 0):
        return 1 - 2* statistics.pstdev(x)/(H-h)
    
    def qos_2(self, data): # qos_2 in Danner and de Meer (2021)
        # NOTE trivially 1 in this setup, as upon arrival, we charge EVs at full power till the next planning interval
        # determine first interval with positive charge
        # FIXME only 135 entries instead of 139!! --> has to do with non-unique EVids. Some EVs register for 2 charging sessions in a single day.
        first_pos_charging_power = [self.sim_profiles[id].index([i for i in self.sim_profiles[id] if i!=0][0]) for id in self.sim_profiles.keys() if id[1:7] != 'EV0000']
        t_start_charging = [dt.datetime.combine(data['start_datetime'].iloc[0].date(),self.breaks[i].time()) for i in first_pos_charging_power]

        # determine qos2
        if len(t_start_charging) != len(data):
            logger.info('[WARNING]: indexing gone wrong. You better double check your assumptions.') # likely non-unique EVids.
        self.jobs_qos2_waiting_real = [max(0,1 - (t_start_charging[j] - data['start_datetime'].iloc[j])/(data['end_datetime'].iloc[j] - data['start_datetime'].iloc[j])) for j in range(0,len(t_start_charging))]

        # determine qos2 from the first full quarter
        # we do not determine qo2_plan, because of perfect predictions. plan = real.

        return self.jobs_qos2_waiting_real
    
    def qos_3(self, data):
        # NOTE Checking correctness of the statements by Danner and de Meer with Maria rn.
        # NOTE Checked. They use the biased sample standard deviation (divide by N). See https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9639.1980.tb00398.x for proof of bound.

        # update power
        data['maxPower'] = [self.power_default_kW for x in range(0,len(data))]
        data['average_power_kW'] = data['total_energy']/((data['end_datetime'] - data['start_datetime'])/ np.timedelta64(1, 'h'))
        data['power'] = [22 if data["average_power_kW"].iloc[j]>data["maxPower"].iloc[j] else data["maxPower"].iloc[j] for j in range(0,len(data))] # in W
        
        # create lists to store results. Index = job index.
        jobs_qos3_powervar_real = []
        
        for id in self.sim_profiles.keys():
            if id[1:7] != 'EV0000':
                if len(self.len_i) != len(self.sim_profiles[id]):
                    logger.debug("[WARNING]: sim_profiles has different number of intervals than len_i.")
                # list of powers per second
                s_j = [[self.sim_profiles[id][i] for x in range(0,int(self.len_i[i]))] for i in range(0,len(self.len_i))]
                s_j = [x for xx in s_j for x in xx] # to flatten the list         

                # retrieve first and last non-zero index
                temp = [i for i,x in enumerate(s_j) if x!=0]

                # raise exception if no charging at all (statistics error std with empty sample)
                if len(temp) == 0:
                    logger.debug('[WARNING]: EV id = {} does not charge. QoS_3 set to 1.'.format(id))
                    jobs_qos3_powervar_real += [1]
                    continue

                t_start = min(temp)
                t_end = max(temp)+1

                # due to the updating session features, may require higher power later (22kW instead of 13). --> need to also check max used power. 
                try:
                    temp_max = max(data[data['session_id'] == id]['power'].iloc[0], max(s_j).iloc[0])
                except:
                    temp_max = max(data[data['session_id'] == id]['power'].iloc[0], max(s_j))
                if temp_max>data[data['session_id'] == id]['power'].iloc[0]:
                    temp_max = 22

                # qos for j add to list
                jobs_qos3_powervar_real += [1 - (2*statistics.pstdev(s_j[t_start:t_end])/temp_max) ]
               
        return jobs_qos3_powervar_real
    
    def supplied_by_milestone(self,id,milestone_t):
        if len(self.breaks) > len(set(self.breaks)):
            logger.info("[WARNING]: breaks are not unique at QoE step")
            self.breaks = list(set(self.breaks))
            self.breaks.sort()
        if len(self.breaks) != len(self.sim_profiles[id]): # =1 in other sims. 
            logger.debug('[WARNING]: breaks doesnt correspond to profile intervals + 1')
            logger.debug('len breaks vs len profile: {}, {}'.format(len(self.breaks), len(self.sim_profiles[id])))
        if milestone_t not in self.breaks:
            logger.info('[WARNING]: Milestone_t = {} not in breakpoints. Adding breakpoint and updating sim_profiles.'.format(milestone_t))
            # augment all sim profiles
            # identify index to split up in two intervals
            i = sum(b < milestone_t for b in self.breaks)-1
            # insert intermediate breakpoint in all power profiles
            for key in self.sim_profiles.keys():
                self.sim_profiles[key] = self.sim_profiles[key][:i] + [self.sim_profiles[key][i]] + self.sim_profiles[key][i:]
            # add to breaks
            self.breaks += [milestone_t]
            self.breaks.sort()
            self.len_i = [(self.breaks[i] - self.breaks[i-1]).total_seconds() for i in range(1,len(self.breaks))] + [1] # the +1 only applies if padded sim_profiles with a 0 at the end.
            self.taus = np.array([x*self.tau/self.timeStep for x in self.len_i]) # conversion factors

        m_absent = len(self.breaks) - self.breaks.index(milestone_t) -1
        supplied = sum(np.array(self.sim_profiles[id][:-m_absent])*self.taus[:-m_absent])
        #FIXME doublecheck index offset
        return supplied

    def qoe_j(self,data,id,milestones_j, err=0.0000001):
        self.taus = np.array([x*self.tau/self.timeStep for x in self.len_i]) # conversion factors
        for milestone in milestones_j:
            supplied = self.supplied_by_milestone(id, milestone[1])
            # check if made milestone or fully charged for this milestone. 
            if supplied < min(milestone[0],data[data['session_id']==id]['total_energy'].iloc[0])-err:
                return 0
        return 1

    def qoe(self, data, err = 0.0000001, milestones_predef = None): # check milestones and guarantees.
        # Binary metric per EV. 1 if requirements met. 0 if not.

        # if button is off, make guarantee static asr
        if milestones_predef is None:
            # add asr milestones: 23kWh by 4pm (if quarters then by quarter 64)
            milestones = [[[23,dt.datetime.combine(dt.datetime(1,1,1), dt.time(hour=16))]] for id in data['session_id']] # if multiple guarantees, energy is additive!
        else:
            milestones = milestones_predef

        jobs_qoe_real = [self.qoe_j(data,data['session_id'].iloc[j], milestones[j]) for j in range(0, len(data))]
        logger.debug('\n jobs qoe real {}'.format(jobs_qoe_real))
        logger.debug('check that later!! Does it hold true with the outputted simprofiles?')
        return jobs_qoe_real
    
    def flatness(self):
        #flatness objective for validation
        #note needs to be normalized if timestep and timebase granularity don't match.    
        #determine power per interval
        p_i = [sum([self.sim_profiles[y][x] for y in self.sim_profiles.keys() if y != 'EV0000']) for x in range(0,len(set(self.breaks))-1)]
        #determine weighted squared powers
        powerSquare = [(p_i[i]**2)*(self.len_i[i]) for i in range(0,len(p_i))]
        self.flat_value = sum(powerSquare)
        return self.flat_value
    
    def peak_power(self):
        return max([sum([self.sim_profiles[y][x] for y in self.sim_profiles.keys() if y != 'EV0000']) for x in range(0,len(set(self.breaks))-1)])

    def publish_results(self, output_dir: str, prefix: Optional[str] = None) -> None:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug('start publishing')
        prefix = '{}_{}_{}'.format(self.sessions['start_datetime'].iloc[0].date(), self.identifier, self.learning_prefix)
        f_name = "sim_profiles.csv" if prefix == None else f"{prefix}_sim_profiles.parquet"
        file_path = os.path.join(output_dir, f_name)
        self.breaks = list(set(self.breaks))
        self.breaks.sort()
        self.sim_profiles['agg'] = [sum([self.sim_profiles[y][x] for y in self.sim_profiles.keys() if y != 'EV0000']) for x in range(0,len(set(self.breaks)))]#-1)]
        self.sim_profiles['start_time'] = [x.time() for x in self.breaks]#[:-1]]
        self.sim_profiles['start'] = [(x - x.replace(hour=0, minute=0, second=0)).seconds for x in self.breaks]#[:-1]]
        pd.DataFrame(self.sim_profiles).to_parquet(file_path)
        pd.DataFrame(self.sim_profiles).to_csv(os.path.join(output_dir,'{}_sim_profiles.csv'.format(prefix)))
        del self.sim_profiles['start_time']
        del self.sim_profiles['agg']
        del self.sim_profiles['start']

        f_name = "supplied_energy.csv" if prefix == None else f"{prefix}_supplied_energy.parquet"
        file_path = os.path.join(output_dir, f_name)
        pd.DataFrame(self.supplied_energy, index=[0]).to_parquet(file_path)
        pd.DataFrame(self.supplied_energy, index=[0]).to_csv(os.path.join(output_dir,'{}_supplied_energy.csv'.format(prefix)))

        # evaluate qos and qoe and fairness metrics
        self.breaks = list(set(self.breaks))
        self.evaluate(real_data=self.sessions)
        if len(self.sessions) != len(self.jobs_ens_abs):
            logger.error('Day likely contains >= 2 sessions by a single EV. Cannot save qosqoe metrics.')
        mets_jobs = {'ids': self.sessions['session_id'].to_list(), 'day':[self.sessions['start_datetime'].iloc[0].date() for i in range(0,len(self.sessions))], 'ens_abs_exact': self.jobs_ens_abs_exact, 'ens_rel_exact': self.jobs_ens_rel_exact,'ens_abs': self.jobs_ens_abs, 'ens_rel': self.jobs_ens_rel, 'qos1': self.jobs_es_rel, 'qos2': self.jobs_qos2_waiting_real, 'qos3': self.jobs_qos3_powervar_real, 'qoe': self.jobs_qoe_real}
        mets_global = {'day':[self.sessions['start_datetime'].iloc[0].date()], 'ens_abs_max': [self.jobs_ens_abs_max], 'ens_rel_max': [self.jobs_ens_rel_max], 'qos1_min': [self.qos_1_min], 'qos2_min': [self.qos_2_real_min], 'qos3_min': [self.qos_3_real_min], 'qoe_total_exact': [self.qoe_total_exact], 'qoe_total_rel': [self.qoe_total_rel], 'jain_ens_rel_exact': [self.jain_ens_rel_exact], 'jain_ens_rel': [self.jain_ens_rel], 'jain_qos1': [self.jain_qos_1], 'jain_qos2': [self.jain_qos_2_real], 'jain_qos3': [self.jain_qos_3_real], 'hossfeld_ens_rel_exact': [self.hossfeld_ens_rel_exact], 'hossfeld_ens_rel': [self.hossfeld_ens_rel], 'hossfeld_qos1': [self.hossfeld_qos_1], 'hossfeld_qos2': [self.hossfeld_qos_2_real], 'hossfeld_qos3': [self.hossfeld_qos_3_real], 'es_total': [self.es], 'es_exact_total': [self.es_exact], 'ens_abs_exact_total': [self.ens_abs_exact], 'ens_rel_exact_avg': [self.ens_rel_exact_avg], 'ens_rel_avg': [self.ens_rel_avg], 'flat':[self.flat_value], 'peak': [self.peak] }
        
        f_name = "qosqoe.csv" if prefix == None else f"{prefix}_qosqoe.parquet"
        file_path = os.path.join(output_dir, f_name)
        pd.DataFrame(data=mets_jobs).to_parquet(file_path)
        pd.DataFrame(data=mets_jobs).to_csv(os.path.join(output_dir,"{}_qosqoe.csv".format(prefix)))
        
        f_name = "globalmetrics.csv" if prefix == None else f"{prefix}_globalmetrics.parquet"
        file_path = os.path.join(output_dir, f_name)
        pd.DataFrame(data=mets_global).to_parquet(file_path)
        pd.DataFrame(data=mets_global).to_csv(os.path.join(output_dir, "{}_globalmetrics.csv".format(prefix)))
        logger.debug('what does the greedy rat say? l;ajsdf;llkdkjfjkdkdsfkkdkdkdkdkdkdkdkdkdkdkdkdk')
        logger.info('Evaluation of current day {} complete'.format(self.sessions['start_datetime'].iloc[0].date()))