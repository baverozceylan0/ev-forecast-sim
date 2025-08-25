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

class Optimizer(Simulator):
    def __init__(self) -> None: 
        self.timeStep = 900
        self.timeBase = 3600
        self.power_default_kW = 13

        self.time_bins = pd.date_range("00:00", "23:45", freq="15min").time
        self.time_to_idx = {t: i for i, t in enumerate(self.time_bins)}

        self.identifier = 'focs'

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

        # filter past sessions
        if curr_time is not None:
            input = input[input['end_time']>curr_time]
            upcoming = upcoming[upcoming['end_time']>curr_time]

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
        logger.info("We assume 900 second time granularity input.")
        input['t0_' + str(900)] = input['start'].apply(lambda x: math.floor((x.hour*3600 + x.minute*60 + x.second)/900))
        input['t1_' + str(900)] = input['end'].apply(lambda x: math.ceil((x.hour*3600 + x.minute*60 + x.second)/900))

        return input, upcoming

    def focs_scheduler(self,input, upcoming, curr_time):
        '''-------------define and solve FOCS instance ----------------'''
        self.instance = FOCSinstance(input, self.timeStep)
        flowNet = FlowNet()
        flowNet.focs_instance_to_network(self.instance)
        flowOp = FlowOperations(flowNet.G, self.instance)
        self.focs = FOCS(self.instance, flowNet, flowOp)
        self.f = self.focs.solve_focs()
        # logger.debug('schedule at tick {} = \n{}'.format(curr_time, f))

        '''-------------save schedule-----------------------'''
        # check if intervals are 900 seconds apart from previous time:
        if self.breaks:
            if self.breaks[-1] == dt.datetime.combine(dt.datetime(1,1,1),curr_time):
                logger.debug("Normal execution mode: consecutive 15 min intervals detected")
                self.offset = 0
                pass
            else:
                logger.info("[WARNING]: Abnormal execution mode: non-consecutive 15 minute intervals detected.\nSchedule for intermediate time steps has not been recorded.")
                self.offset = 1
                self.len_i +=[(dt.datetime.combine(dt.datetime(1,1,1), curr_time)-self.breaks[-1]).total_seconds()]
        # extract breakpoints within planningsinterval (15 min)
        breaks = list(set(upcoming['start_time'].to_list() + upcoming['end_time'].to_list() + [curr_time, (dt.datetime.combine(dt.datetime(1,1,1,0,0,0),curr_time)+dt.timedelta(minutes=15)).time()]))
        breaks = [dt.datetime.combine(dt.datetime(1,1,1,0,0,0),b) for b in breaks if b >= curr_time]
        breaks = [b for b in breaks if b <= (dt.datetime.combine(dt.datetime(1,1,1,0,0,0),curr_time)+dt.timedelta(minutes=15))]
        breaks.sort()
        self.breaks += breaks
        list(set(self.breaks)).sort()
        self.breaks_step = breaks

        # log the lenghts of the intervals
        len_i = [(breaks[i] - breaks[i-1]).total_seconds() for i in range(1,len(breaks))]
        self.len_i += len_i
        # logger.debug("leni = {}".format(len_i))
        if sum(len_i) != self.timeStep:
            logger.info("[WARNGING]: Interval broken into subintervals that sum to {} seconds (<{})!".format(sum(len_i), self.timeStep))
        
        # pad with zeros for all evs
        for id in self.sim_profiles.keys():
            self.sim_profiles[str(id)] += [0 for x in range(0,len(len_i)+self.offset)]
        # logger.debug("after padding sched update\n{}".format(self.sim_profiles))

        # old arrivals
        self.ids_old = input[input['start_time_original']< curr_time]['session_id'].to_list()
        logger.debug('ids old at curr_time {} = {}'.format(curr_time, self.ids_old))
        # update schedule according to charging schedule
        for id in self.ids_old: 
            if input['session_id'].index.get_loc(input['session_id'].index[input['session_id'] == id].values[0]) in self.focs.instance.J['i0']: 
                power = self.f['j{}'.format(input['session_id'].index.get_loc(input['session_id'].index[input['session_id'] == id].values[0]))]['i0']*(self.timeStep/self.focs.instance.len_i[0])/self.focs.instance.tau
                self.sim_profiles[str(id)][-len(len_i):] = [power for x in range(0,len(len_i))]

    def sim_step(self, upcoming, curr_time, breaks, ids_old):
        # new arrivals
        breaks = list(set(breaks))
        breaks.sort()
        new_ids = [id for id in self.new_ids if upcoming['start_time'][upcoming['session_id'] == id].iloc[0] >= curr_time] # this line makes sure we can run the code outside of the pipeline. 
        logger.debug("new_ids at curr_time {}= {}".format(curr_time, new_ids))
        for id in new_ids:
            # identify index
            if dt.datetime.combine(dt.datetime(1,1,1), upcoming['start_time'][upcoming['session_id'] == id].iloc[0]) not in breaks:
                logger.info("[ERROR]: new arrival not actually arriving. (Not in breaks list).")
            else:
                # count number of intervals where present
                m_present = len(breaks)-breaks.index(dt.datetime.combine(dt.datetime(1,1,1), upcoming['start_time'][upcoming['session_id'] == id].iloc[0])) -1 # -1 because we care about intervals, not breakpoints
                # assign default to end of array
                self.sim_profiles[id][-m_present:] = [self.power_default_kW for x in range(0,m_present)]    
        
        # departures
        ids_depart = [id for id in ids_old+new_ids if dt.datetime.combine(dt.datetime(1,1,1),upcoming[upcoming['session_id']==id]['end_time'].iloc[0]) < dt.datetime.combine(dt.datetime(1,1,1,0,0,0),curr_time)+dt.timedelta(minutes=15)]
        logger.debug("ids departure at curr_time {} = {}".format(curr_time, ids_depart))
        if len(ids_depart) ==0:
            logger.info('No departures detected at curr_time {}'.format(curr_time))
        else:
            # logger.debug('{} departures detected at curr_time {}'.format(len(ids_depart),curr_time))
            for id in ids_depart:
                if len(breaks) > len(set(breaks)):
                    logger.info("[WARNING]: breaks are not unique.")
                m_absent = len(breaks) - breaks.index(dt.datetime.combine(dt.datetime(1,1,1), upcoming['end_time'][upcoming['session_id'] == id].iloc[0])) -1
                # logger.debug("{},{}".format(id,m_absent))
                self.sim_profiles[id][-m_absent:] = [0 for x in range(0,m_absent)] # offset different from new arrivals. Checked with data though. This is correct.               

        # energy completed
        # update supplied energy
        taus = np.array([x*self.instance.tau/self.timeStep for x in self.len_i]) # conversion factors
        for key in self.supplied_energy.keys(): 
            # except 'EV0000' key
            if key[1:7] != 'EV0000':
                self.supplied_energy[key] = sum(np.array(self.sim_profiles[key])*taus)
        upcoming['served'] = upcoming['session_id'].map(self.supplied_energy)
        upcoming['demand'] = upcoming['total_energy'] - upcoming['served']
        
        # identify completed charging
        id_complete = upcoming[upcoming['demand']<0]['session_id'].to_list()

        if id_complete:
            logger.debug('id_complete at curr_time {} = {}'.format(curr_time, id_complete))
            # identify point of completed charging for each completed charging session
            bp_per_id = []
            for id in id_complete:
                i = 1
                while self.supplied_energy[id] - sum((np.array(self.sim_profiles[id])*taus)[-i:]) > upcoming[upcoming['session_id']==id]['total_energy'].iloc[0]:
                    i+=1
                demand = upcoming[upcoming['session_id']==id]['total_energy'].iloc[0] - self.supplied_energy[id] + sum((np.array(self.sim_profiles[id])*taus)[-i:])
                # seconds to deliver demand
                delta_secs = math.ceil(self.timeBase*demand/self.sim_profiles[id][-i])
                bp_per_id += [breaks[-i-1]+dt.timedelta(seconds=delta_secs)]
                logger.debug('id, i, demand, deltasecs - {}, {}, {}, {}'.format(id, i, demand, delta_secs))

            list(set(self.breaks))
            self.breaks.sort()
            
            for bp in list(set(bp_per_id)):
                if bp not in self.breaks:
                    # identify index to split up in two intervals
                    i = sum(b < bp for b in set(self.breaks))-1
                    # insert intermediate breakpoint in all power profiles
                    for key in self.sim_profiles.keys():
                        self.sim_profiles[key] = self.sim_profiles[key][:i] + [self.sim_profiles[key][i]] + self.sim_profiles[key][i:]
                    self.breaks += [bp]
                    self.breaks.sort()

            # update self.breaks
            self.breaks = list(set(self.breaks))
            self.breaks.sort()
            # update self.len_i
            self.len_i = [(self.breaks[i] - self.breaks[i-1]).total_seconds() for i in range(1,len(self.breaks))]            

            for idx, id in enumerate(id_complete): #FIXME
                # set power to 0 after charging is complete.
                m_absent = len(self.breaks) - self.breaks.index(bp_per_id[idx]) # removed -1. Failed with instance where 1st interval is relevant (delta = 12 seconds).
                self.sim_profiles[id][-m_absent:] = [0 for x in range(0,m_absent)] 
        
    def step(self, curr_time: time, df_agg_timeseries: pd.DataFrame, df_usr_sessions: pd.DataFrame, active_session_info: pd.DataFrame) -> None:
        self.sessions = df_usr_sessions
        energy_agg = df_agg_timeseries
        upcoming = active_session_info

        logger.info('Generate sessions from predicted energy and occupancy profile.')
        # generate dummy sessions from forecast
        input = generate_sessions_from_profile(self.sessions, energy_agg, curr_time)

        logger.info('start optimizer.step() at curr_time {}'.format(curr_time))
        input = input.loc[input['end_datetime'].notnull()]
        input = input.reset_index()

        '''--------------update instance based on previous timesteps--------------'''
        # converts start and end times (in new columns)
        # updates start times to curr_time
        # filters past sessions from input and upcoming sessions
        # updates self.new_ids
        # applies padding to self.supplied_energy and self.sim_profiles (for new_ids)
        # adds column with input energy demand - energy already served [in Wh]
        # adds power columns
        # adds columns with 15-minute indexed time stamps (0,1,2) instead of (00:00,00:15,00:30)
        input, upcoming = self.preprocessing(input, upcoming, curr_time)
        self.input = input

        # temporary filter: #FIXME how does that relate to new power 
        n = len(input)
        input = input[input['average_power_W']>0] # remove sessions with departure before arrival
        if n- len(input)>0:
            logger.info("[WARNING]: removed {} sessions from data based on non-negativity of power.".format(n-len(input)))
        input = input[input['average_power_W']<=22000] # remove sessions that require >22kW to complete
        # input['maxPower'][input['average_power_W']>22000] = input['average_power_W'][input['average_power_W']>22000] #FIXME
        if n- len(input)>0:
            logger.info("[WARNING]: removed {} sessions from data based on average power needed to complete session.".format(n-len(input)))

        # # logger.debug(input.to_string())

        '''--------------start scheduler (FOCS) with local info--------------'''
        # solves instance using FOCS
        # updates self.sim_profiles with zeros or schedule according to predicted avail.
        self.focs_scheduler(input, upcoming, curr_time)

        '''--------------update schedule with upcoming events--------------'''
        self.sim_step(upcoming, curr_time, self.breaks, self.ids_old)
        
        '''--------------bookkeeping--------------'''
        # update supplied energy
        taus = np.array([x*self.instance.tau/self.timeStep for x in self.len_i]) # conversion factors
        for key in self.supplied_energy.keys(): 
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
        try:
            self.state.drop(['EV0000'],inplace=True)
        except:
            pass
        self.state['idx'] = [i for i in range(0,len(self.state))]
        ids = self.state.index.tolist()

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

        return
    
    def Jain(self, x):
        if len(x) * sum([x_i**2 for x_i in x]) == 0:
            return 1 # all equally bad, therefore fair. 
        return sum(x)**2/(len(x) * sum([x_i**2 for x_i in x]))
    
    def Hossfeld(self, x, H = 1, h = 0):
        return 1 - 2* statistics.pstdev(x)/(H-h)
    
    def qos_2(self, data): # qos_2 in Danner and de Meer (2021)

        # determine first interval with positive charge
        # FIXME only 135 entries instead of 139!! --> has to do with non-unique EVids. Some EVs register for 2 charging sessions in a single day.
        first_pos_charging_power = [self.sim_profiles[id].index([i for i in self.sim_profiles[id] if i!=0][0]) for id in self.sim_profiles.keys() if id[1:7] != 'EV0000']
        t_start_charging = [dt.datetime.combine(data['start_datetime'].iloc[0].date(),self.breaks[i].time()) for i in first_pos_charging_power]

        # determine qos2
        if len(t_start_charging) != len(data):
            logger.info('[WARNING]: indexing gone wrong. You better double check your assumptions.') # likely non-unique EVids.
        self.jobs_qos2_waiting_real = [max(0,1 - (t_start_charging[j] - data['start_datetime'].iloc[j])/(data['end_datetime'].iloc[j] - data['start_datetime'].iloc[j])) for j in range(0,len(t_start_charging))]
        logger.debug(self.jobs_qos2_waiting_real)
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

                # qos for j add to list
                jobs_qos3_powervar_real += [1 - (2*statistics.pstdev(s_j[t_start:t_end])/data[data['session_id'] == id]['power'].iloc[0]) ]

        return jobs_qos3_powervar_real
    
    def supplied_by_milestone(self,id,milestone_t):
        if len(self.breaks) > len(set(self.breaks)):
            logger.info("[WARNING]: breaks are not unique at QoE step")
            self.breaks = list(set(self.breaks))
            self.breaks.sort()
        if len(self.breaks) != len(self.sim_profiles[id])+1:
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
            self.len_i = [(self.breaks[i] - self.breaks[i-1]).total_seconds() for i in range(1,len(self.breaks))] #+ [1] # the +1 only applies if padded sim_profiles with a 0 at the end.
            self.taus = np.array([x*self.instance.tau/self.timeStep for x in self.len_i]) # conversion factors

        m_absent = len(self.breaks) - self.breaks.index(milestone_t) -1
        supplied = sum(np.array(self.sim_profiles[id][:-m_absent])*self.taus[:-m_absent])
        #FIXME doublecheck index offset

        return supplied

    def qoe_j(self,data,id,milestones_j, err=0.0000001):
        self.taus = np.array([x*self.instance.tau/self.timeStep for x in self.len_i]) # conversion factors
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
        return jobs_qoe_real

    def publish_results(self, output_dir: str, prefix: Optional[str] = None) -> None:
        os.makedirs(output_dir, exist_ok=True)
        prefix = '{}_focs'.format(self.sessions['start_datetime'].iloc[0].date())
        f_name = "sim_profiles.csv" if prefix == None else f"{prefix}_sim_profiles.parquet"
        file_path = os.path.join(output_dir, f_name)
        self.breaks = list(set(self.breaks))
        self.breaks.sort()
        self.sim_profiles['agg'] = [sum([self.sim_profiles[y][x] for y in self.sim_profiles.keys() if y != 'EV0000']) for x in range(0,len(set(self.breaks))-1)]
        self.sim_profiles['start_time'] = [x.time() for x in self.breaks[:-1]]
        self.sim_profiles['start'] = [(x - x.replace(hour=0, minute=0, second=0)).seconds for x in self.breaks[:-1]]
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
        mets_global = {'day':[self.sessions['start_datetime'].iloc[0].date()], 'ens_abs_max': [self.jobs_ens_abs_max], 'ens_rel_max': [self.jobs_ens_rel_max], 'qos1_min': [self.qos_1_min], 'qos2_min': [self.qos_2_real_min], 'qos3_min': [self.qos_3_real_min], 'qoe_total_exact': [self.qoe_total_exact], 'qoe_total_rel': [self.qoe_total_rel], 'jain_ens_rel_exact': [self.jain_ens_rel_exact], 'jain_ens_rel': [self.jain_ens_rel], 'jain_qos1': [self.jain_qos_1], 'jain_qos2': [self.jain_qos_2_real], 'jain_qos3': [self.jain_qos_3_real], 'hossfeld_ens_rel_exact': [self.hossfeld_ens_rel_exact], 'hossfeld_ens_rel': [self.hossfeld_ens_rel], 'hossfeld_qos1': [self.hossfeld_qos_1], 'hossfeld_qos2': [self.hossfeld_qos_2_real], 'hossfeld_qos3': [self.hossfeld_qos_3_real], 'es_total': [self.es], 'es_exact_total': [self.es_exact], 'ens_abs_exact_total': [self.ens_abs_exact], 'ens_rel_exact_avg': [self.ens_rel_exact_avg], 'ens_rel_avg': [self.ens_rel_avg] }
        
        f_name = "qosqoe.csv" if prefix == None else f"{prefix}_qosqoe.parquet"
        file_path = os.path.join(output_dir, f_name)
        pd.DataFrame(data=mets_jobs).to_parquet(file_path)
        pd.DataFrame(data=mets_jobs).to_csv(os.path.join(output_dir,"{}_qosqoe.csv".format(prefix)))
        
        f_name = "globalmetrics.csv" if prefix == None else f"{prefix}_globalmetrics.parquet"
        file_path = os.path.join(output_dir, f_name)
        pd.DataFrame(data=mets_global).to_parquet(file_path)
        pd.DataFrame(data=mets_global).to_csv(os.path.join(output_dir, "{}_globalmetrics.csv".format(prefix)))
