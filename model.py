#%%
import json
import pandas as pd
import numpy as np
from time import time
from copy import deepcopy
import datetime
import pickle

import numpy as np
from global_variables import *
from setup import *
from functions import *
from model_functions import *

from pprint import pprint

def run_models(Dict):
    m = Model(**Dict)
    del m

class Model():
    
    def __init__(self, **args):

        # Data & info
        self.params = args
        self.df = args['population']
        del self.params['population']
        self.similarity = args['similarity']
        self.config = args['config'].split('/')[-1].split('.')[0]
        self.time = args['time']
        self.no_agents = self.df.shape[0]
        self.chronic_threshold = self.params['chronic_threshold']
        self.no_interactions = args['interactions']
        self.stress_max = args['stress_max']
        self.repeats = args['repeats']
        
        # Seeds
        self.noise_seeds = args['noise_seeds']
        
        # Model parameters
        self.ses_noise = args['ses_noise']
        self.vul_param = args['vul_param']
        self.psr_param = args['psr_param']
        self.stressor_param = args['stressor_param']
        self.recover_param = args['recover_param']
        self.similarity_minimum = args['similarity_min']
        self.coping_noise = args['coping_noise']
        self.prestige_param = args['prestige_param']
        self.prestige_beta = args['prestige_beta']
        
        # Calulations
        self.calc_similarities()
        self.similarity_base = deepcopy(self.similarity)
        self.events_m = self.get_events()
        

        # Collector placeholders
        self.stress_m = np.zeros((self.no_agents, self.time, self.repeats), dtype=np.float16)
        self.prestige_m = np.zeros((self.no_agents, self.time, self.repeats), dtype=np.float16)
        self.interactions_m = np.zeros((self.no_agents, self.time, self.repeats), dtype=np.int16)
        self.chronic_m = np.zeros((self.no_agents, self.time, self.repeats), dtype=np.uint8)
        
        # Run collectors
        self.stress = np.zeros(self.no_agents)
        self.prestige = copy.deepcopy(self.df['prestige'].values)
        self.interaction_history = np.zeros((self.no_agents, self.no_agents), dtype=np.uint8)
        self.interactions = np.zeros(self.no_agents, dtype=np.int16)
        
        # Quick acces values
        self.status = np.array(self.df['status'].values)
        self.psr = np.array(self.df['psr'].values)

        # Global collectors        
        self.status_difference = []
        self.stressors = []
        self.coped_stress = []
        self.similarity_interactions = []
        self.low_status, self.med_status, self.high_status = self.split_population_status()
        # self.low_psr, self.med_psr, self.high_psr = self.split_population_psr()
        # self.print_init()
        
        self.runtimes = []
        for self.i in range(self.repeats):
            # print("job_nr: ", args['job_nr'], "\tRepeat: ", self.i)
            s = time()
            np.random.seed(int(args['noise_seeds'][self.i]))
            # np.random.seed(1)
            self.events = self.events_m[self.i, :, :]
            self.run()
            self.reset_runcollectors()            
            self.runtimes.append((time() - s)/60)
        print("job_nr: ", args['job_nr'], "\tFinished in: ", np.sum(self.runtimes))
        
        if DEBUG ==True:
            self.results = {
                "job": args["job_nr"],
                "stress": self.stress_m,
                "stress_std": self.stress_m,
                "prestige": self.prestige_m,
                "prestige_std": self.prestige_m,
                "interactions": self.interactions_m,
                "chronic": self.chronic_m,
                "params": self.params, 
                "df": args['population'][['status', 'psr', 'eth','prestige']],
            }            
        elif SAVETYPE == "all":
            self.results = {
                "job": args["job_nr"],
                "stress": self.stress_m,
                "chronic": self.chronic_m,
                "prestige": self.prestige_m,
                "params": self.params, 
                "df": self.df[['status', 'psr', 'eth','prestige']],
            }
        elif SAVETYPE == "group":
            self.results = {
                "stress": np.mean(self.stress_m[:,:,:], axis=(0, 2)),
                "stress_agents": np.mean(self.stress_m[:,:,:], axis=(2)),
                "stress_std": np.std(np.mean(self.stress_m[:,:,:], axis=0), axis=1),
                "stress_low": np.mean(self.stress_m[self.low_status,:,:], axis=(0, 2)),
                "stress_med": np.mean(self.stress_m[self.med_status,:,:], axis=(0, 2)),
                "stress_high": np.mean(self.stress_m[self.high_status,:,:], axis=(0, 2)),
                "stress_low_std": np.std(np.mean(self.stress_m[self.low_status,:,:], axis=0), axis=1),
                "stress_med_std": np.std(np.mean(self.stress_m[self.med_status,:,:], axis=0), axis=1),
                "stress_high_std": np.std(np.mean(self.stress_m[self.high_status,:,:], axis=0), axis=1),

                "chronic": np.mean(self.chronic_m[:,:,:], axis=(0, 2)),
                "chronic_agents": np.mean(self.chronic_m[:,:,:], axis=(2)),
                "chronic_std": np.std(np.mean(self.chronic_m[:,:,:], axis=0), axis=1),
                "chronic_low": np.mean(self.chronic_m[self.low_status,:,:], axis=(0, 2)),
                "chronic_med": np.mean(self.chronic_m[self.med_status,:,:], axis=(0, 2)),
                "chronic_high": np.mean(self.chronic_m[self.high_status,:,:], axis=(0, 2)),                
                "chronic_low_std": np.std(np.mean(self.chronic_m[self.low_status,:,:], axis=0), axis=1),
                "chronic_med_std": np.std(np.mean(self.chronic_m[self.med_status,:,:], axis=0), axis=1),
                "chronic_high_std": np.std(np.mean(self.chronic_m[self.high_status,:,:], axis=0), axis=1),

                "prestige": np.mean(self.prestige_m[:,:,:], axis=(0, 2)),
                "prestige_agents": np.mean(self.prestige_m[:,:,:], axis=(2)),
                "prestige_std": np.std(np.mean(self.prestige_m[:,:,:], axis=0), axis=1),
                "prestige_low": np.mean(self.prestige_m[self.low_status,:,:], axis=(0, 2)),
                "prestige_med": np.mean(self.prestige_m[self.med_status,:,:], axis=(0, 2)),
                "prestige_high": np.mean(self.prestige_m[self.high_status,:,:], axis=(0, 2)),                
                "prestige_low_std": np.std(np.mean(self.prestige_m[self.low_status,:,:], axis=0), axis=1),
                "prestige_med_std": np.std(np.mean(self.prestige_m[self.med_status,:,:], axis=0), axis=1),
                "prestige_high_std": np.std(np.mean(self.prestige_m[self.high_status,:,:], axis=0), axis=1),

                "interactions": np.mean(self.interactions_m[:,:,:], axis=(0, 2)),
                "interactions_agents": np.mean(self.interactions_m[:,:,:], axis=(2)),
                "interactions_std": np.std(np.mean(self.interactions_m[:,:,:], axis=0), axis=1),
                "interactions_low": np.mean(self.interactions_m[self.low_status,:,:], axis=(0, 2)),
                "interactions_med": np.mean(self.interactions_m[self.med_status,:,:], axis=(0, 2)),
                "interactions_high": np.mean(self.interactions_m[self.high_status,:,:], axis=(0, 2)),                
                "interactions_low_std": np.std(np.mean(self.interactions_m[self.low_status,:,:], axis=0), axis=1),
                "interactions_med_std": np.std(np.mean(self.interactions_m[self.med_status,:,:], axis=0), axis=1),
                "interactions_high_std": np.std(np.mean(self.interactions_m[self.high_status,:,:], axis=0), axis=1),
                
                "params": self.params, 
                "df": self.df[['status', 'psr', 'eth']],
                "events": self.events_m
        }
           
        if not os.path.exists("./results/"+ args["save_folder"]):
            os.mkdir(os.path.dirname(os.path.realpath(__file__)) + "\\results\\"+ args["save_folder"])
            
        output = os.path.dirname(os.path.realpath(__file__)) + '\\results\\'+ args["save_folder"] + "{0:03}".format(args['job_nr'])  + ".pkl"
        f = open(output, "wb")
        pickle.dump(self.results, f)

        
        # self.df['stress'] = self.stress_mean
    
    
        
    def split_population_status(self):
        idx = np.argsort(self.df['status'])
        low = idx[:int(self.no_agents/3)]
        med = idx[int(self.no_agents/3):int(self.no_agents/3*2)]
        high = idx[int(self.no_agents/3*2):]
        return low, med, high
    
    def reset_runcollectors(self):
        self.stress = np.zeros(self.no_agents)
        self.prestige = copy.deepcopy(self.df['prestige'].values)
        self.interaction_history = np.zeros((self.no_agents, self.no_agents), dtype=np.uint8)
        self.interactions = np.zeros(self.no_agents, dtype=np.int16)            
    
    def print_init(self):
        print("\nGLOBALS:")
        print("population_shape\t", self.df.shape)
        print("time\t\t\t", self.params['time'])
        print("repeats\t\t\t", self.params['repeats'])
        print("stress_max\t\t", self.params['stress_max'])
        print("\nPARAMS:")
        print("interactions\t\t", self.params['interactions'])
        print("similarity_min\t\t", self.params['similarity_min'])
        print("ses_noise\t\t", self.params['ses_noise'])
        print("vul_param\t\t", self.params['vul_param'])
        print("psr_param\t\t", self.params['psr_param'])
        print("recover_param\t\t", self.params['recover_param'])
        print("stressor_param\t\t", self.params['stressor_param'])
        print("prestige_param\t\t", self.params['prestige_param'])
        print("prestige_beta\t\t", self.params['prestige_beta'])
        

    def calc_similarities(self):
        self.similarity[self.similarity < self.similarity_minimum] = self.similarity_minimum
        np.fill_diagonal(self.similarity, 0)
    
    def get_events(self):
        return np.random.poisson(self.no_interactions/DAY, (self.no_agents, self.time, self.repeats)).T
    

    def collect(self):
        self.stress_m[:, self.step, self.i] = copy.deepcopy(self.stress)
        self.interactions_m[:, self.step, self.i] = copy.deepcopy(self.interactions)
        self.prestige_m[:, self.step, self.i] = copy.deepcopy(self.prestige)
        if self.step >= WEEK:
           self.chronic_m[:, self.step, self.i] = calculate_chronic_state(self.stress_m, self.chronic_threshold, self.time, self.step, self.i)
        

            
    # @profile
    def run(self):
        self.step = 0
        while(self.step < self.time - 1):
            # print(self.step)
            actors = select_actors(self.events, self.step)
            self.interactions, self.interaction_history, self.stress, self.similarity, self.prestige = interact_actors(actors, self.interactions, self.interaction_history, self.similarity, self.prestige, self.status, self.psr, self.prestige_param, self.ses_noise, self.prestige_beta, self.psr_param, self.stress, self.vul_param, self.stressor_param)

            self.stress = recover(self.stress, self.recover_param, self.stress_max)
            
            if self.step%DAY==0: 
                self.reset_day()

            self.step += 1
            self.collect()
    
            
    # @profile
    def select_alter(self, agent):
        actor = int(agent[0])
        interactions = int(agent[1])
        actor_similarity = self.similarity[actor, :]
        # print(actor_similarity)
        for interaction in range(interactions):
            while(True):
                interaction_probability = transform_probability(actor_similarity)
                if interaction_probability[0] == 999:
                    return
                sorted_probability = np.argsort(interaction_probability)[::-1]
                interaction_probability = np.cumsum(interaction_probability[sorted_probability])
                dice = np.random.sample()
                alter = sorted_probability[find_nearest_above(interaction_probability, dice)]

                idx = np.sort([actor, alter])
                if(self.interaction_history[idx[0], idx[1]]!=1):
                    self.interactions[[idx[0], idx[1]]] += 1
                    return alter
                else:
                    self.similarity[idx[0], idx[1]] = 0

    def calc_status_difference(self, actor, alter):
        agents = [actor, alter]
        
        #collector
        # self.similarity_interactions.append(self.similarity[actor, alter])
        
        prestige_actor, prestige_alter = get_prestige(actor, alter, self.prestige, self.prestige_param)
        noise_actor = np.random.uniform(-self.ses_noise + prestige_actor, self.ses_noise + prestige_actor)
        noise_alter = np.random.uniform(-self.ses_noise + prestige_alter, self.ses_noise + prestige_alter)
        
        actor_pses = np.max([self.status[actor] + noise_actor, 0])
        alter_pses = np.max([self.status[alter] + noise_alter, 0]) 
        
        perceiver = agents[np.argmin([actor_pses, alter_pses])]
        discriminator = agents[np.argmax([actor_pses, alter_pses])]

        status_difference = abs(actor_pses - alter_pses)
        # collector
        # self.status_difference.append(status_difference)
        return status_difference, perceiver, discriminator


    # @profile
    def update_prestige(self, agents, target):
        status_group = self.status[agents[target]]
        group_member = random.choice(np.where(self.status == status_group)[0])
        member_distance = 1 / self.prestige[group_member]
        
        if target == 0:
            new_member_distance = member_distance * (1 + self.params['prestige_beta'])
        else:
            new_member_distance = member_distance * (1 - self.params['prestige_beta'])

        new_member_prestige = 1 / new_member_distance
        self.prestige[group_member] = new_member_prestige    

    def reset_day(self):
        self.similarity = deepcopy(self.similarity_base) 
        self.interaction_history = np.zeros((self.no_agents, self.no_agents), dtype=np.uint8)
            
    def recover(self):
        self.stress -= self.recover_param
        self.stress[self.stress < 0] = 0
        self.stress[self.stress > self.stress_max] = self.stress_max
    
    # @profile    
    def cope(self, stressor, psr):
        
        #collector
        # self.stressors.append(stressor)
        stress = stressor - psr*self.psr_param + np.absolute(np.random.normal(loc=0.0, scale=0.05))
        
        #collector
        # self.coped_stress.append(stressor-stress)
        return stress if stress > 0 else 0
            
    def calc_stressor(self, status_difference, vulnerability, stressor_p):
        stressor = self.get_stressor(status_difference)
        return stressor * (1 + vulnerability)
        

    def get_stressor(self, status_difference):
        x_null = 3
        L = 1.049
        k = 1
        return -0.049 + L/(1+np.exp(-k*(status_difference-x_null))) * self.stressor_param


    def get_vulnerability(self, stress):
        x_null = 4 
        L = 1.043
        k = .8
        return -0.04 + L/(1+np.exp(-k*(stress-x_null))) * self.vul_param
    

    def get_prestige(self, actor, alter):
        return self.prestige[actor]*self.params['prestige_param'], self.prestige[alter]*self.params['prestige_param']
    

                        
if __name__ == "__main__":
    root = os.path.dirname(os.path.realpath(__file__)) + "\\configs\\pre-test\\"
    configs = [root + "sample_" + "{0:03}".format(i) + ".json" for i in np.arange(0, 96)]
    configs = [
        # 'ofat_recovery', 
    #     # 'ofat_similarity', 
    #     # 'ofat_psr',
    #     # 'ofat_interactions',
    #     # 'ofat_time',
    #     # 'ofat_vulnerability',
    #     # 'ofat_noise',
    #     # "ofat_prestigebeta",
        "./configs/debug.json",
        ]
    
    jobs = []
    for i, config in enumerate(configs):
        print("preparing job: ", i)
        setup = Setup(config_file=config)
        if i == 0: jobs = setup.jobs
        else: jobs += setup.jobs
        
    print("Total jobs: ", len(jobs))
    run_models(jobs[0])
    # with Pool() as p:
        # p.map(run_models, jobs)
    print("FINISH")

# %%
