#%%
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from global_variables import *
from model_functions import get_vulnerability, calculate_chronic_state
import glob
import os
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns

class Debug():
    
    def __init__(self, results):
        self.params = results['params']
        print(self.params['noise_seeds'])
        self.chronic_threshold = results['params']['chronic_threshold']
        self.df = results['df']
        self.no_interactions = results['params']['interactions']
        self.ses_noise = results['params']['ses_noise']
        self.repeats = results['params']['repeats']
        self.vul_param = results['params']['vul_param']
        self.psr_param = results['params']['psr_param']
        self.recover_param = results['params']['recover_param']
        self.noise_seeds = results['params']['noise_seeds']
        self.similarity = results['params']['similarity']
        self.time = results['params']['time']
        
        self.data = results
        self.stress = self.data['stress']
        self.stress_agents = self.data['stress_agents']
        self.stress_std = self.data['stress_std']
        self.stress_low = self.data['stress_low']
        self.stress_med = self.data['stress_med']
        self.stress_high = self.data['stress_high']
        self.stress_low_std = self.data['stress_low_std']
        self.stress_med_std = self.data['stress_med_std']
        self.stress_high_std = self.data['stress_high_std']

        self.chronic = self.data['chronic']
        self.chronic_agents = self.data['chronic_agents']
        self.chronic_std = self.data['chronic_std']
        self.chronic_low = self.data['chronic_low']
        self.chronic_med = self.data['chronic_med']
        self.chronic_high = self.data['chronic_high']
        self.chronic_low_std = self.data['chronic_low_std']
        self.chronic_med_std = self.data['chronic_med_std']
        self.chronic_high_std = self.data['chronic_high_std']

        self.prestige = self.data['prestige']
        self.prestige_agents = self.data['prestige_agents']
        self.prestige_std = self.data['prestige_std']
        self.prestige_low = self.data['prestige_low']
        self.prestige_med = self.data['prestige_med']
        self.prestige_high = self.data['prestige_high']
        self.prestige_low_std = self.data['prestige_low_std']
        self.prestige_med_std = self.data['prestige_med_std']
        self.prestige_high_std = self.data['prestige_high_std']
        
        self.interactions = self.data['interactions']
        self.interactions_agents = self.data['interactions_agents']
        self.interactions_std = self.data['interactions_std']
        self.interactions_low = self.data['interactions_low']
        self.interactions_med = self.data['interactions_med']
        self.interactions_high = self.data['interactions_high']
        self.interactions_low_std = self.data['interactions_low_std']
        self.interactions_med_std = self.data['interactions_med_std']
        self.interactions_high_std = self.data['interactions_high_std']
        
        self.params = self.params 
        self.df = self.df[['status', 'psr', 'eth']],
        self.df = self.df[0]            
             
        self.no_agents = self.df.shape[0]
        self.low_status, self.med_status, self.high_status = self.split_population_status()
        
     
    def split_population_status(self):
        idx = np.argsort(self.df['status'])
        low = idx[:int(self.no_agents/3)]
        med = idx[int(self.no_agents/3):int(self.no_agents/3*2)]
        high = idx[int(self.no_agents/3*2):]
        return low, med, high

    def split_population_psr(self):
        idx = np.argsort(self.df['psr'])
        low = idx[:int(self.no_agents/3)]
        med = idx[int(self.no_agents/3):int(self.no_agents/3*2)]
        high = idx[int(self.no_agents/3*2):]
        return low, med, high
    
    def print_init(self):
        print("job_nr\t\t", self.params['job_nr'])
        print("chronic_threshold\t", self.params['chronic_threshold'])
        print("similarity_base\t\t", self.params['similarity_base'])
        print("interactions\t\t", self.params['interactions'])
        print("ses_noise\t\t", self.params['ses_noise'])
        print("repeats\t\t\t", self.params['repeats'])
        print("vul_param\t\t", self.params['vul_param'])
        print("psr_param\t\t", self.params['psr_param'])
        print("recover_param\t\t", self.params['recover_param'])
        print("time\t\t\t", self.params['time'])
        print("stress_max\t\t", self.params['stress_max'])
        print("noise_seeds\t\t", len(self.params['noise_seeds']))
        print("prestige beta\t\t", self.params['prestige_beta'])
        print("prestige param\t\t", self.params['prestige_param'])
        print("stressor_param\t\t", self.params['stressor_param'])
        print("population_shape\t", self.population.shape)

if __name__ == "__main__":
    with open("./results/debug/100.pkl", 'rb') as f:
        file = pickle.load(f)
        d = Debug(file)


    # print(d.stress_agents.shape)
    plt.scatter(d.df['status'].iloc[d.low_status] + np.mean(d.prestige_agents[d.low_status], axis=1), np.mean(d.stress_agents[d.low_status], axis=1), alpha=0.2, c='b')
    plt.scatter(d.df['status'].iloc[d.med_status] + np.mean(d.prestige_agents[d.med_status], axis=1), np.mean(d.stress_agents[d.med_status], axis=1), alpha=0.2, c='r')
    plt.scatter(d.df['status'].iloc[d.high_status] + np.mean(d.prestige_agents[d.high_status], axis=1), np.mean(d.stress_agents[d.high_status], axis=1), alpha=0.2, c='g')
    # # z = np.polyfit(np.array(d.df['status']+ np.mean(d.prestige_agents, axis=1), dtype=np.float32), np.array(np.mean(d.stress_agents, axis=1), dtype=np.float32), 1)
    # # p = np.poly1d(z)
    # # x = np.array(d.df['status']+ np.mean(d.prestige_agents, axis=1), dtype=np.float32)
    # plt.plot(x,p(x),"--", c='k')
    plt.show()
    
    # plt.hist(x, bins=50)
    # plt.show()
    print(d.stress_agents.shape)
    
    stress_end = d.stress_agents[:, 99].flatten()
    print(len(stress_end[stress_end < 0.0001]))
    plt.hist(np.array(d.stress_agents[:, 99].flatten(), dtype=np.float32), bins=100)
    plt.show()
    
    d.df['stress'] = np.mean(d.stress_agents, axis=1)
    d.df['interactions'] = np.mean(d.interactions_agents, axis=1)
    
    df = d.df.loc[d.df['stress'] < 0.0001]
    df.hist(figsize = (12,12))
    df.plot.scatter(x='status', y='stress')
    print(len(df.index))
    plt.show()
    # plt.ylim([-1,500])
    # plt.xlim([-0.1,0.1])
    # plt.ylim([-0.1, 5])
    # stress = np.mean(d.stress_m, axis=0)
    # chronic = np.mean(d.chronic_m, axis=0)
    
    # for i in range(stress.shape[1]):
    #     plt.plot(stress[:, i], c='gray')
    # plt.plot(np.mean(stress, axis=1), c='red')
    # plt.ylabel('stress')
    # plt.show()        

    # # for i in range(stress.shape[1]):
    # #     plt.plot(chronic[:, i], c='gray')
    # #     plt.plot(np.mean(chronic, axis=1), c='red')
    # # plt.ylabel('chronic')
    # # plt.show()        
    
    # print("\n\n")

    # stress = np.mean(d.stress_m[d.low_status, :, :], axis=0)
    # for i in range(stress.shape[1]):
    #     plt.plot(stress[:, i], c='gray')
    # plt.plot(np.mean(stress, axis=1), c='red')
    # plt.ylabel('stress')
    # plt.show()        
    
    # stress = np.mean(d.stress_m[d.med_status, :, :], axis=0)
    # for i in range(stress.shape[1]):
    #     plt.plot(stress[:, i], c='gray')
    # plt.plot(np.mean(stress, axis=1), c='red')
    # plt.ylabel('stress')
    # plt.show()        
    
    # stress = np.mean(d.stress_m[d.high_status, :, :], axis=0)
    # for i in range(stress.shape[1]):
    #     plt.plot(stress[:, i], c='gray')
    # plt.plot(np.mean(stress, axis=1), c='red')
    # plt.ylabel('stress')
    # plt.show()        
        