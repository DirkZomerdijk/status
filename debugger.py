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
import seaborn as sns; sns.set()
from model_functions import *

class Debug():
    
    def __init__(self, results):
        self.params = results['params']
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
        
        self.stress = results['stress']
        # self.stress_agents = results['stress_agents']
        # self.stress_std = results['stress_std']
        # self.stress_low = results['stress_low']
        # self.stress_med = results['stress_med']
        # self.stress_high = results['stress_high']
        # self.stress_low_std = results['stress_low_std']
        # self.stress_med_std = results['stress_med_std']
        # self.stress_high_std = results['stress_high_std']

        self.chronic = results['chronic']
        # self.chronic_agents = results['chronic_agents']
        # self.chronic_std = results['chronic_std']
        # self.chronic_low = results['chronic_low']
        # self.chronic_med = results['chronic_med']
        # self.chronic_high = results['chronic_high']
        # self.chronic_low_std = results['chronic_low_std']
        # self.chronic_med_std = results['chronic_med_std']
        # self.chronic_high_std = results['chronic_high_std']

        self.prestige = results['prestige']
        # self.prestige_agents = results['prestige_agents']
        # self.prestige_std = results['prestige_std']
        # self.prestige_low = results['prestige_low']
        # self.prestige_med = results['prestige_med']
        # self.prestige_high = results['prestige_high']
        # self.prestige_low_std = results['prestige_low_std']
        # self.prestige_med_std = results['prestige_med_std']
        # self.prestige_high_std = results['prestige_high_std']
        
        self.interactions = results['interactions']
        # self.interactions_agents = results['interactions_agents']
        # self.interactions_std = results['interactions_std']
        # self.interactions_low = results['interactions_low']
        # self.interactions_med = results['interactions_med']
        # self.interactions_high = results['interactions_high']
        # self.interactions_low_std = results['interactions_low_std']
        # self.interactions_med_std = results['interactions_med_std']
        # self.interactions_high_std = results['interactions_high_std']
        print(results.keys())
        self.status_difference = results['status_difference'],
        self.events = results['events'].T
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
        print("similarity_base\t\t", self.params['similarity_min'])
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
        print("population_shape\t", self.df.shape)


if __name__ == "__main__":
    jobs = 10
    data = []
    for i in range(jobs):
        job = "{0:03}".format(i)
        # job = "{0:03}".format(200+i)
        with open("./results/pre-test/"+job+".pkl", 'rb') as f:
            file = pickle.load(f)
            d = Debug(file)
        data.append(d)

    print("data\t", len(data))
    
    for i, d in enumerate(data):
        print(d.print_init())
        print(d.status_difference)
        print(d.params['job_nr'])
        slopes = calculate_slopes(d.stress)
        
        mean_slopes = np.mean(slopes, axis=(0, 2))
        std_slopes = np.std(np.mean(slopes, axis=(0)), axis=(1))
        mean_stress = np.mean(d.stress, axis=(0, 2))
        std_stress = np.std(np.mean(d.stress, axis=(0)), axis=(1))
        
        plt.errorbar(x=np.arange(d.time), y=mean_stress, yerr = std_stress)
        plt.show()

        plt.errorbar(x=np.arange(d.time-1), y=mean_slopes, yerr = std_slopes)
        plt.show()
        
        
        plt.errorbar(x=np.arange(d.time), y=np.mean(d.stress[d.low_status,:,:], axis=(0, 2)), yerr = np.std(np.mean(d.stress[d.low_status,:,:], axis=(0)), axis=(1)))
        plt.errorbar(x=np.arange(d.time), y=np.mean(d.stress[d.med_status,:,:], axis=(0, 2)), yerr = np.std(np.mean(d.stress[d.med_status,:,:], axis=(0)), axis=(1)))
        plt.errorbar(x=np.arange(d.time), y=np.mean(d.stress[d.high_status,:,:], axis=(0, 2)), yerr = np.std(np.mean(d.stress[d.high_status,:,:], axis=(0)), axis=(1)))
        plt.show()        
        
        plt.errorbar(x=np.arange(d.time-1), y=np.mean(slopes[d.low_status,:,:], axis=(0, 2)), yerr = np.std(np.mean(slopes[d.low_status,:,:], axis=(0)), axis=(1)))
        plt.errorbar(x=np.arange(d.time-1), y=np.mean(slopes[d.med_status,:,:], axis=(0, 2)), yerr = np.std(np.mean(slopes[d.med_status,:,:], axis=(0)), axis=(1)))
        plt.errorbar(x=np.arange(d.time-1), y=np.mean(slopes[d.high_status,:,:], axis=(0, 2)), yerr = np.std(np.mean(slopes[d.high_status,:,:], axis=(0)), axis=(1)))
        plt.show()
        
        print(d.prestige.shape)
        print(np.unique(d.prestige))
        print(d.df['status'].unique())
        plt.hist(d.df['status'], bins=15)
        plt.show()
        # plt.scatter((np.array(d.stress.shape[2]*[d.df['status']]).T) , d.prestige[:, -1, :], alpha=0.2)
        plt.scatter(d.df['status'] , d.prestige[:, -1, 0], alpha=0.2)
        # plt.ylim([-1,2])
        # plt.yscale('symlog')
        plt.show()
        plt.scatter((np.array(d.stress.shape[2]*[d.df['status']]).T), d.stress[:, -1, :], alpha=0.2)
        # plt.xscale('symlog')
        plt.show()

# %%
