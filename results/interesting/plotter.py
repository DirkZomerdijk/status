#%%
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from global_variables import *
from functions import calc_vulnerability
import glob
import os
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class Analyzer():

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
        self.population = results['params']['population']
        self.similarity = results['params']['similarity']
        self.time = results['params']['time']
        self.no_agents = self.population.shape[0]
        
        # self.status_difference = results['status_difference']
        # self.stressors = results['stressors']
        # self.coped_stress = results['coped_stress']
        # self.similarity_interactions = results['similarity_interactions']
        
        self.status = self.df['status'].values
        
        self.stress_m = np.array(results['stress'], dtype=np.float32)
        self.stress_mean = np.mean(self.stress_m, axis = (1, 2))
        self.stress_ts = np.mean(self.stress_m, axis=2)
        self.stress_ts_mean = np.mean(self.stress_m, axis=(0,2))
        
        self.chronic_m = results['chronic_state']
        self.chronic_mean = np.mean(self.chronic_m, axis = (1, 2))
        self.chronic_ts = np.mean(self.chronic_m, axis=2)
        self.chronic_ts_mean = np.mean(self.chronic_m, axis=(0,2))    
        
        self.chronic_i_ts = np.array(np.mean([np.multiply(self.chronic_m[:,:,i], self.stress_m[:,:,i]) for i in range(self.repeats)], axis=0))
        print(self.chronic_i_ts.shape)
        # print(self.chronic_m[0,:,0])
        # print(self.stress_m[0,:,0])
        # self.chronic_intensities = np.multiply(self.chronic_m)
        
        self.no_agents = self.population.shape[0]
        self.idx_population = np.arange(self.population.shape[0])
        
        self.low_status, self.med_status, self.high_status = self.split_population_status()
        self.low_psr, self.med_psr, self.high_psr = self.split_population_psr()
        
        self.df['stress'] = self.stress_mean
        
        
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

    def stress_ts_class(self):
        plt.plot(np.arange(self.time), np.mean(self.stress_ts[self.low_status], axis=0), label="low")
        plt.plot(np.arange(self.time), np.mean(self.stress_ts[self.med_status], axis=0), label="med")
        plt.plot(np.arange(self.time), np.mean(self.stress_ts[self.high_status], axis=0), label="high")
        plt.ylabel('stress')
        plt.legend()
        plt.show()
    
    def chronic_ts_class(self):
        plt.plot(np.arange(self.time), np.mean(self.chronic_ts[self.low_status], axis=0), label="low")
        plt.plot(np.arange(self.time), np.mean(self.chronic_ts[self.med_status], axis=0), label="med")
        plt.plot(np.arange(self.time), np.mean(self.chronic_ts[self.high_status], axis=0), label="high")
     
        plt.ylabel('chronic state')
        plt.legend()
        plt.show()

    def chronic_i_ts_class(self):
        idx = np.argsort(self.df['status'])
        plt.plot(np.arange(self.time), np.mean(self.chronic_i_ts[self.low_status], axis=0), label="low")
        plt.plot(np.arange(self.time), np.mean(self.chronic_i_ts[self.med_status], axis=0), label="med")
        plt.plot(np.arange(self.time), np.mean(self.chronic_i_ts[self.high_status], axis=0), label="high")
     
        plt.ylabel('chronic intensity')
        plt.legend()
        plt.show()
        
    def distribution_stress(self):
        idx = np.argsort(self.df['status'])
        low = np.mean(self.stress_ts[self.low_status, int(self.time/2):], axis=1)
        med = np.mean(self.stress_ts[self.med_status, int(self.time/2):], axis=1)
        high = np.mean(self.stress_ts[self.high_status, int(self.time/2):], axis=1)

        
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize = (8,2))
        bins=np.histogram(np.hstack((low,med,high)), bins=100)[1] #get the bin edges

        axs[0].hist(low, bins)
        axs[1].hist(med, bins)
        axs[2].hist(high, bins)
        plt.show()

    def distribution_chronic(self):
        idx = np.argsort(self.df['status'])
        low = np.mean(self.chronic_ts[self.low_status, int(self.time/2):], axis=1)
        med = np.mean(self.chronic_ts[self.med_status, int(self.time/2):], axis=1)
        high = np.mean(self.chronic_ts[self.high_status, int(self.time/2):], axis=1)
        
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize = (8,2))
        
        bins=np.histogram(np.hstack((low,med,high)), bins=100)[1] #get the bin edges

        axs[0].hist(low, bins)
        axs[1].hist(med, bins)
        axs[2].hist(high, bins)
        plt.show()

    def distribution_status(self):
        idx = np.argsort(self.df['status'])
        print(len(idx))
        low = self.status[self.low_status]
        med = self.status[self.med_status]
        high = self.status[self.high_status]
        
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize = (8,2))
        bins=np.histogram(np.hstack((low,med,high)), bins=20)[1] #get the bin edges

        axs[0].hist(low, bins)
        axs[1].hist(med, bins)
        axs[2].hist(high, bins)
        plt.show()
    
    def scatter(self, x, y, title="", xlabel="", ylabel="", alpha=0.2, scale="linear"):
        plt.scatter(x,y, alpha=alpha)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        # plt.plot(x,p(x),"--")
        # print(p)
        
        plt.yscale(scale)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        
    def plot(self, x):
        plt.plot(x)
        plt.show()
        
    def bar_status_stress(self):
        low_mean = self.df.iloc[self.low_status]['stress'].mean()
        med_mean = self.df.iloc[self.med_status]['stress'].mean()
        high_mean = self.df.iloc[self.high_status]['stress'].mean()
        
        low_std = self.df.iloc[self.low_status]['stress'].std()
        med_std = self.df.iloc[self.med_status]['stress'].std()
        high_std = self.df.iloc[self.high_status]['stress'].std()
        
        conditions = ['low', 'med', 'high']
        x_pos = np.arange(len(conditions))
        mean = [low_mean, med_mean, high_mean]
        error = [low_std, med_std, high_std]
        
        fig, ax = plt.subplots()
        ax.bar(x_pos, mean, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Stress')
        ax.set_xlabel('Status')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(conditions)
        ax.yaxis.grid(True)
        plt.tight_layout()
        plt.show()

    def gbar_status_psr(self):
       
        low_status = self.df.iloc[self.low_status]
        med_status = self.df.iloc[self.med_status]
        high_status = self.df.iloc[self.high_status]
        
        low_status_low_psr = low_status.loc[low_status['psr'] <= 0.33, 'stress'].mean()
        low_status_med_psr = low_status.loc[(low_status['psr'] > 0.33) & (low_status['psr'] <= 0.67), 'stress'].mean()
        low_status_high_psr = low_status.loc[low_status['psr'] > 0.67, 'stress'].mean()

        med_status_low_psr = med_status.loc[med_status['psr'] <= 0.33, 'stress'].mean()
        med_status_med_psr = med_status.loc[(med_status['psr'] > 0.33) & (med_status['psr'] <= 0.67), 'stress'].mean()
        med_status_high_psr = med_status.loc[med_status['psr'] > 0.67, 'stress'].mean()
                
        high_status_low_psr = high_status.loc[high_status['psr'] <= 0.33, 'stress'].mean()
        high_status_med_psr = high_status.loc[(high_status['psr'] > 0.33) & (high_status['psr'] <= 0.67), 'stress'].mean()
        high_status_high_psr = high_status.loc[high_status['psr'] > 0.67, 'stress'].mean()
        
        pd.DataFrame({
            'low psr': [low_status_low_psr,med_status_low_psr, high_status_low_psr],
            'med psr': [low_status_med_psr,med_status_med_psr, high_status_med_psr],  
            'high psr': [low_status_high_psr, med_status_high_psr, high_status_high_psr]  
        }, index=["low_status", "mid_status", "high_status"]).plot.bar(rot=0)

        plt.ylabel('stress')
        plt.show()
  
def analyse_functionality(folder, index):
    data = open_data_objects(folder=folder, index=index)[0]
    
    plt.hist(data.status_difference, bins=100)
    plt.title("status difference occurence")
    plt.xlabel('status difference')
    plt.ylabel('frequency')
    plt.show()
    
    plt.hist(data.stressors, bins=100)
    plt.title("stressor occurence")
    plt.xlabel('stressor')
    plt.ylabel('frequency')
    plt.show()
    
    plt.hist(data.coped_stress, bins=100)
    plt.title("coped stress occurence")
    plt.xlabel('coped stress')
    plt.ylabel('frequency')
    plt.show()
    
    plt.hist(data.df['prestige'], bins=100)
    plt.title("presitge occurence")
    plt.xlabel('presitge')
    plt.ylabel('frequency')
    plt.show()
    
    plt.hist(data.similarity_interactions, bins=100)
    plt.title("similarity interactions occurence")
    plt.xlabel('similarity interactions')
    plt.ylabel('frequency')
    plt.show()
     
def analyse_single(folder, index):
    data = open_data_objects(folder=folder, index=index)[0]
    
    # data.bar_status_stress()
    # data.gbar_status_psr()
    # data.stress_ts_class()
    # data.chronic_ts_class()     
    # data.chronic_i_ts_class()     
    # data.scatter(data.df['status'], data.stress_mean, xlabel="status", ylabel="stress")
    # data.scatter(data.df['status'] + data.df['prestige'], data.stress_mean, xlabel="status + prestige", ylabel="stress")
    # data.scatter(data.df['status'], data.df['prestige'], xlabel="status + prestige", ylabel="stress")
    # data.scatter(data.df['status'], data.chronic_mean, xlabel="status", ylabel="chronic")    
    
def open_last_data_object(experiment, folder):
    data = []
    file = glob.glob(root+ "results/*"+folder+"*/*"+experiment+"*.pkl")
    print(file[-1])
    with open(file[-1], 'rb') as f:
        results = pickle.load(f)
    return Analyzer(results)

def open_data_objects(folder, index=[]):
    data = []
    # print(root+ "results/*"+folder+"*/*"+experiment+"*.pkl")
    files = np.array(glob.glob(root+ "results/*"+folder+"*/*.pkl"))
    print(folder, index)
    if(len(index) == 0):
        for file in files:
            with open(file, 'rb') as f:
                results = pickle.load(f)
            data.append(Analyzer(results))
    else:
        for file in files[index]:
            with open(file, 'rb') as f:
                results = pickle.load(f)
            data.append(Analyzer(results))
    return data


def analyze_all_in_foler(folder):
    data = open_data_objects(folder=folder, index=[])

    for d in data:
        d.print_init()
        d.stress_ts_class()
        d.chronic_ts_class()    
        d.distribution_chronic()
        d.distribution_stress()
        d.distribution_status()
        d.stress_ts_class()
        d.chronic_ts_class()


def explore(folder, index):
    plt.rcParams['figure.figsize'] = [3, 3]

    data = open_data_objects(folder=folder, index=index)

    for i, d in enumerate(data):
        d.print_init()
        d.stress_ts_class()
        d.chronic_ts_class()     
        d.chronic_i_ts_class()     
        d.scatter(d.df['status'], d.stress_mean, xlabel="status", ylabel="stress")
        d.scatter(d.df['status'] + d.df['prestige'], d.stress_mean, xlabel="status + prestige", ylabel="stress")
        d.scatter(d.df['status'], d.df['prestige'], xlabel="status", ylabel="prestige")
        d.scatter(d.df['status'], d.chronic_mean, xlabel="status", ylabel="chronic")
        

def get_parameters(folder, index):
    data = open_data_objects(folder=folder, index=index)

    
    p = {
        'similarity_base':[], 
        'ses_noise':[], 
        'stressor_param':[], 
        'vul_param':[], 
        'psr_param':[], 
        'recover_param':[], 
        'prestige_param':[], 
        'prestige_beta':[], 
        }
    
    for d in data:
        for k in p.keys():
            p[k].append(d.params[k])
    for k in p.keys():
        a = plt.hist(p[k], bins=100)
        # print(a)
        print(a[1][np.nonzero(a[0])])
        plt.title(k)
        plt.show()
        
if __name__ == "__main__":
    # analyse_functionality(folder="single", index=[0])
    # index = np.arange(1150, 1200)
    # index = [
    #     216, 377, 924,935,946,957,968,979
    # ]
    # index = np.arange(0,100)
    index = np.arange(0, 1000, 20)
    i = 2
    print(index[i], index[i+1])
    explore(folder="pre-test", index=np.arange(index[i], index[i+1]))
    # explore(folder="keep", index=[])
    # explore(folder="best", index=[])
    # get_parameters(folder="keep", index=[])
    # analyze_all_in_foler("single")
    # plt.plot(data.stress_ts_mean)
    # plt.show()
    # plt.plot(data.chronic_ts_mean)
# %%

# %%
