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
import seaborn as sns; sns.set()

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
        self.similarity = results['params']['similarity']
        self.time = results['params']['time']
        
        self.data = results
        
        if len(self.data['stress'].shape) != 1:
            self.datatype = 'grouped'
            self.stress = np.mean(self.data['stress'][:,:,:], axis=(0, 2)),
            self.stress_low = np.mean(self.data['stress'][self.low_status,:,:], axis=(0, 2)),
            self.stress_med = np.mean(self.data['stress'][self.med_status,:,:], axis=(0, 2)),
            self.stress_high = np.mean(self.data['stress'][self.high_status,:,:], axis=(0, 2)),
            self.stress_std = np.std(np.mean(self.data['stress'][:,:,:], axis=0), axis=1),
            self.stress_low_std = np.std(np.mean(self.data['stress'][self.low_status,:,:], axis=0), axis=1),
            self.stress_med_std = np.std(np.mean(self.data['stress'][self.med_status,:,:], axis=0), axis=1),
            self.stress_high_std = np.std(np.mean(self.data['stress'][self.high_status,:,:], axis=0), axis=1),      

            self.chronic = np.mean(self.data['chronic'][:,:,:], axis=(0, 2)),
            self.chronic_low = np.mean(self.data['chronic'][self.low_status,:,:], axis=(0, 2)),
            self.chronic_med = np.mean(self.data['chronic'][self.med_status,:,:], axis=(0, 2)),
            self.chronic_high = np.mean(self.data['chronic'][self.high_status,:,:], axis=(0, 2)),
            self.chronic_std = np.std(np.mean(self.data['chronic'][:,:,:], axis=0), axis=1),
            self.chronic_low_std = np.std(np.mean(self.data['chronic'][self.low_status,:,:], axis=0), axis=1),
            self.chronic_med_std = np.std(np.mean(self.data['chronic'][self.med_status,:,:], axis=0), axis=1),
            self.chronic_high_std = np.std(np.mean(self.data['chronic'][self.high_status,:,:], axis=0), axis=1),
            self.params = self.params, 
            self.df = self.df[['status', 'psr', 'eth','prestige']],            
                     
        else:
            self.stress = self.data['stress'],
            self.stress_agents = self.data['stress_agents']
            self.stress_std = self.data['stress_std'],
            self.stress_low = self.data['stress_low'],
            self.stress_med = self.data['stress_med'],
            self.stress_high = self.data['stress_high'],
            self.stress_low_std = self.data['stress_low_std'],
            self.stress_med_std = self.data['stress_med_std'],
            self.stress_high_std = self.data['stress_high_std'],

            self.chronic = self.data['chronic'],
            self.chronic_agents = self.data['chronic_agents']
            self.chronic_std = self.data['chronic_std'],
            self.chronic_low = self.data['chronic_low'],
            self.chronic_med = self.data['chronic_med'],
            self.chronic_high = self.data['chronic_high'],
            self.chronic_low_std = self.data['chronic_low_std'],
            self.chronic_med_std = self.data['chronic_med_std'],
            self.chronic_high_std = self.data['chronic_high_std'],

            self.prestige = self.data['prestige'],
            self.prestige_agents = self.data['prestige_agents']
            self.prestige_std = self.data['prestige_std'],
            self.prestige_low = self.data['prestige_low'],
            self.prestige_med = self.data['prestige_med'],
            self.prestige_high = self.data['prestige_high'],
            self.prestige_low_std = self.data['prestige_low_std'],
            self.prestige_med_std = self.data['prestige_med_std'],
            self.prestige_high_std = self.data['prestige_high_std'],
            self.params = self.params, 
            self.df = self.df[['status', 'psr', 'eth','prestige']],
            self.df = self.df[0]            
             
        self.no_agents = self.df.shape[0]
        self.low_status, self.med_status, self.high_status = self.split_population_status()
        # self.status_difference = results['status_difference']
        # self.stressors = results['stressors']
        # self.coped_stress = results['coped_stress']
        # self.similarity_interactions = results['similarity_interactions']
        
        # self.status = self.df['status'].values
        
        # self.stress_ts = np.array(results['stress'], dtype=np.float32)
        # self.stress_mean = np.mean(self.stress_ts, axis = (1))
        # self.stress_std = np.std(self.stress_ts, axis = (1))
        # self.stress_ts_mean = np.mean(self.stress_ts, axis=(0))
        # self.stress_ts_std = np.std(self.stress_ts, axis = (0))
        
        
        # self.prestige_ts = np.array(results['prestige'], dtype=np.float32)
        # self.prestige_mean = np.mean(self.prestige_ts, axis = (1))
        # self.prestige_ts_mean = np.mean(self.stress_ts, axis=(0))
        # self.prestige_ts_std = np.std(self.prestige_ts, axis=(0))
                
        
        # self.chronic_ts = np.array(results['chronic'], dtype=np.float32)
        # self.chronic_mean = np.mean(self.chronic_ts, axis = (1))
        # self.chronic_ts_mean = np.mean(self.chronic_ts, axis=(0))            
        
        # self.no_agents = self.population.shape[0]
        # self.idx_population = np.arange(self.population.shape[0])
        
        # self.low_psr, self.med_psr, self.high_psr = self.split_population_psr()
        
        # self.df['stress'] = self.stress_mean
    
    
        
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

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def prior_analysis(folder, index):
    print("prior_analysis")
    plt.rcParams['figure.figsize'] = [12,3]


    accepted = [] 
    rejected = []

    files = np.array(glob.glob(root+ "results/*"+folder+"*/*.pkl"))
    print(folder)
    print("data files: ", len(files))
    for file in files:
        # print(file)
        with open(file, 'rb') as f:
            results = pickle.load(f)
        d = Analyzer(results)

        # print(np.mean(d.stress))
        mean_stress = np.mean(d.stress)
        mean_stress_check = mean_stress > 0.01 
        if not mean_stress_check: rejected.append(file); continue; 
        
        median_stress = np.median(d.stress)
        median_stress_check = median_stress > 0.01
        if not median_stress_check: rejected.append(file); continue; 

        
        mean_chronic = np.mean(d.data['chronic'])
        mean_chronic_check = mean_chronic > 0.001 
        if not mean_chronic_check: rejected.append(file); continue; 

        
        mean_stress_low = np.mean(d.stress_low)
        mean_stress_med = np.mean(d.stress_med)
        mean_stress_high = np.mean(d.stress_high)
        mean_stress_groups_check =  mean_stress_low > mean_stress_med > mean_stress_high
        if not mean_stress_groups_check: rejected.append(file); continue; 

             
        mean_chronic_low = np.mean(d.chronic_low)
        mean_chronic_med = np.mean(d.chronic_med)
        mean_chronic_high = np.mean(d.chronic_high)
        mean_chronic_groups_check =  mean_chronic_low > mean_chronic_med >= mean_chronic_high
        if not mean_chronic_groups_check: rejected.append(file); continue; 

                
        
        
        accepted.append(file)
        # del d
        
    print("accepted: ", len(accepted))
    print("rejected: ", len(rejected))
    output = os.path.dirname(os.path.realpath(__file__)) + '\\results\\rejacc\\folder_'+ folder + ".pkl"
    f = open(output, "wb")
    pickle.dump([accepted, rejected], f)
    print("PRIOR ANALYSIS SAVED")

        
def explore_accepted(folder):
    plt.rcParams['figure.figsize'] = [12, 3]

   
    with open("./results/rejacc/folder_" + folder +'.pkl', 'rb') as f:
        files = pickle.load(f)[0]

    print("files: ", len(files)) 
    for file in files[0:10]:
        with open(file, 'rb') as f:
            results = pickle.load(f)
        d = Analyzer(results)
        
        print("ALL MEAN/STD: \t", np.mean(d.stress)/np.std(d.stress),"\t",np.mean(d.chronic)/np.std(d.chronic))
        print("LOW MEAN/STD: \t", np.mean(d.stress_low)/np.std(d.stress_low),"\t",np.mean(d.chronic_low)/np.std(d.chronic_low))
        print("MED MEAN/STD: \t", np.mean(d.stress_med)/np.std(d.stress_med),"\t",np.mean(d.chronic_med)/np.std(d.chronic_med))
        print("HIGH MEAN/STD: \t", np.mean(d.stress_high)/np.std(d.stress_high),"\t",np.mean(d.chronic_high)/np.std(d.chronic_high))
        
        
        # print(type(d.stress[0]))
        # print(type(d.stress_std[0]))
        
        # plt.errorbar(x=np.arange(d.time), y=d.stress[0], yerr=d.stress_std[0])
        # plt.show()
        plt.scatter(d.df['status'], np.mean(d.stress_agents, axis=1), alpha=0.5)
        plt.show()
        
        plt.errorbar(x=np.arange(d.time), y=d.stress_low[0], yerr=d.stress_low_std[0])
        plt.errorbar(x=np.arange(d.time), y=d.stress_med[0], yerr=d.stress_med_std[0])
        plt.errorbar(x=np.arange(d.time), y=d.stress_high[0], yerr=d.stress_high_std[0])
        plt.show()
        
        plt.errorbar(x=np.arange(d.time), y=d.chronic_low[0], yerr=d.chronic_low_std[0])
        plt.errorbar(x=np.arange(d.time), y=d.chronic_med[0], yerr=d.chronic_med_std[0])
        plt.errorbar(x=np.arange(d.time), y=d.chronic_high[0], yerr=d.chronic_high_std[0])
        plt.show()
        
        # plt.scatter()
        # plt.show()
        # plt.errorbar(x=np.arange(d.time), y=d.data["chronic_low"], yerr=d.data['chronic_low_std'])
        # plt.errorbar(x=np.arange(d.time), y=d.data["chronic_med"], yerr=d.data['chronic_med_std'])
        # plt.errorbar(x=np.arange(d.time), y=d.data["chronic_high"], yerr=d.data['chronic_high_std'])
        # plt.show()
        df = pd.DataFrame()
        for i in np.arange(0, 500, 50):
            df["{0:03}".format(i)] = d.stress_agents[:, i]
        dfm = df.melt()
        dfm.sort_values(by="variable", inplace=True) 
        # # print(dfm.columns)
        ax = sns.violinplot(x="variable", y="value", data=dfm, cut=0)
        plt.ylabel("stress")
        plt.show()
        
        df = pd.DataFrame()
        for i in np.arange(0, 500, 50):
            df["{0:03}".format(i)] = d.chronic_agents[:, i]
        dfm = df.melt()
        dfm.sort_values(by="variable", inplace=True) 
        # # print(dfm.columns)
        ax = sns.violinplot(x="variable", y="value", data=dfm, cut=0)
        plt.ylabel("chronic")
        plt.show()        
        # plt.hist(d.stress_mean)
        # print("mean stress:\t\t", mean_stress, "\t"+str(mean_stress_check))
        # print("median stress:\t\t", median_stress, "\t"+str(median_stress_check))
        # print("mean chronic:\t\t", mean_chronic, "\t"+str(mean_chronic_check))
        # print()
        # print("mean stress low:\t", mean_stress_low) 
        # print("mean stress med:\t", mean_stress_med, "\t"+str(mean_stress_groups_check)) 
        # print("mean stress high:\t", mean_stress_high) 
        # print()
        # print("mean chronic low:\t", mean_chronic_low) 
        # print("mean chronic med:\t", mean_chronic_med, "\t"+str(mean_chronic_groups_check))
        # print("mean chronic high:\t", mean_chronic_high) 
        # print()
        # print("max prestige:\t\t", max_prestige, "\t"+str(max_presitge_check))
        # stress = np.mean(d.stress_ts, axis=2)
        
      
        # plt.errorbar(x=np.arange(d.time), y=d.stress, yerr=np.std(np.mean(d.stress_ts[d.low_status,:,:], axis=0), axis=1), c='r')
        
        # y = np.mean(stress_med, axis=1)
        # yerr = np.std(stress_med, axis=1)
        # plt.errorbar(x=np.arange(len(stress_med)), y=y, yerr=yerr, c='b')
        
        # y = np.mean(stress_high, axis=1)
        # yerr = np.std(stress_high, axis=1)
        # plt.errorbar(x=np.arange(len(stress_high)), y=y, yerr=yerr, c='g')
        # plt.show()
        # print(stress.shape)
        # df = pd.DataFrame()         
        # columns = []

        
        # for i in np.arange(0, 500, 50):
        #     df["{0:03}".format(i)] = d.chronic_ts[:, i]
        # dfm = df.melt()
        # dfm.sort_values(by="variable", inplace=True) 
        # # print(dfm.columns)
        # ax = sns.violinplot(x="variable", y="value", data=dfm, cut=0)
        # plt.ylabel("chronic_state")
        
        # plt.show()
        
        # for i in np.arange(0, 500, 50):
        #     df["{0:03}".format(i)] = d.chronic_i_ts[:, i]
        # dfm = df.melt()
        # dfm.sort_values(by="variable", inplace=True) 
        # # print(dfm.columns)
        # ax = sns.violinplot(x="variable", y="value", data=dfm, cut=0)
        # plt.ylabel("chronic_intensity")
        # plt.show()
        
        # plt.plot(stress, alpha=0.1, c='black')
        # plt.plot(np.mean(d.stress_ts[d.low_status], axis=0), c='r')
        # plt.show()
        del d
        plt.close('all')
        print("\n\n\n")
   
        
if __name__ == "__main__":
    folder = "pre-test"
    # folder = "pre_test_200repsB"
    index = []
    prior_analysis(folder=folder, index=index)
    explore_accepted(folder=folder)
    
    # explore(folder="debug", index=[0])
    
    # analyse_functionality(folder="single", index=[0])
    # index = np.arange(1150, 1200)
    # index = [
    #     216, 377, 924,935,946,957,968,979
    # ]
    # index = np.arange(0,100)
    # index = np.arange(0, 1000, 20)
    # i = 3
    # print(index[i], index[i+1])
    # explore(folder="wrong*", index=[])
    # explore(folder="best", index=[])
    # get_parameters(folder="keep", index=[])
    # analyze_all_in_foler("single")
    # plt.plot(data.stress_ts_mean)
    # plt.show()
    # plt.plot(data.chronic_ts_mean)
# %%

# %%
