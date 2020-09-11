#%%
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from global_variables import *
from model_functions import get_vulnerability, calculate_chronic_state, normalize
import glob
import os
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns; sns.set()
import itertools
import copy

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
        
        # if len(self.data['stress'].shape) != 1:
        #     self.datatype = 'grouped'
        #     self.stress = np.mean(self.data['stress'][:,:,:], axis=(0, 2)),
        #     self.stress_low = np.mean(self.data['stress'][self.low_status,:,:], axis=(0, 2)),
        #     self.stress_med = np.mean(self.data['stress'][self.med_status,:,:], axis=(0, 2)),
        #     self.stress_high = np.mean(self.data['stress'][self.high_status,:,:], axis=(0, 2)),
        #     self.stress_std = np.std(np.mean(self.data['stress'][:,:,:], axis=0), axis=1),
        #     self.stress_low_std = np.std(np.mean(self.data['stress'][self.low_status,:,:], axis=0), axis=1),
        #     self.stress_med_std = np.std(np.mean(self.data['stress'][self.med_status,:,:], axis=0), axis=1),
        #     self.stress_high_std = np.std(np.mean(self.data['stress'][self.high_status,:,:], axis=0), axis=1),      

        #     self.chronic = np.mean(self.data['chronic'][:,:,:], axis=(0, 2)),
        #     self.chronic_low = np.mean(self.data['chronic'][self.low_status,:,:], axis=(0, 2)),
        #     self.chronic_med = np.mean(self.data['chronic'][self.med_status,:,:], axis=(0, 2)),
        #     self.chronic_high = np.mean(self.data['chronic'][self.high_status,:,:], axis=(0, 2)),
        #     self.chronic_std = np.std(np.mean(self.data['chronic'][:,:,:], axis=0), axis=1),
        #     self.chronic_low_std = np.std(np.mean(self.data['chronic'][self.low_status,:,:], axis=0), axis=1),
        #     self.chronic_med_std = np.std(np.mean(self.data['chronic'][self.med_status,:,:], axis=0), axis=1),
        #     self.chronic_high_std = np.std(np.mean(self.data['chronic'][self.high_status,:,:], axis=0), axis=1),
        #     self.params = self.params, 
        #     self.df = self.df[['status', 'psr', 'eth','prestige']],            
                     
        # else:
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

        self.chronic_i = self.data['chronic_i'],
        self.chronic_i_agents = self.data['chronic_i_agents']
        self.chronic_i_std = self.data['chronic_i_std'],
        self.chronic_i_low = self.data['chronic_i_low'],
        self.chronic_i_med = self.data['chronic_i_med'],
        self.chronic_i_high = self.data['chronic_i_high'],
        self.chronic_i_low_std = self.data['chronic_i_low_std'],
        self.chronic_i_med_std = self.data['chronic_i_med_std'],
        self.chronic_i_high_std = self.data['chronic_i_high_std'],
        
        self.prestige = self.data['prestige'],
        self.prestige_agents = self.data['prestige_agents']
        self.prestige_std = self.data['prestige_std'],
        self.prestige_low = self.data['prestige_low'],
        self.prestige_med = self.data['prestige_med'],
        self.prestige_high = self.data['prestige_high'],
        self.prestige_low_std = self.data['prestige_low_std'],
        self.prestige_med_std = self.data['prestige_med_std'],
        self.prestige_high_std = self.data['prestige_high_std'],
        
        self.interactions = self.data['interactions'],
        self.interactions_agents = self.data['interactions_agents']
        self.interactions_std = self.data['interactions_std'],
        self.interactions_low = self.data['interactions_low'],
        self.interactions_med = self.data['interactions_med'],
        self.interactions_high = self.data['interactions_high'],
        self.interactions_low_std = self.data['interactions_low_std'],
        self.interactions_med_std = self.data['interactions_med_std'],
        self.interactions_high_std = self.data['interactions_high_std'],
        
        self.df = self.df[['status', 'psr', 'eth']],
        self.df = self.df[0]            
            
        self.params = self.params, 
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
        
        
        # self.interactions_ts = np.array(results['prestige'], dtype=np.float32)
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
    root = os.path.dirname(os.path.realpath(__file__)) 
    print(root+ "\\results\\"+folder+"\\*.pkl")
    files = np.array(glob.glob(root+ "\\results\\"+folder+"\\*.pkl"))
    print(folder)
    print("data files: ", len(files))
    
    p = param_dict
    params_per_model = np.empty(shape=(len(files), len(p.keys())))
    
    
    for i, file in enumerate(files):
        print(file)
        with open(file, 'rb') as f:
            results = pickle.load(f)
        
        d = Analyzer(results)
        
        for j, k in enumerate(p.keys()):
            params_per_model[i, j] = d.params[0][k]
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
    
    
    # n = normalize(params_per_model[:,9], 0, 1)
    # eds = np.empty(shape=(len(files),len(files)))
    # print(params_per_model.shape, n.shape)
    
    # for i in range(params_per_model.shape[1]):
    #     values = np.asarray(params_per_model[:, i], dtype=np.float64)
    #     values = scale(values)

    #     params_per_model[:, i] = values
    #     # plt.hist(values)
    #     # plt.title(list(p.keys())[i])
    #     # plt.show()
        
    # combos = list(itertools.combinations(range(len(files)), 2))
    # for combo in combos:
    #     eds[combo[0], combo[1]] = np.linalg.norm(params_per_model[combo[0], :] - params_per_model[combo[1], :])
    
    # plt.figure(figsize=(12,12))
    # plt.imshow(eds, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.show()
    print("PRIOR ANALYSIS SAVED")

def get_paramters_rejected():
    rejaccs = glob.glob("./results/rejacc/folder_*.pkl")
    rejects = []
    accepts = []
    p_r = copy.deepcopy(param_dict)
    p_a = copy.deepcopy(param_dict)
    
    for rejacc in rejaccs:
        with open(rejacc, 'rb') as f:
            files = pickle.load(f)
            accepts += files[0]
            rejects += files[1]

    print("rejects: ", len(rejects))
    for reject in rejects:
        with open(reject, 'rb') as f:
            results = pickle.load(f)        
        for k in p_r.keys():
            p_r[k].append(results["params"][k])
            
    for accept in accepts:
        with open(accept, 'rb') as f:
            results = pickle.load(f)        
        for k in p_r.keys():
            p_a[k].append(results["params"][k])
    
    for k in p_r.keys():
        fig, axs = plt.subplots(1, 2, sharex=True)
        bins=np.histogram(np.hstack((p_r[k],p_a[k])), bins=50)[1] #get the bin edges

        axs[0].hist(p_r[k], bins=bins)
        # plt.title("accept: " + k)
        axs[1].hist(p_a[k], bins=bins)
        plt.title(k)
        plt.show()      
        
def explore_accepted(folder):
    plt.rcParams['figure.figsize'] = [12, 3]

    p = param_dict
    with open("./results/rejacc/folder_" + folder +'.pkl', 'rb') as f:
        files = pickle.load(f)[1]

    print()
    print("files: ", len(files)) 
    for file in files:
        print(file)
        with open(file, 'rb') as f:
            results = pickle.load(f)
        d = Analyzer(results)
        print("obtained results...")
        
        print(d.stress_agents[d.low_status, :].shape)
        plt.plot(d.stress_agents[d.low_status, :].T, c='k', alpha=0.05)
        # plt.plot(np.mean(d.stress_agents[d.low_status], axis=0))
        plt.show()   
        # plt.scatter(d.df['status'].iloc[d.low_status] + d.prestige_agents[d.low_status,-1], np.mean(d.stress_agents[d.low_status], axis=1), alpha=0.3, c='b')
        # plt.scatter(d.df['status'].iloc[d.med_status] + d.prestige_agents[d.med_status,-1], np.mean(d.stress_agents[d.med_status], axis=1), alpha=0.3, c='r')
        # plt.scatter(d.df['status'].iloc[d.high_status] + d.prestige_agents[d.high_status,-1], np.mean(d.stress_agents[d.high_status], axis=1), alpha=0.3, c='g')
        # plt.xlabel('status + prestige') 
        # plt.ylabel('stress') 
        # plt.show()
        
        # plt.scatter(d.df['status'], d.prestige_agents[:, -1], alpha=0.3)   
        # plt.xlabel('status') 
        # plt.ylabel('prestige') 
        # plt.show()
        
        # plt.scatter(d.df['status'].iloc[d.low_status] + d.prestige_agents[d.low_status,-1], np.mean(d.interactions_agents[d.low_status], axis=1), alpha=0.3, c='b')
        # plt.scatter(d.df['status'].iloc[d.med_status] + d.prestige_agents[d.med_status,-1], np.mean(d.interactions_agents[d.med_status], axis=1), alpha=0.3, c='r')
        # plt.scatter(d.df['status'].iloc[d.high_status] + d.prestige_agents[d.high_status,-1], np.mean(d.interactions_agents[d.high_status], axis=1), alpha=0.3, c='g')
        # plt.xlabel('status + prestige') 
        # plt.ylabel('interactions') 
        # plt.show()
        
        # stat = d.df['status']
        # prest = d.prestige_agents[:, -1] + stat
        # bins=np.histogram(np.hstack((stat,prest)), bins=50)[1] #get the bin edges
        # plt.hist(stat, bins, color='b', alpha=0.5)
        # plt.hist(prest, bins, color='r', alpha=0.5)
        # plt.show()
        
        # stress_end = d.stress_agents[:, d.time-1].flatten()
        # plt.hist(np.array(d.stress_agents[:, d.time-1].flatten(), dtype=np.float32), bins=100)
        # plt.xlabel('stress')
        # plt.show()
        
        # plt.errorbar(x=np.arange(d.time), y=d.stress_low[0], yerr=d.stress_low_std[0])
        # plt.errorbar(x=np.arange(d.time), y=d.stress_med[0], yerr=d.stress_med_std[0])
        # plt.errorbar(x=np.arange(d.time), y=d.stress_high[0], yerr=d.stress_high_std[0])
        # plt.ylabel("stress")
        # plt.show()
        
        # plt.errorbar(x=np.arange(d.time), y=d.chronic_low[0], yerr=d.chronic_low_std[0])
        # plt.errorbar(x=np.arange(d.time), y=d.chronic_med[0], yerr=d.chronic_med_std[0])
        # plt.errorbar(x=np.arange(d.time), y=d.chronic_high[0], yerr=d.chronic_high_std[0])
        # plt.ylabel("chronic")
        # plt.show()
        
        # plt.errorbar(x=np.arange(d.time), y=d.chronic_i_low[0], yerr=d.chronic_i_low_std[0])
        # plt.errorbar(x=np.arange(d.time), y=d.chronic_i_med[0], yerr=d.chronic_i_med_std[0])
        # plt.errorbar(x=np.arange(d.time), y=d.chronic_i_high[0], yerr=d.chronic_i_high_std[0])
        # plt.show()
        
        # plt.bar(["low", "med", "high"], [np.mean(d.stress_low),np.mean(d.stress_med),np.mean(d.stress_high)],  yerr = [np.std(d.stress_low),np.std(d.stress_med),np.std(d.stress_high)])
        # plt.show()
        # df = pd.DataFrame()
        # for i in np.arange(0, 1000, 50):
        #     df["{0:03}".format(i)] = d.stress_agents[:, i]
        # dfm = df.melt()
        # dfm.sort_values(by="variable", inplace=True) 
        # # # print(dfm.columns)
        # ax = sns.violinplot(x="variable", y="value", data=dfm, cut=0)
        # plt.ylabel("stress")
        # plt.show()
        
        # df = pd.DataFrame()
        # for i in np.arange(0, 1000, 50):
        #     df["{0:03}".format(i)] = d.chronic_agents[:, i]
        # dfm = df.melt()
        # dfm.sort_values(by="variable", inplace=True) 
        # # # print(dfm.columns)
        # ax = sns.violinplot(x="variable", y="value", data=dfm, cut=0)
        # plt.ylabel("chronic")
        # plt.show()
                
        
        # df = pd.DataFrame()
        # for i in np.arange(0, 500, 25):
        #     df["{0:03}".format(i)] = d.chronic_agents[d.low_status, i]
        # dfm = df.melt()
        # dfm.sort_values(by="variable", inplace=True) 
        # # # print(dfm.columns)
        # ax = sns.violinplot(x="variable", y="value", data=dfm, cut=0)
        # plt.ylabel("chronic state")
        # plt.show()
        
        # df = pd.DataFrame()
        # for i in np.arange(0, 500, 50):
        #     df["{0:03}".format(i)] = d.chronic_agents[:, i]
        # dfm = df.melt()
        # dfm.sort_values(by="variable", inplace=True) 
        # # # print(dfm.columns)
        # ax = sns.violinplot(x="variable", y="value", data=dfm, cut=0)
        # plt.ylabel("chronic")
        # plt.show()        


    
        del d
        plt.close('all')
        print("\n")
        
if __name__ == "__main__":
    folder = "pre_testC"
    folder = "debug"
    # folder = "pre-test"
    
    index = []
    prior_analysis(folder=folder, index=index)
    explore_accepted(folder=folder)
    # get_paramters_rejected()
    
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
