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
        self.chronic_threshold = 0.0
        self.df = results['df']
        self.no_interactions = results['params']['interactions']
        self.ses_noise = results['params']['ses_noise']
        self.repeats = results['params']['repeats']
        self.vul_param = results['params']['vul_param']
        self.psr_param = results['params']['psr_param']
        self.recover_param = results['params']['recover_param']
        self.noise_seeds = results['params']['noise_seeds']
        self.population = results['params']['population']
        self.no_agents = self.population.shape[0]
        self.idx_population = np.arange(self.population.shape[0])
        self.similarity = results['params']['similarity']
        self.time = results['params']['time']
        self.stress_m = np.array(results['stress'], dtype=np.float32)
        self.interactions_m = results['interactions']
        self.interactions_mean = np.mean(self.interactions_m, axis=(1, 2))
        self.stress_ts = np.mean(self.stress_m, axis=2)
        self.stress_ts_mean = np.mean(self.stress_m, axis=(0,2))
        self.interactions_ts = np.mean(self.interactions_m, axis=2)
        self.stress_end = np.mean(self.interactions_m[:, -1, :], axis=1)
        self.chronic_intensities = results['chronic_intensity']
        self.chronic_m = results['chronic_state']
        self.chronic_ts = np.mean(self.chronic_m, axis=2)
        self.chronic_ts_mean = np.mean(self.chronic_ts, axis=0)
        self.df['chronic_state'] = np.mean(self.chronic_states, axis=(1,2))
        self.df['chronic_intensity'] = np.mean(self.chronic_intensities, axis=(1,2))  
        self.stress_mean = np.mean(self.stress_m, axis=(1, 2))
        self.df['stress'] = self.stress_mean
        self.stress_std = np.std(np.array(self.stress_ts, dtype=np.float64), axis=1)
        self.df['stress_std'] = self.stress_std
        self.vulnerability = [calc_vulnerability(x, self.params['vul_param']) for x in self.stress_mean]
        self.df['vulnerability'] = self.vulnerability
        self.stress_mean_final_week = np.mean(self.stress_m[:, -70:, :], axis=(1, 2))
        self.stress_std_final_week = np.std(self.stress_m[:, -70:, :], axis=(1, 2))
        self.status_stress_coeff = self.get_status_stress_coeff()    
        
        # self.print_init()

    def get_status_stress_coeff(self):
        z = np.polyfit(self.df['status'], np.array(self.stress_mean_final_week, dtype=np.float32), 1)
        # print(z)
   
    
    def print_init(self):
        print("chronic_threshold\t", self.params['chronic_threshold'])
        print("similarity_base\t\t", self.params['similarity_base'])
        print("interactions\t\t", self.params['interactions'])
        print("ses_noise\t\t", self.params['ses_noise'])
        print("repeats\t\t\t", self.params['repeats'])
        print("stressor_param\t\t", self.params['stressor_param'])
        print("vul_param\t\t", self.params['vul_param'])
        print("psr_param\t\t", self.params['psr_param'])
        print("recover_param\t\t", self.params['recover_param'])
        print("prestige param\t\t", self.params['prestige_param'])
        print("prestige beta\t\t", self.params['prestige_beta'])
        print("stress_max\t\t", self.params['stress_max'])
        print("noise_seeds\t\t", len(self.params['noise_seeds']))
        print("time\t\t\t", self.params['time'])
        print("population_shape\t", self.population.shape)
        
    
    def scatter(self, x, y, xlabel = '', ylabel = '', label='', scale="linear"):
        
        if label != '':
            plt.scatter(x, y, alpha=0.2, label=label)
        else:
            plt.scatter(x, y, alpha=0.2)
            
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale(scale)
        
        
        # plt.xlabel(x)
        
        plt.plot(x,p(x),"--")

    def scatter_nonzero(self, x, y):
        df = self.df.loc[self.df['stress']!=0]
        x = df[x]
        y = df[y]
        plt.scatter(x, y, alpha=0.1)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x,p(x),"r--")
    
    def scatter_mean(self, group, y):
        x = self.df[group].unique()
        y = self.df.groupby(group).mean()[y]
        plt.scatter(x = x, y = y)
        
    def max(self, x):
        return np.max(self.df[x])
    
    def hist(self, x):
        plt.hist(x, bins=100)

    def box_status(self, x):
        self.df.boxplot(column = x, by = 'status', figsize=(12, 6))


    
    def hist_count(self, group, xlabel):
        y = self.df.groupby(group).count()
        print(y.index.get_level_values(0).values)
        print(y[xlabel].values)
        # xlabels = y.groups.keys()[0]
        # print(xlabels)
        # xlabels = y[group]
        # print(xlabels)

        plt.bar(y.index.get_level_values(0).values, y[xlabel])
        # plt.xticks(y[xlabel].values)
                
    def bar_psr_stress(self):
            low_mean = self.df.loc[self.df['psr'] <= 0.33, 'stress'].mean()
            med_mean = self.df.loc[(self.df['psr'] > 0.33) & (self.df['psr'] <= 0.67), 'stress'].mean()
            high_mean = self.df.loc[self.df['psr'] > 0.67, "stress"].mean()
            
            low_std = self.df.loc[self.df['psr'] <= 0.33, 'stress'].std()
            med_std = self.df.loc[(self.df['psr'] > 0.33) & (self.df['psr'] <= 0.67), 'stress'].std()
            high_std = self.df.loc[self.df['psr'] > 0.67, "stress"].std()
            
            print(low_mean, med_mean, high_mean)
            print(low_std, med_std, high_std)
            # print(ttest_ind(self.df.loc[self.df['psr'] < 0.5, 'stress'], self.df.loc[self.df['psr'] > 0.5, 'stress'], equal_var = False))
            
            
            conditions = ['low', 'med', 'high']
            x_pos = np.arange(len(conditions))
            mean = [low_mean, med_mean, high_mean]
            error = [low_std, med_std, high_std]
            
            fig, ax = plt.subplots()
            ax.bar(x_pos, mean, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel('Stress')
            ax.set_xlabel('psr')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(conditions)
            # ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
            ax.yaxis.grid(True)

            # Save the figure and show
            plt.tight_layout()
            # plt.savefig('bar_plot_with_error_bars.png')
            plt.show() 
            
        
    def bar_status_stress(self):
            low_mean = self.df.loc[self.df['status'] <= 4.67, 'stress'].mean()
            med_mean = self.df.loc[(self.df['status'] > 4.67) & (self.df['status'] <= 9.33), 'stress'].mean()
            high_mean = self.df.loc[self.df['status'] > 9.33, "stress"].mean()
            
            low_std = self.df.loc[self.df['status'] <= 4.67, 'stress'].std()
            med_std = self.df.loc[(self.df['status'] > 4.67) & (self.df['status'] <= 9.33), 'stress'].std()
            high_std = self.df.loc[self.df['status'] > 9.33, "stress"].std()
            
            print(low_mean, med_mean, high_mean)
            print(low_std, med_std, high_std)
            # print(ttest_ind(self.df.loc[self.df['psr'] < 0.5, 'stress'], self.df.loc[self.df['psr'] > 0.5, 'stress'], equal_var = False))
            
            
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
            # ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
            ax.yaxis.grid(True)

            # Save the figure and show
            plt.tight_layout()
            # plt.savefig('bar_plot_with_error_bars.png')
            plt.show()
        
    def heatmap_similarity(self):
        idx = np.argsort(self.df['status']).values
        # print(self.similarity[0, 3] == self.similarity[3,0])
        similarity = deepcopy(self.similarity)
        similarity = similarity[idx, :]
        similarity = np.array([row[idx] for row in similarity])
        plt.imshow(similarity, cmap='hot', interpolation='nearest')
        plt.colorbar()

    def compare_zero_nonzero(self):
        # print(self.df.head())
        conditions = ['stress < 0.5', 'stress > 0.5']
        x_pos = np.arange(len(conditions))
        
        mean_zero = self.df.loc[self.df['stress'] <= 0.1, 'psr'].mean()
        std_zero = self.df.loc[self.df['stress'] <= 0.1, 'psr'].std()
        mean_nonzero = self.df.loc[self.df['stress'] > 0.1, 'psr'].mean()
        std_nonzero = self.df.loc[self.df['stress'] > 0.1, 'psr'].std()
        
        mean = [mean_zero, mean_nonzero]
        error = [std_zero, std_nonzero]

        fig, ax = plt.subplots()
        ax.bar(x_pos, mean, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Psr')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(conditions)
        # ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        # plt.savefig('bar_plot_with_error_bars.png')
        plt.show()
    
    def mean_stress_ts_class(self):
        idx = np.argsort(self.df['status'])
        plt.plot(np.arange(self.time), np.mean(self.stress_ts[idx][0:int(self.no_agents/3)], axis=0), label="low")
        plt.plot(np.arange(self.time), np.mean(self.stress_ts[idx][int(self.no_agents/3):int(self.no_agents/1.5)], axis=0), label="med")
        plt.plot(np.arange(self.time), np.mean(self.stress_ts[idx][int(self.no_agents/1.5):], axis=0), label="high")
        
        
        # plt.axhlines(low_m)
        # plt.axhlines(med_m)
        # plt.axhlines(high_m)
        
        plt.legend()

    def chronic_ts_class(self):
        idx = np.argsort(self.df['status'])
        plt.plot(np.arange(self.time), np.mean(self.chronic_intensities[idx][0:int(self.no_agents/3)], axis=0), label="low")
        plt.plot(np.arange(self.time), np.mean(self.chronic_intensities[idx][int(self.no_agents/3):int(self.no_agents/1.5)], axis=0), label="med")
        plt.plot(np.arange(self.time), np.mean(self.chronic_intensities[idx][int(self.no_agents/1.5):], axis=0), label="high")
        
        
        # plt.axhlines(low_m)
        # plt.axhlines(med_m)
        # plt.axhlines(high_m)
        
        plt.legend()
        
    def chronic_states_ts_class(self):
        idx = np.argsort(self.df['status'])
        plt.plot(np.arange(self.time), np.mean(self.chronic_states[idx][0:int(self.no_agents/3)], axis=0), label="low")
        plt.plot(np.arange(self.time), np.mean(self.chronic_states[idx][int(self.no_agents/3):int(self.no_agents/1.5)], axis=0), label="med")
        plt.plot(np.arange(self.time), np.mean(self.chronic_states[idx][int(self.no_agents/1.5):], axis=0), label="high")
        
        
        # plt.axhlines(low_m)
        # plt.axhlines(med_m)
        # plt.axhlines(high_m)
        
        plt.legend()
        
    def gbar_psr_status(self):
        low_psr = self.df.loc[self.df['psr'] <= 0.33]
        low_low_mean = low_psr.loc[low_psr['status'] <= 4.67, 'stress'].mean()
        low_med_mean = low_psr.loc[(low_psr['status'] > 4.67) & (low_psr['status'] <= 9.33), 'stress'].mean()
        low_high_mean = low_psr.loc[low_psr['status'] > 9.33, "stress"].mean()
            
        print(low_low_mean, low_med_mean, low_high_mean)
        
        med_psr = self.df.loc[(self.df['psr'] > 0.33) & (self.df['psr'] <= 0.67)]        
        med_low_mean = med_psr.loc[med_psr['status'] <= 4.67, 'stress'].mean()
        med_med_mean = med_psr.loc[(med_psr['status'] > 4.67) & (med_psr['status'] <= 9.33), 'stress'].mean()
        med_high_mean = med_psr.loc[med_psr['status'] > 9.33, "stress"].mean()
        
        print(med_low_mean, med_med_mean, med_high_mean)
        
        high_psr = self.df.loc[self.df['psr'] > 0.67]
        high_low_mean = high_psr.loc[high_psr['status'] <= 4.67, 'stress'].mean()
        high_med_mean = high_psr.loc[(high_psr['status'] > 4.67) & (high_psr['status'] <= 9.33), 'stress'].mean()
        high_high_mean = high_psr.loc[high_psr['status'] > 9.33, "stress"].mean()
        
        print(high_low_mean, high_med_mean, high_high_mean)
        
        
        low_psr = self.df.loc[self.df['psr'] <= 0.33]
        low_low_std = low_psr.loc[low_psr['status'] <= 4.67, 'stress'].std()
        low_med_std = low_psr.loc[(low_psr['status'] > 4.67) & (low_psr['status'] <= 9.33), 'stress'].std()
        low_high_std = low_psr.loc[low_psr['status'] > 9.33, "stress"].std()
            
        print(low_low_std, low_med_std, low_high_std)
        
        med_psr = self.df.loc[(self.df['psr'] > 0.33) & (self.df['psr'] <= 0.67)]        
        med_low_std = med_psr.loc[med_psr['status'] <= 4.67, 'stress'].std()
        med_med_std = med_psr.loc[(med_psr['status'] > 4.67) & (med_psr['status'] <= 9.33), 'stress'].std()
        med_high_std = med_psr.loc[med_psr['status'] > 9.33, "stress"].std()
        
        print(med_low_std, med_med_std, med_high_std)
        
        high_psr = self.df.loc[self.df['psr'] > 0.67]
        high_low_std = high_psr.loc[high_psr['status'] <= 4.67, 'stress'].std()
        high_med_std = high_psr.loc[(high_psr['status'] > 4.67) & (high_psr['status'] <= 9.33), 'stress'].std()
        high_high_std = high_psr.loc[high_psr['status'] > 9.33, "stress"].std()
        
        print(high_low_std, high_med_std, high_high_std)
        
        df = pd.DataFrame({
            'low status': [low_low_mean,med_low_mean, high_low_mean],
            'med status': [low_med_mean,med_med_mean, high_med_mean],  
            'high status': [low_high_mean, med_high_mean, high_high_mean]  
        }, index=["low_psr", "mid_psr", "high_psr"]).plot.bar(
            rot=0, 
            # yerr=[[low_low_std,med_low_std, high_low_std],[low_med_std,med_med_std, high_med_std], [low_high_std, med_high_std, high_high_std]]
            )

        # print(stats.ttest_ind(low_psr.loc[low_psr['status'] <= 4.67, 'stress'],low_psr.loc[(low_psr['status'] > 4.67) & (low_psr['status'] <= 9.33), 'stress'], equal_var=False))
    

    def gbar_status_psr(self):
       
        low_status = self.df.loc[self.df['status'] <= 4.67]
        med_status = self.df.loc[(self.df['status'] > 4.67) & (self.df['status'] <= 9.33)]
        hig_status = self.df.loc[self.df['status'] > 9.33]
        
        low_status_low_psr = low_status.loc[low_status['psr'] <= 0.33, 'stress'].mean()
        low_status_med_psr = low_status.loc[(low_status['psr'] > 0.33) & (low_status['psr'] <= 0.67), 'stress'].mean()
        low_status_hig_psr = low_status.loc[low_status['psr'] > 0.67, 'stress'].mean()

        med_status_low_psr = med_status.loc[med_status['psr'] <= 0.33, 'stress'].mean()
        med_status_med_psr = med_status.loc[(med_status['psr'] > 0.33) & (med_status['psr'] <= 0.67), 'stress'].mean()
        med_status_hig_psr = med_status.loc[med_status['psr'] > 0.67, 'stress'].mean()
                
        hig_status_low_psr = hig_status.loc[hig_status['psr'] <= 0.33, 'stress'].mean()
        hig_status_med_psr = hig_status.loc[(hig_status['psr'] > 0.33) & (hig_status['psr'] <= 0.67), 'stress'].mean()
        hig_status_hig_psr = hig_status.loc[hig_status['psr'] > 0.67, 'stress'].mean()
        
        df = pd.DataFrame({
            'low psr': [low_status_low_psr,med_status_low_psr, hig_status_low_psr],
            'med psr': [low_status_med_psr,med_status_med_psr, hig_status_med_psr],  
            'high psr': [low_status_hig_psr, med_status_hig_psr, hig_status_hig_psr]  
        }, index=["low_status", "mid_status", "high_status"]).plot.bar(rot=0)

        plt.ylabel('stress')


    def gbar_status_psr_prestige(self):
       
        low_status = self.df.loc[self.df['status'] <= 4.67]
        med_status = self.df.loc[(self.df['status'] > 4.67) & (self.df['status'] <= 9.33)]
        hig_status = self.df.loc[self.df['status'] > 9.33]
        
        low_status_low_psr = low_status.loc[low_status['psr'] <= 0.33, 'prestige'].mean()
        low_status_med_psr = low_status.loc[(low_status['psr'] > 0.33) & (low_status['psr'] <= 0.67), 'prestige'].mean()
        low_status_hig_psr = low_status.loc[low_status['psr'] > 0.67, 'prestige'].mean()

        med_status_low_psr = med_status.loc[med_status['psr'] <= 0.33, 'prestige'].mean()
        med_status_med_psr = med_status.loc[(med_status['psr'] > 0.33) & (med_status['psr'] <= 0.67), 'prestige'].mean()
        med_status_hig_psr = med_status.loc[med_status['psr'] > 0.67, 'prestige'].mean()
                
        hig_status_low_psr = hig_status.loc[hig_status['psr'] <= 0.33, 'prestige'].mean()
        hig_status_med_psr = hig_status.loc[(hig_status['psr'] > 0.33) & (hig_status['psr'] <= 0.67), 'prestige'].mean()
        hig_status_hig_psr = hig_status.loc[hig_status['psr'] > 0.67, 'prestige'].mean()
        
        df = pd.DataFrame({
            'low psr': [low_status_low_psr,med_status_low_psr, hig_status_low_psr],
            'med psr': [low_status_med_psr,med_status_med_psr, hig_status_med_psr],  
            'high psr': [low_status_hig_psr, med_status_hig_psr, hig_status_hig_psr]  
        }, index=["low_status", "mid_status", "high_status"]).plot.bar(rot=0)
        plt.ylabel('prestige')
        
        
    def gbar_status_psr_chronic(self):
       
        low_status = self.df.loc[self.df['status'] <= 4.67]
        med_status = self.df.loc[(self.df['status'] > 4.67) & (self.df['status'] <= 9.33)]
        hig_status = self.df.loc[self.df['status'] > 9.33]
        
        low_status_low_psr = low_status.loc[low_status['psr'] <= 0.33, 'chronic_state'].mean()
        low_status_med_psr = low_status.loc[(low_status['psr'] > 0.33) & (low_status['psr'] <= 0.67), 'chronic_state'].mean()
        low_status_hig_psr = low_status.loc[low_status['psr'] > 0.67, 'chronic_state'].mean()

        med_status_low_psr = med_status.loc[med_status['psr'] <= 0.33, 'chronic_state'].mean()
        med_status_med_psr = med_status.loc[(med_status['psr'] > 0.33) & (med_status['psr'] <= 0.67), 'chronic_state'].mean()
        med_status_hig_psr = med_status.loc[med_status['psr'] > 0.67, 'chronic_state'].mean()
                
        hig_status_low_psr = hig_status.loc[hig_status['psr'] <= 0.33, 'chronic_state'].mean()
        hig_status_med_psr = hig_status.loc[(hig_status['psr'] > 0.33) & (hig_status['psr'] <= 0.67), 'chronic_state'].mean()
        hig_status_hig_psr = hig_status.loc[hig_status['psr'] > 0.67, 'chronic_state'].mean()
        
        df = pd.DataFrame({
            'low psr': [low_status_low_psr,med_status_low_psr, hig_status_low_psr],
            'med psr': [low_status_med_psr,med_status_med_psr, hig_status_med_psr],  
            'high psr': [low_status_hig_psr, med_status_hig_psr, hig_status_hig_psr]  
        }, index=["low_status", "mid_status", "high_status"]).plot.bar(rot=0)
        plt.ylabel('chronic state')
        # print(stats.ttest_ind(low_psr.loc[low_psr['status'] <= 4.67, 'chronic_state'],low_psr.loc[(low_psr['status'] > 4.67) & (low_psr['status'] <= 9.33), 'stress'], equal_var=False))
    
        
    def smoking_status(self):
        pass
    
    def bmi_status(self):
        pass
    
    def mesh(self):
        x = self.df['status'] + self.df['prestige'] 
        y = self.df['chronic_state']
        z = self.df['chronic_intensity']
        
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.view_init(azim=45, elev=90)
        surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf)
        # plt.savefig('teste.pdf')
        plt.xlabel('status+prestige')
        plt.ylabel('chronic state')
        
        plt.show()
        
def open_data_objects(experiment, folder, index):
    data = []
    # print(root+ "results/*"+folder+"*/*"+experiment+"*.pkl")
    files = np.array(glob.glob(root+ "results/*"+folder+"*/*"+experiment+"*.pkl"))
    for file in files[index]:
        with open(file, 'rb') as f:
            results = pickle.load(f)
        data.append(Analyzer(results))
    return data

def open_last_data_object(experiment, folder):
    data = []
    # print()
    file = glob.glob(root+ "results/*"+folder+"*/*"+experiment+"*.pkl")
    # for file in glob.glob(root+ "results/*"+folder+"*/*"+experiment+"*.pkl"):
    print(file[-1])
    with open(file[-1], 'rb') as f:
        results = pickle.load(f)
        data.append(Analyzer(results))
    return data

def analyze_ofat(data, param):
    
    arr = []
    for dat in data:
        print(dat.params[param])
        arr.append(dat.params[param])

    idx = np.argsort(arr)
    data = [data[i] for i in idx]

    
    coeffs = []
    plt.figure()
    for i, d in enumerate(data):
        d.scatter(d.df['status'], np.array(d.df['chronic_state'], dtype=np.float32))
        z = np.polyfit(d.df['status'], np.array(d.df['chronic_state'], dtype=np.float32), 1)
        print([d.params[param], z[0]])
        coeffs.append([d.params[param], z[0]])
    
    plt.figure()
    coeffs = np.sort(coeffs, axis=0)
    plt.scatter(coeffs[:, 0], coeffs[:, 1], label=coeffs[:, 0])
    plt.legend()

    plt.figure()        
    for i, d in enumerate(data):
        d.hist(np.array(d.df['chronic_state'], dtype=np.float32))
        plt.xlabel('frquency')
        plt.ylabel('chronic')
        plt.title(param +": "+str(d.params[param]))
        
        # plt.ylim([0, 250])
    
    plt.figure()        
    for i, d in enumerate(data):
        d.hist(np.array(d.interactions_mean, dtype=np.float32))
        print(np.mean(d.interactions_mean))
        plt.title(param +": "+str(d.params[param]))
        # plt.ylim([0, 80])

# def analyze_nominal():
#     pass
    
if __name__ == "__main__":
    # data = open_last_data_object("*", "single")[0]
    # print(data)
    # print("Max stress:\t\t", np.max(a.stress_m))
    # print("Mean stress:\t\t", np.mean(a.stress_m))
    # print("Median stress:\t\t", np.median(a.stress_m))
    # print("Mean chronic: \t\t", a.df['chronic_state'].mean()) z
    # print("Mean chronic int: \t", a.df['chronic_intensity'].mean())
    # print("N chronic: \t\t", a.df[a.df['chronic_state'] != 0].count()['chronic_state'])
    

    # interest = [
    #     216, 377, 581,582,583,584, 946
    # ]
    # interest = np.arange(0, 48)
    # print(interest[-1])
    # p = {
    #      'similarity_base':[], 
    #      'ses_noise':[], 
    #      'stressor_param':[], 
    #      'vul_param':[], 
    #      'psr_param':[], 
    #      'recover_param':[], 
    #      'prestige_param':[], 
    #      'prestige_beta':[], 
    #      }
    index = np.arange(0,6)
    data = open_data_objects("*", "single", index)

    # for d in data:
    #     for k in p.keys():
    #         p[k].append(d.params[k])
    # for k in p.keys():
    #     a = plt.hist(p[k])
    #     print(a)
    #     plt.title(k)
    #     plt.show()
#%%
    data = np.array(data)  
    for i, a in enumerate(data):
        a.print_init()
        # a = data[i]
        print(i)

        
        plt.figure()
        a.mean_stress_ts_class()
        plt.xlabel('time')
        plt.ylabel('Stress')
        plt.show()

        plt.figure()
        a.chronic_states_ts_class()
        plt.xlabel('time')
        plt.ylabel('Chronic state')
        plt.show()
        
        plt.figure()
        a.scatter((a.df['status']),  a.df['stress'],  xlabel="status", ylabel="stress")

    # a.mesh()
#%%    
    a = data
    plt.figure()
    a.scatter(a.df['status'], a.interactions_mean, xlabel="status", ylabel="interactions")
    
    plt.figure()
    plt.hlines(0, 0, 14)
    a.scatter(a.df['status'], a.df['prestige'], xlabel="status", ylabel="prestige")

    plt.figure()
    a.scatter(a.df['prestige'],  a.df['psr'],  xlabel="status", ylabel="psr")
        
    plt.figure()
    a.scatter(a.df['prestige'],  a.df['stress'],  xlabel="prestige", ylabel="stress")
    
    plt.figure()
    a.scatter((a.df['prestige'] + a.df['status']),  a.df['stress'],  xlabel="prestige + status", ylabel="stress")

    plt.figure()
    a.scatter((a.df['prestige'] + a.df['status']),  a.df['vulnerability'],  xlabel="prestige + status", ylabel="vulnerability")

    plt.figure()
    a.scatter(a.df['status'] + a.df['prestige'], np.array(a.df['chronic_state'] , dtype=np.float32), xlabel='status + prestige', ylabel="chronic_state")

    plt.figure()
    plt.hist(a.df['chronic_state'])
    # plt.figure()
    # a.scatter(a.df['status'] + a.df['prestige'], np.array(a.df['chronic_intensity'] , dtype=np.float32), xlabel='status + prestige', ylabel="chronic_intensity")

    plt.figure()
    a.scatter(a.df['prestige'], np.array(a.df['chronic_intensity'] , dtype=np.float32), xlabel='status + prestige', ylabel="chronic_intensity")
        
    plt.figure()
    a.df['status'].hist(bins=50)
    plt.xlabel('status')
    plt.ylabel('frequency')
    plt.xlim([0, 16])
    plt.ylim([0, 75])

    plt.figure()
    plt.hist(a.df['status']+a.df['prestige'], bins=50)
    plt.xlabel('status + prestige')
    plt.ylabel('frequency')
    plt.xlim([0, 16])
    plt.ylim([0, 75])
    
    # plt.ylim([0,100])
    # plt.ylim([-.5, 1.0])
    
    # plt.figure()
    # a.scatter(a.df['status'], np.array(a.chronic_stress, dtype=np.float32), xlabel='status', ylabel="chronic")
    # plt.ylim([-.5, 1.0])
    
    # plt.figure()
    # a.scatter(a.df['status'], np.array(a.chronic_stress, dtype=np.float32), xlabel='status + prestige', ylabel="chronic")

    # plt.figure()
    # a.scatter(a.df['status'], a.vulnerability, xlabel="status", ylabel="vulnerability")
    
    # plt.figure()
    # a.scatter(a.df['status'], a.df['status'] + a.df['prestige'], xlabel='status', ylabel="status + prestige")
    
    # plt.figure()
    # a.scatter(a.df['status'], a.df['psr'], xlabel='status', ylabel="psr")
    
    # plt.figure()
    # a.scatter(a.df['prestige']+a.df['status'], a.df['psr'], xlabel='status + prestige', ylabel="psr")
    
    # plt.figure()
    # a.scatter(a.df['psr'], np.array(a.stress_mean, dtype=np.float32), xlabel='psr', ylabel="stress")
    
    # # plt.ylim([-.5, 1.0])
    # plt.figure()
    # a.scatter(a.df['psr'], np.array(a.chronic_stress, dtype=np.float32), xlabel='psr', ylabel="chronic")
    
    # plt.figure()
    # a.scatter(a.df['psr'], np.array(a.vulnerability, dtype=np.float32), xlabel="psr", ylabel="vulnerability")
    
    # plt.figure(figsize=(14,6))
    # # plt.vlines([70,140,210,280,350,420,490,560,630,700], 0, .015)
    # plt.plot(np.arange(a.time), a.stress_ts_mean)

    plt.figure()
    a.gbar_psr_status()
    plt.figure()
    a.gbar_status_psr()

    # plt.figure()
    # a.gbar_psr_status_chronic()
        
    plt.figure()
    plt.plot(np.arange(a.time), a.stress_ts_mean)
    
    plt.figure()
    plt.plot(np.arange(a.time), np.mean(a.chronic_intensities, axis=0))
    
    plt.figure()
    a.heatmap_similarity()
    
    plt.figure()
    a.bar_psr_stress()
    
    plt.figure()
    a.gbar_status_psr()
    
    # plt.figure()
    # a.gbar_status_psr_prestige()
    
    # plt.figure()
    # a.gbar_status_psr_chronic()
    

    plt.figure()
    a.mean_stress_ts_class()
    plt.xlabel('time')
    plt.ylabel('Stress')
    plt.show()

    plt.figure()
    a.chronic_states_ts_class()
    plt.xlabel('time')
    plt.ylabel('Chronic state')
    plt.show()
    
    # plt.figure()
    # for dat in data:
    #     dat.chronic_ts_class()
    # plt.xlabel('time')
    # plt.ylabel('Chronic intensities')
    # plt.show()
    # print(a.stress_ts_mean[-1])
    # for i in range(a.no_agents):

    #     plt.plot(np.arange(a.time), a.stress_ts[i, :], alpha = 0.1, c = 'grey')
    # plt.figure()
    # plt.plot(np.arange(a.time), a.stress_ts[0, :], c = 'red', label = 'low status')
    # plt.plot(np.arange(a.time), a.stress_ts[5, :], c = 'red')
    # plt.plot(np.arange(a.time), a.stress_ts[10, :], c = 'red')
    # plt.legend()

    # plt.figure()
    # plt.plot(np.arange(a.time), a.stress_ts[20, :], c = 'blue', label = 'mid status')
    # plt.plot(np.arange(a.time), a.stress_ts[30, :], c = 'blue')
    # plt.plot(np.arange(a.time), a.stress_ts[50, :], c = 'blue')
    # plt.legend()

    # plt.figure()
    # plt.plot(np.arange(a.time), a.stress_ts[75, :], c = 'y', label = 'high status')
    # plt.plot(np.arange(a.time), a.stress_ts[80, :], c = 'y')
    # plt.plot(np.arange(a.time), a.stress_ts[94, :], c = 'y')
    # plt.legend()
#%%

    analyze_ofat(data, "psr_param")

#%%    
    


#     stress = []
#     std = []
#     for dat in data:
#         idx = np.argsort(dat.stress_std)
#         plt.errorbar(np.arange(0, 1000), dat.stress_mean[idx], yerr=dat.stress_std[idx])
#         plt.ylim([-1,3])
#         plt.show()
        
# #%%
#     data = []
#     for file in glob.glob(root+ "/results/ofat_similarity_*.pkl"):
#         with open(file, 'rb') as f:
#             results = pickle.load(f)
#         data.append(Analyzer(results))
# #%%
#     c_stress = []
#     c_stress_std = []
#     stress = []
#     std = []
#     conditions = []
#     for dat in data:
#         conditions.append(dat.params['similarity_base'])
#         c_stress.append(np.mean(dat.chronic_stress))
#         c_stress_std.append(np.std(dat.chronic_stress))
#         stress.append(np.mean(dat.stress_mean))
#         std.append(np.mean(dat.stress_std))
#         # idx = np.argsort(dat.stress_std)

#         dat.scatter(dat.df['status'], np.array(dat.stress_mean, dtype=np.float32), label=dat.params['similarity_base'] )
#     plt.legend()
        
#     plt.ylim([-0.05, 0.6])
#     plt.figure(figsize=(6,3))
#     plt.bar(np.arange(len(conditions)), c_stress, yerr=c_stress_std, align='center')
# #%%
#     data = []
#     for file in glob.glob(root+ "/results/ofat_psr_*.pkl"):
#         with open(file, 'rb') as f:
#             results = pickle.load(f)
#         data.append(Analyzer(results))
# #%%
#     c_stress = []
#     c_stress_std = []
#     stress = []
#     std = []
#     conditions = [0.1, 0.3, 0.5, 0.7, 0.9]
#     for i, dat in enumerate(data):
#         c_stress.append(np.mean(dat.chronic_stress))
#         c_stress_std.append(np.std(dat.chronic_stress))
#         stress.append(np.mean(dat.stress_mean))
#         std.append(np.mean(dat.stress_std))
#         # idx = np.argsort(dat.stress_std)
#         # plt.figure(figsize=(6,3))
#         dat.scatter(dat.df['status'], np.array(dat.stress_mean, dtype=np.float32), label = conditions[i])
#     plt.legend()
# #     plt.ylim([-0.05, 0.6])
# #     print(conditions)
# #     plt.figure(figsize=(6,3))
#     plt.figure(figsize=(6,3))
#     plt.xlabel(conditions)
#     plt.bar(np.arange(len(conditions)), c_stress, yerr=c_stress_std, align='center')
# #%%
#     data = []
#     for file in glob.glob(root+ "/results/ofat_recover*.pkl"):
#         with open(file, 'rb') as f:
#             results = pickle.load(f)
#         data.append(Analyzer(results))
# #%%
#     c_stress = []
#     c_stress_std = []
#     stress = []
#     std = []
#     conditions = [0.05, 0.1, 0.2, 0.3, 0.4]
#     for dat in data:
#         c_stress.append(np.mean(dat.chronic_stress))
#         c_stress_std.append(np.std(dat.chronic_stress))
#         stress.append(np.mean(dat.stress_mean))
#         std.append(np.mean(dat.stress_std))
#         # idx = np.argsort(dat.stress_std)
#         # plt.figure(figsize=(6,3))
#         dat.scatter(dat.df['status'], np.array(dat.stress_mean, dtype=np.float32) )
# #     plt.ylim([-0.05, 0.6])
# #     print(conditions)
# #     plt.figure(figsize=(6,3))
#     plt.figure(figsize=(6,3))
#     plt.xlabel(conditions)
#     plt.bar(np.arange(len(conditions)), c_stress, yerr=c_stress_std, align='center')
    
#     plt.figure(figsize=(6,3))
#     plt.xlabel(conditions)
#     plt.bar(np.arange(len(conditions)), stress, yerr=std, align='center')
#     # print(std)
    
    
    
    
    # print(np.array(stress).shape)
    # a = Analyzer(results)
        
    # plt.figure(figsize=(6,3))
    # a.scatter(a.df['status'], np.array(a.stress_mean, dtype=np.float32))
    
    # plt.figure(figsize=(6,3))
    # a.scatter(a.df['status'], a.chronic_stress)
    
    # plt.figure(figsize=(6,3))
    # a.scatter(a.df['status'], a.vulnerability)

    # plt.figure(figsize=(6,3))
    # a.scatter(a.df['psr'], np.array(a.stress_mean, dtype=np.float32))
    
    # plt.figure(figsize=(6,3))
    # a.scatter(a.df['psr'], a.chronic_stress)
    
    # plt.figure(figsize=(6,3))
    # a.scatter(a.df['psr'], a.vulnerability)    
    # # a.scatter(a.df['status'], a.chronic_stress)
    # a.scatter(a.df['status'], a.chronic_stress)
    # a.heatmap_similarity()
    
    # print(len(a.stress_end[a.stress_end==0]))
    # a.compare_zero_nonzero()
    # with open('test_small.pkl', 'rb') as f:
        # model = pickle.load(f)
    
    # idx = np.argsort(a.df['status'])
    # a.stress_ts = a.stress_ts[idx,:]
    # stress_time = np.mean(model.stress_m[:, :, :], axis=2)
    # # interactions = np.mean(model.stress_m[idx, :, :], axis=(1,2))
    
    # interactions = np.mean(model.interactions_m, axis=(1, 2))
    # model.population['interactions'] = interactions

    # model.population.sort_values(by=['status'])


    # # plt.scatter()
    # # (model.population['interactions'])
    
    # group = 'status'
    # xlabel = 'interactions'
    # plt.figure(figsize=(6,3))
    # print(a.df['status'].shape, a.stress_mean.shape)
    # plt.scatter(a.df['status'], a.stress_mean)
    
    # plt.figure(figsize=(6,3))
    # plt.scatter(a.df['psr'], a.stress_mean)
    
    # a.bar_status_stress()
    # y = model.population.groupby('status').mean()
    # # print(y)
    # # print(y.index.get_level_values(0).values)
    # # print(y[xlabel].values)
    # plt.bar(y.index.get_level_values(0).values, y[xlabel])
    # print(model.no_interactions)
    
    # plt.figure(figsize=(6,3))
    # for i in range(a.no_agents):
    #     plt.plot(np.arange(a.time), a.stress_ts[i, :], alpha = 0.1, c = 'grey')
    # plt.plot(np.arange(results['params']['time']), a.stress_ts[0, :], c = 'red', label = 'low status')
    # plt.plot(np.arange(results['params']['time']), a.stress_ts[5, :], c = 'red')
    # plt.plot(np.arange(results['params']['time']), a.stress_ts[10, :], c = 'red')
    # plt.plot(np.arange(results['params']['time']), a.stress_ts[20, :], c = 'blue', label = 'mid status')
    # plt.plot(np.arange(results['params']['time']), a.stress_ts[30, :], c = 'blue')
    # plt.plot(np.arange(results['params']['time']), a.stress_ts[50, :], c = 'blue')
    # plt.plot(np.arange(results['params']['time']), a.stress_ts[75, :], c = 'y', label = 'high status')
    # plt.plot(np.arange(results['params']['time']), a.stress_ts[80, :], c = 'y')
    # plt.plot(np.arange(results['params']['time']), a.stress_ts[94, :], c = 'y')
    # plt.legend()
    
    # print()
    # model.stress = np.mean(model.stress_m, axis = 1)
    # model.stress_std = np.std(model.stress_m, axis = 1)  
    # model.interactions = np.mean(model.interactions_m, axis = 1)
    # model.interactions_std = np.std(model.interactions_m, axis = 1)
    # model.df = pd.DataFrame({
    #         "stress": model.stress,
    #         "stress_std": model.stress_std, 
    #         "vulnerability": [calc_vulnerability(x, model.vul_param) for x in model.stress],
    #         "status": model.population['status'],
    #         "psr": model.population['psr'],
    #         "interactions": model.interactions,
    #         "interactions_std": model.interactions_std
    #     })
    # print(model.df)
    # print()
    # print(np.std(model.stress_m[:, 0]))
        
    # print(np.mean(np.mean(model.stress_m, 1)))
    # print(np.mean(np.std(model.stress_m, 1)))

# %%

    a = Analyzer(model.params, model.df, model.similarity_base)
    a.bar_psr_stress()
    a.bar_status_stress()
    fix = plt.figure(figsize=(6, 3))
    a.scatter('status', 'stress')
    
    fix = plt.figure(figsize=(6, 3))
    a.scatter('status', 'vulnerability')
    
   
    fix = plt.figure(figsize=(6, 3))
    a.scatter('interactions', 'stress')

    fix = plt.figure(figsize=(6, 3))
    a.scatter('interactions', 'vulnerability')
    
    fix = plt.figure(figsize=(6, 3))
    a.scatter('psr', 'stress')
    
    fix = plt.figure(figsize=(6, 3))
    a.scatter('status', 'interactions')
    
    fix = plt.figure(figsize=(6, 3))
    a.hist_count('status', 'interactions')
    
    fix = plt.figure(figsize=(6, 3))
    a.hist('status')
    
    a.heatmap_similarity()
