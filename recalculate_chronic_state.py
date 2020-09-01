#%%
import os
import numpy as np
import pickle
from plotter import *

if __name__ == "__main__":
    root = os.path.dirname(os.path.realpath(__file__))
    folder = "interesting"    
    files = np.array(glob.glob(root+ "\\results\\"+folder+"\\*.pkl"))
    WEEK = 30
    for file in files:
        print(file)
        with open(file, 'rb') as f:
            d = Analyzer(pickle.load(f))
        for i in range(20):        
            for step in range(500):
                if step >= WEEK:
                    chronic_state = np.min(d.stress_m[:, step-WEEK:step, i], axis=1)
                    chronic_intensity = np.mean(d.stress_m[:, step-WEEK:step, i], axis=1)
                    chronic_state[chronic_state > d.chronic_threshold] = 1
                    d.chronic_m[:, step, i] = chronic_state
        d.chronic_mean = np.mean(d.chronic_m, axis = (1, 2))
        d.chronic_ts = np.mean(d.chronic_m, axis=2)
        d.chronic_ts_mean = np.mean(d.chronic_m, axis=(0,2))    
        
        d.stress_ts_class()
        d.chronic_ts_class()     
        d.chronic_i_ts_class()     
        d.scatter(d.df['status'], d.stress_mean, xlabel="status", ylabel="stress")
        d.scatter(d.df['status'] + d.df['prestige'], d.stress_mean, xlabel="status + prestige", ylabel="stress")
        d.scatter(d.df['status'], d.df['prestige'], xlabel="status", ylabel="prestige")
        d.scatter(d.df['status'], d.chronic_mean, xlabel="status", ylabel="chronic")
        
                