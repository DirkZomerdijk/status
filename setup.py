#%%
import json
import pandas as pd
import numpy as np
from global_variables import *
from scipy.spatial import distance
from scipy.stats import ttest_ind
from functions import *
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool
from time import time
from copy import deepcopy
import pickle
import datetime
import sys 
from randomgen import SeedSequence
import os
import random
import copy

class Setup():
    
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()
        self.parameters = self.config["parameters"]
        self.df = self.load_data()
        self.jobs, self.nr_of_jobs = self.get_jobs()
        self.total_runs = self.get_total_runs()
        self.seeds = self.get_seeds()
        self.populations = self.get_populations()
        self.similarity_matrices = self.get_similarity_matrices()

        self.configure_jobs()

    def get_jobs(self):
        keys = list(self.parameters)
        jobs = [dict(zip(keys, values)) for values in itertools.product(*map(self.parameters.get, keys))]
        return jobs, len(jobs)
    
    def get_total_runs(self):
        return np.sum([job['repeats'] for job in self.jobs])
        
    def get_seeds(self):
        total_seeds = self.nr_of_jobs + self.total_runs
        return list(SeedSequence().generate_state(total_seeds))
        
    def load_config(self):
        with open(self.config_file) as config:
            return json.load(config)
    
    def load_data(self):
        return pd.read_csv('data/'+self.config['data_file']+'.csv')
    
    def get_populations(self):
        return [self.sample_population(self.jobs[i]['population_size']) for i in range(self.nr_of_jobs)]
    
    def sample_population(self, population_size):
        np.random.seed(self.seeds.pop())
        dutch_pop = population_size * self.config["dutch_w"] 
        suriname_pop = population_size * self.config["suriname_w"] 
        turkish_pop = population_size * self.config["turkish_w"]
        moroccan_pop = population_size * self.config["moroccan_w"]
        ghanaian_pop = population_size * self.config["ghanaian_w"]
        
        dutch = self.df.loc[self.df["H1_etniciteit"] == 1].sample(n=int(dutch_pop))
        surinam = self.df.loc[self.df["H1_etniciteit"] == 2].sample(n=int(suriname_pop))
        turkish = self.df.loc[self.df["H1_etniciteit"] == 3].sample(n=int(turkish_pop))
        maroccan = self.df.loc[self.df["H1_etniciteit"] == 4].sample(n=int(moroccan_pop))
        ghanaian = self.df.loc[self.df["H1_etniciteit"] == 5].sample(n=int(ghanaian_pop))
        
        cols = base_columns
        
        return self.normalize_columns(pd.concat([dutch, surinam, maroccan, turkish, ghanaian])[cols].rename(columns=rename_columns))
    
    def normalize_columns(self, df):
        for col in df.columns:
            if col in columns_normalized:
                df[col] = normalize(df[col], 0, 1)
        return df
    
    def get_similarity_matrices(self):
        return [self.calculate_distance(df) for df in self.populations]
    
    def calculate_distance(self, df):
        values = df[["occ","edu"]].values
        distance_matrix = 1 - distance.cdist(values, values, metric=self.config["distance_measure"])
        max_value = np.max(distance_matrix)
        min_value = np.min(distance_matrix)
        distance_matrix = np.array([normalize_minmax(row, 0, 1, min_value, max_value) for row in distance_matrix])

        return distance_matrix
    
    def get_simulation_configs(self):
        return [self.make_simulation_config for _ in range(len(self.populations))]
    
    def get_job_seeds(self, repeats):
        seeds = self.seeds[:repeats]
        self.seeds[:repeats]
        del self.seeds[:repeats]
        return seeds
    
    def configure_jobs(self, track=""):
        for i, job in enumerate(self.jobs):
            self.jobs[i]['population'] = self.populations[i]
            self.jobs[i]['noise_seeds'] = self.get_job_seeds(job['repeats'])
            self.jobs[i]['similarity'] = self.similarity_matrices[i]
            self.jobs[i]['config'] = self.config_file
            self.jobs[i]['save_folder'] = self.config["save_folder"] 
            # if track !="":
            # self.jobs[i]['job_nr'] = int(i)
            # print(self.jobs[i]['job_nr'])





# %%
