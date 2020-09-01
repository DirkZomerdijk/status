import json
import pandas as pd
import numpy as np
from time import time
from copy import deepcopy
import datetime
import pickle

from global_variables import *
from functions import *

def Model(**args):
    
    params = args
    no_interactions = args['interactions']
    ses_noise = args['ses_noise']
    repeats = args['repeats']
    vul_param = args['vul_param']
    psr_param = args['psr_param']
    stressor_param = args['stressor_param']
    recover_param = args['recover_param']
    noise_seeds = args['noise_seeds']
    population = args['population']
    idx_population = np.arange(population.shape[0])
    similarity = args['similarity']
    similarity_base = deepcopy(similarity)
    time = args['time']
    no_agents = population.shape[0]
    stress_m = np.zeros((no_agents, time, repeats), dtype=np.float16)
    interactions_m = np.zeros((no_agents, time, repeats), dtype=np.int16)
    interaction_history_m = np.zeros((no_agents, no_agents), dtype=np.uint8)
    stress = np.zeros(no_agents, dtype=np.float16)
    interactions = np.zeros(no_agents, dtype=np.int16)
    interaction_history = np.zeros((no_agents, no_agents), dtype=np.uint8)
        
    runtimes = []
    for i in range(repeats):
        print(i)
        s = time()
        np.random.seed(args['noise_seeds'][i])
        events = np.random.poisson(no_interactions/DAY, (no_agents, time)).T
        step = 0
        while(step < time - 1):
            actors = select_actors(events, step)
            for actor in actors:
                actor_similarity = similarity[actor[0], :]
                for contact in range(actor[1]):
                    while(True):
                        total = np.sum(actor_similarity)    
                        if total == 0:
                            interaction_probability = []
                        else:  interaction_probability = actor_similarity / total
                        
                        if len(interaction_probability) == 0:
                            break
                        
                        sorted_probability = np.argsort(interaction_probability)[::-1]
                        interaction_probability = np.cumsum(interaction_probability[sorted_probability])
                        dice = np.random.sample()
                        
                        diff = my_array - target
                        mask = np.ma.less_equal(diff, 0)
                        # We need to mask the negative differences and zero
                        # since we are looking for values above
                        if np.all(mask):
                            nearest_neighbor = no_agents-2
                        else: # returns None if target is greater than any value
                            masked_diff = np.ma.masked_array(diff, mask)
                            nearest_neighbor = masked_diff.argmin()
                    
                        alter = sorted_probability[nearest_neighbor]
                        interactions[[actor[0], alter]] += 1
                        idx = np.sort([actor[0], alter])
            #             if(interaction_history[idx[0], idx[1]]!=1):
            #                 self.stress[actor[0]] += calc_stress(self.stress, self.population, actor[0], alter, self.ses_noise, self.vul_param, self.population.loc[self.population.index[actor[0]], "psr"], self.psr_param, self.stressor_param)    
            #                 self.interaction_history[idx[0], idx[1]] = 1
            #                 break
            #             else:
            #                 self.similarity[actor[0], alter] = 0
            # stress = recover(stress, idx_population, actors, recover_param)
            # if step%DAY==0: similarity = deepcopy(similarity_base); interaction_history = np.zeros((no_agents, no_agents), dtype=np.uint8)
            # self.collect()
            step += 1
        
        # run(time, events, stress, idx_population, recover_param, similarity, similarity_base, interaction_history, no_agents, stress_m, interactions_m)
    #     self.collect()
        
    #     self.runtimes.append((time() - s)/60)
    #     print(self.runtimes[self.i])
    # print("Average runtime: ", np.mean(self.runtimes))
    # self.results = {
    #     "stress": self.stress_m, 
    #     "interactions": self.interactions_m,
    #     "params": args,
    #     "df": self.population,
    # }
    # f = open("test_model.pkl", "wb")
    # pickle.dump(self.results, f)


        

    
def get_events(no_interactions, no_agents, time, day_duration):
    return np.random.poisson(no_interactions/day_duration, (no_agents, time)).T


def collect(self):
    self.stress_m[:, self.step, self.i] = self.stress
    self.interactions_m[:, self.step, self.i] = self.interactions
    
    
    
    # return pd.DataFrame({
    #     "stress": self.stress,
    #     "vulnerability": [calc_vulnerability(x, self.vul_param) for x in self.stress],
    #     "status": self.population['status'],
    #     "psr": self.population['psr'],
    #     "interactions": self.interactions
    # })

def run(self):
    step = 0
    while(step < time - 1):
        actors = select_actors(events, step)
        interact(actors)
        stress = recover(stress, idx_population, actors, recover_param)
        if step%DAY==0: similarity = deepcopy(similarity_base); interaction_history = np.zeros((no_agents, no_agents), dtype=np.uint8)
        self.collect()
        step += 1

        # break

def interact(self, actors):
    for actor in actors:
        actor_similarity = self.similarity[actor[0], :]
        for contact in range(actor[1]):
            while(True):
                interaction_probability = transform_probability(actor_similarity)
                
                if len(interaction_probability) == 0:
                    break
                
                sorted_probability = np.argsort(interaction_probability)[::-1]
                interaction_probability = np.cumsum(interaction_probability[sorted_probability])
                dice = np.random.sample()
                alter = sorted_probability[find_nearest_above(interaction_probability, dice, self.no_agents)]
                self.interactions[[actor[0], alter]] += 1
                idx = np.sort([actor[0], alter])
                if(self.interaction_history[idx[0], idx[1]]!=1):
                    self.stress[actor[0]] += calc_stress(self.stress, self.population, actor[0], alter, self.ses_noise, self.vul_param, self.population.loc[self.population.index[actor[0]], "psr"], self.psr_param, self.stressor_param)    
                    self.interaction_history[idx[0], idx[1]] = 1
                    break
                else:
                    self.similarity[actor[0], alter] = 0