import numpy as np
from global_variables import *
from model import *

def normalize(arr, lower, upper):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized = (upper-lower)*(arr-arr_min)/(arr_max-arr_min) + lower
    return np.array(normalized)

def normalize_minmax(arr, lower, upper, min_value, max_value):
    normalized = (upper-lower)*(arr-min_value)/(max_value-min_value) + lower
    return np.array(normalized)

def select_actors(events, step):
    actors = np.nonzero(events[step])[0]
    interaction_per_agent = events[step, actors]
    actors, interaction_per_agent = unison_shuffled_copies(actors, interaction_per_agent)
    return np.array([actors, interaction_per_agent]).T






def run_models(Dict):
    return Model(**Dict)

def normalize(arr, lower, upper):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized = (upper-lower)*(arr-arr_min)/(arr_max-arr_min) + lower
    return np.array(normalized)

def normalize_minmax(arr, lower, upper, min_value, max_value):
    normalized = (upper-lower)*(arr-min_value)/(max_value-min_value) + lower
    return np.array(normalized)


def transform_probability(array):
    total = np.sum(array)    
    if total == 0:
        return []
    return array / total

# @profile
def find_nearest_above(my_array, target, no_agents):
    diff = my_array - target
    mask = np.ma.less_equal(diff, 0)
    # We need to mask the negative differences and zero
    # since we are looking for values above
    if np.all(mask):
        return no_agents-2 # returns None if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

