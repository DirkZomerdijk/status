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
from global_variables import *
from setup import *
from model_functions import *
from functions import *

def normalize(arr, lower, upper):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized = (upper-lower)*(arr-arr_min)/(arr_max-arr_min) + lower
    return np.array(normalized)

if __name__ == "__main__":
    root = os.path.dirname(os.path.realpath(__file__)) + "\\configs\\pre-test\\"
    configs = [root + "sample_" + "{0:03}".format(i) + ".json" for i in np.arange(0, 1)]
    # configs = [
        # 'ofat_recovery', 
    # #     # 'ofat_similarity', 
    # #     # 'ofat_psr',
    # #     # 'ofat_interactions',
    # #     # 'ofat_time',
    # #     # 'ofat_vulnerability',
    # #     # 'ofat_noise',
    # #     # "ofat_prestigebeta",
        # "./configs/debug.json",
        # ]
    
    jobs = []
    for i, config in enumerate(configs):
        print("preparing job: ", i)
        setup = Setup(config_file=config)
        if i == 0: jobs = setup.jobs
        else: jobs += setup.jobs
    print("Total jobs: ", len(jobs))
    # run_models(jobs[0])
    with Pool() as p:
        p.map(run_models, jobs)
    print("FINISH")