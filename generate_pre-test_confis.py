#%%
import numpy as np
import matplotlib.pyplot as plt
import json
import copy
# chronic_threshold
# repeats
# time
# stress_max

global_settings = {
    "data_file": "clean",
    "save_folder": "pre-test\\",
    "status_type": "linear",
    "distance_measure": "euclidean",
    "scenario": "normal",
    # "dutch_w": 0.664,
    # "turkish_w": 0.075,
    # "moroccan_w": 0.13,
    # "ghanaian_w": 0.021,
    # "suriname_w": 0.11,
    "dutch_w": 0.664,
    "turkish_w": 0.075,
    "moroccan_w": 0.13,
    "ghanaian_w": 0.021,
    "suriname_w": 0.11,
    "parameters": {}
}

     
param_range = {
    "similarity_min" : {"range": [0.1, 1], "type": "f"},
    "ses_noise" : {"range": [0, 4], "type": "i"},
    # "vul_param" : {"range": [0.1,1], "type": "f"},
    "psr_param" : {"range": [0.1,1], "type": "f"},
    "recover_param" : {"range": [0.001, 0.1], "type": "f"},
    "prestige_beta" : {"range": [0.005, 0.05], "type": "f"},
    "prestige_param" : {"range": [0.1,1], "type": "f"},
    "stressor_param" : {"range": [0.1,1], "type": "f"},
    "interactions": {"range": [1,3], "type": "i"}, 
    "coping_noise" : {"range": [0.01, 0.1], "type": "f"},
}

if __name__ == "__main__":
    
    samples = 500
    
    configs = []
    
    param_samples = {}
        
    for k in param_range.keys():
        typ = param_range[k]['type']
        if typ is "f":
            params = np.array(np.random.uniform(param_range[k]['range'][0], param_range[k]['range'][1], samples))
        else:
            params = np.array(np.random.randint(param_range[k]['range'][0], param_range[k]['range'][1], samples))
        param_samples[k] = params

    for i in range(samples):
        print(i)
        config_settings = copy.deepcopy(global_settings)
        for k in param_range.keys():
            config_settings['parameters'][k] = [param_samples[k][i].item()]
        config_settings['parameters']['repeats'] = [10]
        config_settings['parameters']['population_size'] = [502]
        config_settings['parameters']['chronic_threshold'] = [0.0001]
        config_settings['parameters']['stress_max'] = [10000]
        config_settings['parameters']['time'] = [500]
        config_settings['parameters']['job_nr'] = [i]
        config_settings['parameters']['vul_param'] = [0]
        
        configs.append(config_settings)

    for i, config in enumerate(configs):
        filenr = "{0:03}".format(i)
        # print(filenr)
        with open('configs/pre-test/sample_'+str(filenr)+'.json', 'w') as outfile:
            json.dump(config, outfile)
# np.random.uniform(similarity_base[0], similarity_base[1])
# np.random.uniform(ses_noise[0], ses_noise[1])
# np.random.uniform(vul_param[0], vul_param[1])
# np.random.uniform(psr_param[0], psr_param[1])
# np.random.uniform(prestige_beta[0], prestige_beta[1])
# np.random.uniform(prestige_param[0], prestige_param[1])
# np.random.uniform(stressor_param[0], stressor_param[1])
# %%
