#%%

# dutch_w = 0.664
# turkish_w = 0.075
# moroccan_w = 0.13
# ghanaian_w = 0.021
# suriname_w = 0.11

# sample_n =  4000

# dutch_pop = sample_n * dutch_w 
# suriname_pop = sample_n * suriname_w
# turkish_pop = sample_n * turkish_w
# moroccan_pop = sample_n * moroccan_w
# ghanaian_pop = sample_n * ghanaian_w

# ethnicity_weights = [
#     dutch_w,
#     suriname_w,
#     moroccan_w,
#     turkish_w,
#     ghanaian_w,
# ]

# pops = [
#     dutch_pop,
#     suriname_pop,
#     turkish_pop,
#     moroccan_pop,
#     ghanaian_pop
# ]
DEBUG = True
SAVETYPE = "group"


root = "C:/Users/Admin/Code/status/"
results_dir = "C:/Users/Admin/Code/status/results/"

param_dict = {
    # "population_size": [],
    # "chronic_threshold": [],
    "similarity_min": [],
    "interactions": [],
    "ses_noise": [],
    # "repeats": [],
    "vul_param": [],
    "psr_param": [],
    "coping_noise": [],
    "recover_param": [],
    "prestige_beta": [],
    "prestige_param": [],
    "stressor_param": [],
    }   

DAY = 1
WEEK = 7*DAY

CHRONIC_STRESS_PERIOD = 10

MAX_STATUS_DIFFERENCE = 14


status_dict_linear = {
    "occ": {
        1: 0,
        2: 1,
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 5,
        8: 6
    },
    "edu": {
        1: 0,
        2: 2,
        3: 4,
        4: 6
    },
    "inc": {
        1: -1,
        2: -0.5,
        3: 0.5,
        4: 1
    }
}

status_dict = {
    "occ": {
        1: 0,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 3,
        7: 4,
        8: 5
    },
    "edu": {
        1: 0,
        2: 1,
        3: 2,
        4: 4
    },
    "inc": {
        1: 0,
        2: 0,
        3: 1,
        4: 1
    }
}

base_columns = ["ID", "occupation","H1_InkHhMoeite","H1_Opleid","H1_Mastery_Sumscore","H1_SSQSa", "status_l", "psr", "H1_etniciteit", "prestige"]

rename_columns = dict({
    "ID": "id",
    "occupation":"occ",
    "H1_InkHhMoeite":"inc",
    "H1_Opleid":"edu",
    "H1_Mastery_Sumscore":"mastery",
    "H1_SSQSa":"support",
    "status_l" :"status",
    "H1_etniciteit": "eth",
    "H1_LO_BMI": "bmi",
    "H1_Roken": "smoke"
    })

columns_normalized = ["occ","edu","mastery","support"]
# %%
