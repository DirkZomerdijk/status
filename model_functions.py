import numpy as np
from numba import njit
from global_variables import *

@njit(fastmath=True)
def unison_shuffled_copies(a):
    np.random.seed(1)
    p = np.random.permutation(len(a))
    return p

@njit(fastmath=True)
def select_actors(events, step):    
    actors = np.nonzero(events[step])[0]
    interactions = events[step][actors]
    shuffled_idx = unison_shuffled_copies(actors)
    
    values = np.empty((len(actors), 2))
    values[:,0] = actors[shuffled_idx]
    values[:,1] = interactions[shuffled_idx]
    
    return values

@njit(fastmath=True)
def transform_probability(array):
    total = np.sum(array)    
    if total == 0:
        values = [999]
    values = array / total 
    return values

@njit(fastmath=True)
def find_nearest_above(my_array, target):
    return np.searchsorted(my_array,[target,],side='right')[0]


# @profile
@njit(fastmath=True)
def select_alter(agent, interaction_m, interaction_history, similarity):
    actor = int(agent[0])
    interactions = int(agent[1])
    actor_similarity = similarity[actor, :]
    # print(actor_similarity)
    for interaction in range(interactions):
        while(True):
            interaction_probability = transform_probability(actor_similarity)
            if interaction_probability[0] == 999:
                return
            sorted_probability = np.argsort(interaction_probability)[::-1]
            interaction_probability = np.cumsum(interaction_probability[sorted_probability])
            dice = np.random.sample()
            alter = sorted_probability[find_nearest_above(interaction_probability, dice)]

            idx = np.sort(np.array([actor, alter]))
            if(interaction_history[idx[0]][idx[1]]!=1):
                interaction_history[idx[0]][idx[1]] = 1
                interaction_m[idx[0]] += 1
                interaction_m[idx[1]] += 1
                return alter, interaction_m, interaction_history, similarity
            else:
                similarity[idx[0], idx[1]] = 0

@njit(fastmath=True)
def get_prestige(actor, alter, prestige, prestige_p):
    return prestige[int(actor)]*prestige_p, prestige[alter]*prestige_p

@njit(fastmath=True)
def calc_status_difference(actor, alter, prestige, status, prestige_p, ses_noise):
    agents = np.array([actor, alter], dtype=np.uint16)
    #collector
    # similarity_interactions.append(similarity[actor, alter])
    
    prestige_actor, prestige_alter = get_prestige(actor, alter, prestige, prestige_p)
    noise_actor = np.random.uniform(-ses_noise + prestige_actor, ses_noise + prestige_actor)
    noise_alter = np.random.uniform(-ses_noise + prestige_alter, ses_noise + prestige_alter)
    
    actor_pses = np.max(np.array([status[actor] + noise_actor, 0]))
    alter_pses = np.max(np.array([status[alter] + noise_alter, 0])) 
    
    pses_arr = np.array([actor_pses, alter_pses])
    
    idx_min = int(np.argmin(pses_arr))
    idx_max = int(np.argmax(pses_arr))
    
    perceiver = agents[idx_min]
    discriminator = agents[idx_max]

    status_difference = np.absolute(actor_pses - alter_pses)
    return status_difference, perceiver, discriminator

@njit(fastmath=True)
def get_vulnerability(stress, vul_p):
    return (-0.01 + 1.0/(1+np.exp(-.05*(stress-90)))) * vul_p


@njit(fastmath=True)
def calc_stressor(status_difference, vulnerability, stressor_p):
    return (-0.04 + 1.05/(1+np.exp(-0.4*(status_difference-8)))) * stressor_p * (1+vulnerability)

@njit(fastmath=True)
def cope(stressor, psr, psr_p, coping_noise):
    #collector
    # stressors.append(stressor)
    stressor = stressor - (psr*psr_p) + np.random.normal(loc=0.0, scale=coping_noise)
    
    #collector
    # coped_stress.append(stressor-stress)
    return stressor if stressor > 0 else 0

# @profile
@njit(fastmath=True)
def update_prestige(agents, target, status, prestige, prestige_beta, prestige_min):
    # if status[agents[0]] == status[agents[1]]: return prestige; 
    
    status_group = status[agents[target]]
    group_members = np.where(status == status_group)[0]
    group_member = group_members[0]
    
    prestige_group = prestige[group_member] if prestige[group_member] > prestige_min else prestige_min 

    group_distance = 1 / prestige_group 
    
    if target == 0: 
        prestige[group_members] = 1 / (group_distance * (1 + (prestige_beta / len(group_members))))
        # prestige[group_members] = 1 / (group_distance * (1 + (prestige_beta)))
    else: 
        prestige[group_members] = 1 / (group_distance * (1 - (prestige_beta / len(group_members))))
        # prestige[group_members] = 1 / (group_distance * (1 - (prestige_beta)))

    return prestige

@njit(fastmath=True)
def recover(stress, recover_p, stress_max):
    stress -= recover_p
    stress[stress < 0] = 0
    stress[stress > stress_max] = stress_max
    return stress

# @profile
@njit(fastmath=True)
def interact_actors(actors, interactions, interaction_history, similarity, prestige, status, psr, prestige_p, ses_noise, prestige_beta, psr_p, stress, vul_p, stressor_p, prestige_min,status_diff, coping_noise, interaction_nr):
    for i in range(len(actors)):
        agent = actors[i]
        alter, interactions, interaction_history, similarity = select_alter(agent, interactions, interaction_history, similarity)
        status_difference, perceiver, discriminator = calc_status_difference(int(agent[0]), int(alter), prestige, status, prestige_p, ses_noise)
        
        #collector
        # status_diff[:, interaction_nr] = [status[perceiver], status[discriminator], status_difference]
        # interaction_nr += 1

        vulnerability = get_vulnerability(stress[perceiver], vul_p)
        stressor = calc_stressor(status_difference, vulnerability, stressor_p)
        stress[perceiver] += cope(stressor, psr[perceiver], psr_p, coping_noise)
        prestige = update_prestige(np.array([perceiver, discriminator]), np.random.randint(2), status, prestige, prestige_beta, prestige_min)
    return interactions, interaction_history, stress, similarity, prestige, status_diff, interaction_nr


def normalize(arr, lower, upper):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized = (upper-lower)*(arr-arr_min)/(arr_max-arr_min) + lower
    return np.array(normalized)


def calculate_chronic_state(stress, chronic_t, time, step, repeat):
    chronic_state = np.min(stress[:,step-WEEK:step, repeat], axis=1)
    # chronic_intensity = np.mean(stress[:, step-WEEK:step], axis=1)
    chronic_state[chronic_state >= chronic_t] = 1
    chronic_state[chronic_state < chronic_t] = 0
    return chronic_state

# @njit
def calculate_slopes(stress):
    slopes = np.zeros(shape=(stress.shape[0],stress.shape[1]-1,stress.shape[2]), dtype=np.float32)
    stress = np.asarray(stress, dtype=np.float32)
    for i in range(stress.shape[2]):
        slopes[:, :, i] = stress[:, 1:stress.shape[1], i] - stress[:, 0:stress.shape[1]-1, i]
    return slopes 