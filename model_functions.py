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
    agents = np.array([actor, alter], dtype=np.int8)
    
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
    # collector
    # status_difference.append(status_difference)
    return status_difference, perceiver, discriminator

@njit(fastmath=True)
def get_vulnerability(stress, vul_p):
    return (-0.040 + 1.043/(1+np.exp(-0.8*(stress-4.0)))) * vul_p

@njit(fastmath=True)
def calc_stressor(status_difference, vulnerability, stressor_p):
    x_null = 3
    L = 1.049
    k = 1
    return (-0.049 + L/(1+np.exp(-k*(status_difference-x_null)))) * stressor_p * (1+vulnerability)

@njit(fastmath=True)
def cope(stressor, psr, psr_p):
    #collector
    # stressors.append(stressor)
    stressor = stressor - psr*psr_p + np.absolute(np.random.normal(loc=0.0, scale=psr_p))
    
    #collector
    # coped_stress.append(stressor-stress)
    return stressor if stressor > 0 else 0

@njit(fastmath=True)
def update_prestige(agents, target, status, prestige, prestige_beta):
    status_group = status[agents[target]]
    group_members = np.where(status == status_group)
    group_member = group_members[np.random.randint(len(group_members))]
    member_distance = 1 / prestige[group_member]
    
    if target == 0:
        new_member_distance = member_distance * (1 + prestige_beta)
    else:
        new_member_distance = member_distance * (1 - prestige_beta)

    new_member_prestige = 1 / new_member_distance
    prestige[group_member] = new_member_prestige    
    return prestige

@njit(fastmath=True)
def recover(stress, recover_p, stress_max):
    stress -= recover_p
    stress[stress < 0] = 0
    stress[stress > stress_max] = stress_max
    return stress

@njit(fastmath=True)
def interact_actors(actors, interactions, interaction_history, similarity, prestige, status, psr, prestige_p, ses_noise, prestige_beta, psr_p, stress, vul_p, stressor_p):
    for i in range(len(actors)):
        agent = actors[i]
        alter, interactions, interaction_history, similarity = select_alter(agent, interactions, interaction_history, similarity)
        status_difference, perceiver, discriminator = calc_status_difference(int(agent[0]), int(alter), prestige, status, prestige_p, ses_noise)
        vulnerability = get_vulnerability(stress[perceiver], vul_p)
        stressor = calc_stressor(status_difference, vulnerability, stressor_p)
        stress[perceiver] += cope(stressor, psr[perceiver], psr_p)
        prestige = update_prestige(np.array([perceiver, discriminator]), np.random.randint(2), status, prestige, prestige_beta)
    return interactions, interaction_history, stress, similarity, prestige


def normalize(arr, lower, upper):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized = (upper-lower)*(arr-arr_min)/(arr_max-arr_min) + lower
    return np.array(normalized)

def calculate_chronic_state(stress, chronic_t, time, step, repeat):
    chronic_state = np.min(stress[:,step-WEEK:step, repeat], axis=1)
    chronic_intensity = np.mean(stress[:, step-WEEK:step], axis=1)
    chronic_state[chronic_state > chronic_t] = 1
    # print(chronic_state)
    return chronic_state
# , chronic_i_ts