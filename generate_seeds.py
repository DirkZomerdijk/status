#%%
import numpy as np
from randomgen import SFC64, SeedSequence
import json 
import pandas as pd
'''
Obtain entropy for seedsequencer 
'''
sq1 = SeedSequence()
print(sq1.entropy)
# entropy = 164091003547948451527772715416140446600
# entropy =196622972979125526551909400345585854006
entropy = sq1.entropy


'''
Load data
'''
# df = pd.read_csv('./data/clean.csv')
'''
generate seeds
'''
sq2 = np.random.SeedSequence(entropy)
prng_seeds = sq2.generate_state(1000)
prng_seeds

seed_sets = []
set_size = 10
for i in range(0, len(prng_seeds), set_size):
    seed_sets.append(list(np.char.mod('%d', prng_seeds[i:i+set_size])))

seedsets = {"seedsets": list(seed_sets)}
with open('seeds.json', 'w') as outfile:
    json.dump(seedsets, outfile)
# np.savetxt("seeds.csv", prng_seeds, delimiter=",", fmt='%i')
# %%


# %%
