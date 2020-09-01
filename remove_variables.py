#%%
import glob
import numpy as np
import pickle
import os


if __name__ == "__main__":
    root = os.path.dirname(os.path.realpath(__file__))
    folder = "pre-testB"    
    files = np.array(glob.glob(root+ "\\results\\"+folder+"\\*.pkl"))
    print(len(files[10:]))
    for file in files[52:]:
        print(file)
        with open(file, 'rb') as f:
            results = pickle.load(f)
        
        del results['status_difference']
        del results['stressors']
        del results['coped_stress']
        del results['similarity_interactions']
        f = open(file, "wb")
        pickle.dump(results, f)
# %%
