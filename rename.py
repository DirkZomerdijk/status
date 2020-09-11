#%%
from plotter import *
plt.rcParams['figure.figsize'] = [12, 3]

p = param_dict
with open('./results/rejacc/folder_pre_testB.pkl', 'rb') as f:
    files = pickle.load(f)[0]

print()
print("files: ", len(files)) 
for file in files:
    with open(file, 'rb') as f:
        results = pickle.load(f)
        
    print(file)
    temp = file.rsplit("\\",1)
    print()
    new_filename = "\\".join([temp[0], "B_" + temp[1]])
    print(new_filename)
    f = open(new_filename, "wb")
    pickle.dump(results, f)
        
