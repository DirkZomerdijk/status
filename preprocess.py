#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import matplotlib as mpl
from global_variables import *
from functions import *


def filter_dataset(df):
    '''
    -1: unknown
    1: never been to school or elementary schooling only 
    2: lower vocational schooling or lower secondary schooling
    3: intermediate vocational schooling or intermediate/higher secondary schooling (general)
    4: higher vocational schooling or university
    
    Filter unkown 
    ''' 
    start = total = df.shape[0]
    
    print("Total participants", total)
    
    df = df[df['H1_Opleid'] != -1]
    print("no education reported", total - df.shape[0])
    total = df.shape[0]

    '''
    Beroeps niveau:
    -9: Missing
     1: elementaire beroepen
     2: lagere beroepen
     3: middelbare beroepen
     4: hogere beroepen
     4: wetenschappelijke beroepen
     
     Filter missing data
    '''
    # df = df[df['H1_BeroepsNiveau'] != -9]
    # print("no occupational level reported", total - df.shape[0])
    # total = df.shape[0]
    
    '''
    Arbeids participatie:
    -9: onbekend
     1: werkend
     2: niet in beroepsbevolking (pensieon, huisvrouw/huisman, studeren/school
     3: werkloos/bijstand
     4: arbeidsongechikt
     
     Filter unknown 
    '''
    df = df[df['H1_Arbeidsparticipatie'] != -9]
    print("no occupation reported",total - df.shape[0])
    total = df.shape[0]

    df = df[df['H1_InkHhMoeite'] != -1]
    print("no education income dificulties reported", total - df.shape[0])
    total = df.shape[0]

    df = df[df['H1_Extraversion_sumscore'] != -1]
    print("no extraversion reported",total - df.shape[0])
    total = df.shape[0]

    df = df[df['H1_etniciteit'] < 6]
    print("no ethnicity reported",total - df.shape[0])
    total = df.shape[0]

    df = df[df['H1_Mastery_Sumscore'] != -1]
    print("no mastery reported",total - df.shape[0])
    total = df.shape[0]

    df = df[df['H1_SSQT'] != -1]
    print("no perceived support reported",total - df.shape[0])
    total = df.shape[0]
    
    df = df[df['H1_LO_BMI'] != " "]
    print("no BMI reported", total - df.shape[0])
    total = df.shape[0]
    
    df["H1_LO_BMI"] = pd.to_numeric(df["H1_LO_BMI"], downcast="float")

    df = df[df['H1_SSQSa'] != -1]
    print("no perceived support reported",total - df.shape[0])
    total = df.shape[0]
    print("Participants left", total)
    print("Participants filtered", start-total)
    return df

def calculate_occupation_composite(df):
    
    '''
    Arbeids participatie:
    -9: onbekend
     1: werkend
     2: niet in beroepsbevolking (pensieon, huisvrouw/huisman, studeren/school
     3: werkloos/bijstand
     4: arbeidsongechikt

    Beroeps niveau:
    -9: Missing
     1: elementaire beroepen
     2: lagere beroepen
     3: middelbare beroepen
     4: hogere beroepen
     4: wetenschappelijke beroepen
    '''
    print('calculating occupation comopsite...')

    df['occupation'] = -1
    df.loc[(df['H1_BeroepsNiveau']==-9) & (df['H1_Arbeidsparticipatie']==3), 'occupation'] = 1
    df.loc[(df['H1_BeroepsNiveau']==-9) & (df['H1_Arbeidsparticipatie']==4), 'occupation'] = 2
    df.loc[(df['H1_BeroepsNiveau']==-9) & (df['H1_Arbeidsparticipatie']==2), 'occupation'] = 3
    df.loc[(df['H1_BeroepsNiveau']==-9) & (df['H1_Arbeidsparticipatie']==1), 'occupation'] = 4
    df.loc[(df['H1_BeroepsNiveau']!=-9), 'occupation'] = df.loc[(df['H1_BeroepsNiveau']!=-9), 'H1_BeroepsNiveau'] + 3
    
    print(df['occupation'].unique())
    return df

def calculate_status(df):
    status = []
    for i, row in df[['occupation', 'H1_Opleid', 'H1_InkHhMoeite']].iterrows():
        status.append(np.sum([1, status_dict['occ'][row['occupation']], status_dict['edu'][row['H1_Opleid']], status_dict['inc'][row['H1_InkHhMoeite']]]))
    return status

def calculate_status_linear(df):
    status = []
    for i, row in df[['occupation', 'H1_Opleid', 'H1_InkHhMoeite']].iterrows():
        status.append(np.sum([1,status_dict_linear['occ'][row['occupation']], status_dict_linear['edu'][row['H1_Opleid']], status_dict_linear['inc'][row['H1_InkHhMoeite']]]) + 1)
    status = np.array(status)
    status[status < 0.1] = 0.1
    return status

def calculate_psr(df):
    return normalize(normalize(df['H1_Mastery_Sumscore'], 0, 1) + normalize(df['H1_SSQSa'], 0, 1), 0, 1)

if __name__ == "__main__":
    df = pd.read_csv('./data/data2.csv')
    df = filter_dataset(df)
    df = calculate_occupation_composite(df)
    df['status_l'] = calculate_status_linear(df) 
    df['status'] = calculate_status(df)
    df['prestige'] = df['status'] / MAX_STATUS_DIFFERENCE
    
    print(df['status'].min(),df['status'].max())
    print(df['status_l'].min(),df['status_l'].max())
    df['psr'] = calculate_psr(df)
    df.to_csv("./data/clean.csv", index=False)
    
    pass

# %%
