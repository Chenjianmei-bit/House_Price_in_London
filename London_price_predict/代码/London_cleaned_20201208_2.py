

#%% Import 

import pandas as pd
import numpy as np


#%% load data  

hpRaw=pd.read_csv("London.csv")

hpB = hpRaw
hpRaw.columns

hpS=hpRaw.iloc[:,[2,4,5,6,7,3,9]]
hpS.columns=['price','area','bedrooms','bathrooms','receptions',
             'houseType', 'cityCounty']

hpSD=pd.get_dummies(hpS,
                    columns = ['houseType', 'cityCounty'],
                    drop_first=True)

hpSD.to_csv('London_cleaned_20201208_2.csv',index=False)