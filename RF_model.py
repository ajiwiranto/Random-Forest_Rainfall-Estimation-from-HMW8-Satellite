import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#split data
from sklearn.model_selection import train_test_split

#ML_model_random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

#metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix,classification_report

d = pd.read_csv('/home/ajiwiranto/Documents/kodingan/TA/df_hmwr-gpm/data_fix/data_all/df_hmw-gpm_final.csv')

d1 = d[['10.4-11.2', '8.6-10.4','8.6-12.4','10.4-12.4', '6.2-6.9', '7.3-10.4','7.3-9.6','6.2-9.6','7.3-12.4','7.3-13.3', 'prec']]

#check null
null_columns = d1.columns[d1.isnull().any()]
d1[null_columns].isnull().sum()

d1.shape

nlen2 = 218199
labels2 = np.array(list(range(nlen2)))
ndarr2 = np.array([0, 10, 20, 30, 40, 50])
rnum = labels2[d1['prec'] > 0. ]
cnum = labels2[d1['prec'] == 0.]
rcnt = len(rnum)
ccnt = len(cnum)
print('rnum(rain)   =',rcnt)
print('cnum(no_rain)=',ccnt)

hnum = labels2[ d1['prec'] > 1.8 ]
snum = labels2[ (d1['prec'] > 0.) & (d1['prec'] <= 1.8) ]

hcnt = len(hnum)
scnt = len(snum)

print('hcnt(strong)=', hcnt)
print('scnt(weak)=', scnt)

ratioR = 1.5
probR = ((rcnt*ratioR)/ccnt)

from sklearn.model_selection import train_test_split

cnum_train, cnum_test, = train_test_split(cnum, train_size=probR, random_state = 40)

dfc = d1.loc[cnum_train,:]
dfr = d1.loc[rnum,:]
dfc['classification']= 0
dfr['classification']= 1
df = pd.concat((dfc, dfr), axis=0)


bands = df.iloc[:,0:10]
clf   = df.iloc[:,11]
#separate to train & test data
X_train, X_val, y_train, y_val = train_test_split(bands, clf, random_state = 40, test_size = 0.1)
# define model
model_rainarea = RandomForestClassifier(n_estimators=500, oob_score= True)

#making model
model_rainarea.fit(X_train, y_train)

#==============================================================
# Predict Rain no rain
#==============================================================

ratioT =  1
probT = ((hcnt*ratioT)/scnt)
from sklearn.model_selection import train_test_split

snum_train, snum_test, = train_test_split(snum, train_size=probT, random_state = 40)
snum_test.sort()
snum_train.sort()

dfs = d1.loc[snum_train,:]
dfh = d1.loc[hnum,:]

dfs['class_T']= 0
dfh['class_T']= 1
df = pd.concat((dfs, dfh), axis=0)

bands_T = df.iloc[:,0:10]
clf_T   = df.iloc[:,11]

#separate to train & test data
X_train_T, X_val_T, y_train_T, y_val_T = train_test_split(bands_T, clf_T, random_state = 92, test_size = 0.1)
# define model
model_Type = RandomForestClassifier(n_estimators=500, oob_score= True)

#making model
model_Type.fit(X_train_T, y_train_T)

#===============================================
#Weak Rain
#===============================================

band_weak = dfs.iloc[:,0:10] 
rain_weak = dfs.iloc[:,10]

#separate to train & test data
X_train_Rw, X_val_Rw, y_train_Rw, y_val_Rw = train_test_split(band_weak, rain_weak, random_state = 40, test_size = 0.1)

# define model
model_Rw = RandomForestRegressor(oob_score = True)

#making model
model_Rw.fit(X_train_Rw, y_train_Rw)

#=============================================
#strongrain
#=============================================

band_strong = dfh.iloc[:,0:10] 
rain_strong = dfh.iloc[:,10]

#separate to train & test data
X_train_Rs, X_val_Rs, y_train_Rs, y_val_Rs = train_test_split(band_strong, rain_strong, random_state = 40, test_size = 0.1)

# define model
model_Rs = RandomForestRegressor(oob_score = True)

#making model
model_Rs.fit(X_train_Rs, y_train_Rs)

#==============
#nama model
#==============
print (" model_rainarea \n, model_Type \n, model_Rw \n ,model_Rs")





