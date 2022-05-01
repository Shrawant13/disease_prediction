import pandas as pd
import numpy as np


train=pd.read_csv('Training.csv')
test=pd.read_csv('Testing.csv')

train['prognosis']=train['prognosis'].astype('category')
test['prognosis']=test['prognosis'].astype('category')
# Divide into train and test
from sklearn.model_selection import train_test_split
train_target=train["prognosis"]
train1=train.drop('prognosis', axis=1)
train2,val2,train_target2,val_target2 = train_test_split(train1, train_target, test_size=0.3, random_state=42)  

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
model1 = RandomForestClassifier(n_estimators=500,max_features=11,max_depth=2)
model1.fit(train2, train_target2)
y_pred_rfr = model1.predict(train2)

import pickle
pickle.dump(RandomForestClassifier, open('model1.pkl','wb'))