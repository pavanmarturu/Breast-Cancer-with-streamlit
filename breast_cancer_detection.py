#importing the essental libraries 

import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer


cancer = load_breast_cancer()
cancer
cancer.keys()
cancer.values()

print(cancer['DESCR'])


print(cancer['target'])

print(cancer['target_names'])

print(cancer['feature_names'])

cancer['data'].shape

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns= np.append(cancer['feature_names'],['target']))

df_cancer.head()

df_cancer.tail()


# # *SPLITTING THE DATASET*

x = df_cancer.drop(['target'],axis =1)

x

y= df_cancer['target']

y

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

x_train

y_train

x_test

y_test


# # *TRAINING THE MODEL USING SVM*

from sklearn.svm import SVC
from sklearn.metrics import classification_report , confusion_matrix

svc_model= SVC()

svc_model.fit(x_train,y_train)


# # *EVALUATING THE MODEL*

y_predict =svc_model.predict(x_test)

y_predict


cm = confusion_matrix(y_test,y_predict)


# # Model Improvisation

min_train =x_train.min()


range_train =(x_train - min_train).max()

x_train_scaled =(x_train-min_train)/range_train

min_test =x_test.min()
range_test =(x_test - min_test).max()
x_test_scaled =(x_test-min_test)/range_test

svc_model.fit(x_train_scaled,y_train)

y_predict =svc_model.predict(x_test_scaled)

cn = confusion_matrix(y_test,y_predict)

print(classification_report(y_test,y_predict))


# ### An accuracy of 96% has been achieved after appying the technique of Normalization for Improvisation

param_grid ={'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']}

from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


grid.fit(x_train_scaled,y_train)

grid.best_params_


grid_predictions=grid.predict(x_test_scaled)

cn =confusion_matrix(y_test,grid_predictions)

print(classification_report(y_test,grid_predictions))


#Saving the trained model using pickle
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(svc_model, model_file)

# ### Accuracy of 97% has been achieved by further Improvisation by optimization of C and Gamma Parameters



