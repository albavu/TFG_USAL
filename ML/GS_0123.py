#############################################################################
            # Autor: Alba Vallejo Urruchi
            # TFG Ingeniería Informática 2020-2021
            # Universidad de Salamanca
#############################################################################

#!/usr/bin/env python
# coding: utf-8


import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np

c_in = '/lhome/ext/usal053/usal0531/TFG/data'
c_out = '/lhome/ext/usal053/usal0531/TFG/ML'

## Datos (solo ejecutar en caso de que no tengamos ya el archivo con todas las secuencias)
# =========================================================
dataN = pd.read_csv(c_in+'/0123_Nuc_SP3.csv')
data = pd.read_csv(c_in+'/0123_NONuc_Pombe2.csv')

dataN = dataN.iloc[:,2:152]
data = data.iloc[0:dataN.shape[0],:]
dataN['nucleosoma'] = 1
data['nucleosoma'] = 0
total = pd.concat([dataN,data],axis=0)
## Mezclamos las filas para que no estén en orden: primero todas las secuencias que posicionan nucleosomas y luego todas las que no posicionan
total = total.sample(frac=1).reset_index(drop=True)



#np.split funciona cogiendo el %que quitamos del total y luego de lo que queda el otro %
#En nuestro caso va a ser 80% train, 10% validate y 10% test
train, validate, test = np.split(total.sample(frac=1, random_state=42), 
                       [int(.8*len(total)), int(.9*len(total))])

X_train = train.iloc[:,0:150]
y_train = train['nucleosoma']
X_validate = validate.iloc[:,0:150]
y_validate = validate['nucleosoma']
X_test = test.iloc[:,0:150]
y_test = test['nucleosoma']

##################################################################################

##                      GRID_SEARCH RANDOM FOREST

##################################################################################

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


rfc=RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [10,20,30,40,50,70,90,100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid,scoring = 'accuracy', cv= 5, n_jobs=-1)
model = CV_rfc.fit(X_train, y_train)

RF = pd.concat([pd.DataFrame(model.cv_results_["params"]),pd.DataFrame(model.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
RF = RF.sort_values(by='Accuracy', ascending=False)
RF = RF.reset_index(drop=True)
RF.to_csv(c_out+'/0123_RF_results.csv')


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

rf1 = CV_rfc.best_params_

cf1 = RandomForestClassifier(random_state=42, criterion=rf1['criterion'], max_depth=rf1['max_depth'], max_features=rf1['max_features'], n_estimators=rf1['n_estimators'])
cf1.fit(X_train, y_train)

y_pred_1 = cf1.predict(X_test)

titles_options = [("Confusion matrix", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(cf1, X_test, y_test,
                                 display_labels=['Nucleosoma','No-nucleosoma'],
                                 cmap=plt.cm.Greens,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    plt.savefig(fname=c_out+"/confusionMatrix/0123_RF_"+title)
    print(title)
    print(disp.confusion_matrix)

##################################################################################

##                      GRID_SEARCH LOGISTIC REGRESSION

##################################################################################

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

rg=LogisticRegression(random_state=42)

param_grid = {
    'penalty': ['l2', 'elasticnet'],
    'C' : [0.1,0.3,0.7,0.5,1],
    'class_weight':['balanced','None'],
    'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

CV_rg = GridSearchCV(estimator=rg, param_grid=param_grid,scoring = 'accuracy', cv= 7, n_jobs=-1)
model = CV_rg.fit(X_train, y_train)

RG = pd.concat([pd.DataFrame(model.cv_results_["params"]),pd.DataFrame(model.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
RG = RG.sort_values(by='Accuracy', ascending=False)
RG = RG.reset_index(drop=True)
RF.to_csv(c_out+'/0123_RG_results.csv')


from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

rf1 = CV_rg.best_params_

cf1 = LogisticRegression(random_state=42, penalty=rf1['penalty'], C=rf1['C'], class_weight=rf1['class_weight'], solver=rf1['solver'])
cf1.fit(X_train, y_train)

y_pred_1 = cf1.predict(X_test)

titles_options = [("Confusion matrix", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(cf1, X_test, y_test,
                                 display_labels=['Nucleosoma','No-nucleosoma'],
                                 cmap=plt.cm.Greens,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    plt.savefig(fname=c_out+"/confusionMatrix/0123_RG_"+title)
    print(title)
    print(disp.confusion_matrix)


##################################################################################

##                      GRID_SEARCH K-NEIGHBORS 

##################################################################################

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

nn=KNeighborsClassifier()

param_grid = {
    'n_neighbors': range(10, 51) 
}

CV_nn = GridSearchCV(estimator=nn, param_grid=param_grid,scoring = 'accuracy', cv= 7, n_jobs=-1)
model = CV_nn.fit(X_train, y_train)


# plot the results
# this is identical to the one we generated above
plt.plot(range(10,51), model.cv_results_["mean_test_score"])
plt.xlabel('K-vecinos')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Accuracy of k-vecinos')
plt.savefig(fname=c_out+"/confusionMatrix/KNN"+title)

NN = pd.concat([pd.DataFrame(model.cv_results_["params"]),pd.DataFrame(model.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
NN = NN.sort_values(by='Accuracy', ascending=False)
NN = NN.reset_index(drop=True)
NN.to_csv(c_out+'/0123_KNN_results.csv')


from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

rf1 = CV_nn.best_params_

cf1 = KNeighborsClassifier(n_neighbors=rf1['n_neighbors'])
cf1.fit(X_train, y_train)

y_pred_1 = cf1.predict(X_test)

titles_options = [("Confusion matrix", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(cf1, X_test, y_test,
                                 display_labels=['Nucleosoma','No-nucleosoma'],
                                 cmap=plt.cm.Greens,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    plt.savefig(fname=c_out+"/confusionMatrix/0123_KNN_"+title)
    print(title)
    print(disp.confusion_matrix)

##################################################################################

##                      GRID_SEARCH DECISION TREE

##################################################################################

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

dt=DecisionTreeClassifier()

param_grid = {
    'criterion': ['gini','entropy'],
    'class_weight': ['balanced']
}

CV_dt = GridSearchCV(estimator=dt, param_grid=param_grid,scoring = 'accuracy', cv= 7, n_jobs=-1)
model = CV_dt.fit(X_train, y_train)

DT = pd.concat([pd.DataFrame(model.cv_results_["params"]),pd.DataFrame(model.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
DT = DT.sort_values(by='Accuracy', ascending=False)
DT = DT.reset_index(drop=True)
DT.to_csv(c_out+'/0123_DT_results.csv')



from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

rf1 = CV_dt.best_params_

cf1 = DecisionTreeClassifier(class_weight=rf1['class_weight'],criterion=rf1['criterion'])
cf1.fit(X_train, y_train)

y_pred_1 = cf1.predict(X_test)

titles_options = [("Confusion matrix", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(cf1, X_test, y_test,
                                 display_labels=['Nucleosoma','No-nucleosoma'],
                                 cmap=plt.cm.Greens,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    plt.savefig(fname=c_out+"/confusionMatrix/0123_DT_"+title)
    print(title)
    print(disp.confusion_matrix)

##################################################################################

##                      GRID_SEARCH BERNOULLINB

##################################################################################

from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV


nb=BernoulliNB()

param_grid = {
    'alpha': [0,0.25,0.4,0.7,1.0],
    'binarize':[0,0.2,0.3,0.5,1.0]
}

CV_nb = GridSearchCV(estimator=nb, param_grid=param_grid,scoring = 'accuracy', cv= 7, n_jobs=-1)
model = CV_nb.fit(X_train, y_train)


NB = pd.concat([pd.DataFrame(model.cv_results_["params"]),pd.DataFrame(model.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
NB = NB.sort_values(by='Accuracy', ascending=False)
NB = NB.reset_index(drop=True)
NB.to_csv(c_out+'/0123_NB_results.csv')


from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

rf1 = CV_nb.best_params_

cf1 = BernoulliNB(alpha=rf1['alpha'],binarize=rf1['binarize'])
cf1.fit(X_train, y_train)

y_pred_1 = cf1.predict(X_test)

titles_options = [("Confusion matrix", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(cf1, X_test, y_test,
                                 display_labels=['Nucleosoma','No-nucleosoma'],
                                 cmap=plt.cm.Greens,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    plt.savefig(fname=c_out+"/confusionMatrix/0123_NB_"+title)
    print(title)
    print(disp.confusion_matrix)





