from matplotlib.pyplot import axis

__author__ = 'Edisnel C.C.'

#Import libraries:
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier  #GBM algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import model_selection, metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.preprocessing import LabelEncoder # For One Hot Enconding
from sklearn import preprocessing
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

from sklearn.preprocessing import Imputer
from sklearn import neighbors

from sklearn.neural_network import MLPClassifier


def normalize(df, colums):
    result = df.copy()
    for feature_name in colums:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=False, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Loan_Status'])

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['Loan_Status'], cv=cv_folds, scoring='roc_auc')

    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Loan_Status'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Loan_Status'], dtrain_predprob))

    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train['source'] = 'train'
test['source'] = 'test'

data = pd.concat([train, test], ignore_index=True)

# Data frame dimensions, like dim in R
#print train.shape, test.shape, data.shape

# Missing values
data.apply(lambda x: sum(x.isnull()), axis = 0)

# Unique values
data.apply(lambda x: len(x.unique()))

# Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x] == 'object']

# Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']]

# Print frequency of different categories in each categorical variable
for col in categorical_columns:
    print '\nFrequency of Categories for varible %s'%col
    print data[col].value_counts()

# Imputing Missing Values for Gender
miss_gender = data['Gender'].isnull()
data.loc[miss_gender, 'Gender'] = 'Male'
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0}) # data[data['Gender'] == 'Male'] = 1

# Converting Education values to 1 and 0
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})

# Imputing missing values in Married
miss_married = data['Married'].isnull()
data.loc[miss_married, 'Married'] = 'Yes'
data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})

# ------ Imputing Dependents ---------
data['Dependents'].fillna('0', inplace=True)
data['Dependents'].replace('3+', '3', inplace=True)

# ------ Imputing Self Employed ---------

data.loc[data['Gender'] == 0 & data['Self_Employed'].isnull(), 'Self_Employed'] = 'No' # df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
data.loc[data['Gender'] == 1 & data['Self_Employed'].isnull(), 'Self_Employed'] = 'Yes'
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})

# ------ Imputing Loan Amount Term ---------
data['Loan_Amount_Term'].fillna((data['Loan_Amount_Term'].median()), inplace=True)

# ------ Imputing Loan Amount --------- poner Gender en las columnas
X = data[data['LoanAmount'].notnull()].as_matrix(columns=['Married','Self_Employed', 'Dependents'])
Y = data[data['LoanAmount'].notnull()].as_matrix(columns=['LoanAmount'])
T = data[data['LoanAmount'].isnull()].as_matrix(columns=['Married','Self_Employed', 'Dependents'])
knn = neighbors.KNeighborsRegressor(n_neighbors= 10, weights='distance')
predictions = knn.fit(X, Y).predict(T)
data.loc[data['LoanAmount'].isnull(), 'LoanAmount'] = predictions

# ------ Log transformation to nullify outlier's effect ---------
#data['LoanAmount'] = np.log(data['LoanAmount'])

# ------  Applicant Income ---------

data.loc[data['ApplicantIncome'] == 0, 'ApplicantIncome'] = data['ApplicantIncome'].median()
data['ApplicantIncome'] = np.log(data['ApplicantIncome'])

# ------ Imputing Coapplicant Income --------- poner Gender
X = data[data['CoapplicantIncome'] != 0].as_matrix(columns=['Married','Gender', 'Loan_Amount_Term'])
Y = data[data['CoapplicantIncome'] != 0].as_matrix(columns=['CoapplicantIncome'])
T = data[data['CoapplicantIncome'] == 0].as_matrix(columns=['Married','Gender', 'Loan_Amount_Term'])

knn = neighbors.KNeighborsRegressor(n_neighbors= 10, weights='distance')
predictions = knn.fit(X, Y).predict(T)
data.loc[data['CoapplicantIncome'] == 0, 'CoapplicantIncome'] = predictions

data['CoapplicantIncome'] = np.log(data['CoapplicantIncome'])

# ------ Imputing Credit_History ---------

X = data[data['Credit_History'].notnull()].as_matrix(columns=['Married', 'Education', 'Self_Employed'])
Y = data[data['Credit_History'].notnull()].as_matrix(columns=['Credit_History'])
T = data[data['Credit_History'].isnull()].as_matrix(columns=['Married', 'Education', 'Self_Employed'])

knn = KNeighborsClassifier(n_neighbors= 7)
#predictions1 = knn.fit(X, Y).predict(T)
predictions = knn.fit(X,Y).predict_proba(T)
predictions[predictions > 0.8] = 1 # Tenia 0.7
predictions[predictions <= 0.8] = 0

#data.loc[data['Credit_History'].isnull(), 'Credit_History'] = predictions1
data.loc[data['Credit_History'].isnull(), 'Credit_History'] = predictions[:,1]

#predictions[predictions > 0.9] = 1
#predictions[predictions <= 0.9] = 0
#param_test2 = {'n_neighbors':range(3,12,1)}
#gsearch1 = GridSearchCV(estimator = KNeighborsClassifier(n_neighbors=11), param_grid=param_test2, scoring='accuracy',n_jobs=4,iid=True, verbose=True, cv = 5)
#gsearch1.fit(X, Y[:,0])
#print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_ {'n_neighbors': 7}
#predictions = gsearch1.predict(T)

# ------ Property_Area One Hot Encoding ---------

le = LabelEncoder()
d = le.fit_transform(pd.Series(data['Property_Area']))
df = pd.get_dummies(d, prefix='Property_Area')

data['Property_Area_0'] = df['Property_Area_0']
data['Property_Area_1'] = df['Property_Area_1']
data['Property_Area_2'] = df['Property_Area_2']
data.drop(['Property_Area'],axis=1,inplace=True)

#----------- Dependents One Hot Encoding -----------

le = LabelEncoder()
d = le.fit_transform(pd.Series(data['Dependents']))
df = pd.get_dummies(d, prefix='Dependents')

data['Dependents_0'] = df['Dependents_0']
data['Dependents_1'] = df['Dependents_1']
data['Dependents_2'] = df['Dependents_2']
data['Dependents_3'] = df['Dependents_3']

data.drop(['Dependents'],axis=1,inplace=True)

# --------------------------After data pre-processing-------------------------------------------------------------------

train = data.loc[data['source'] == 'train']
test = data.loc[data['source'] == 'test']

train.drop(['Loan_ID','source'], axis=1,inplace=True)
test.drop(['Loan_Status','source'], axis=1,inplace=True)

target = 'Loan_Status'
IDcol = 'Loan_ID'
predictors = [x for x in train.columns if x not in [target, IDcol]]

train = normalize(train, predictors)
test = normalize(test, predictors)

train.to_csv("data_train.csv",index=False)
test.to_csv("data_test.csv",index=False)

# ------------------------ GradientBoostingClassifier -------------------------------------

#param_test2 = {'n_estimators':range(20,151,10), 'random_state':range(10,21,2), 'learning_rate':[0.01,0.02,0.03,0.04]}
#param_test2 = {'max_features':range(3,10,1)}
param_test2 = {'random_state':range(11,31,5)}
#max_features=4, randomstate

gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.01,\
       n_estimators=150, max_depth=2, max_features=4, min_samples_split=25, min_samples_leaf=5,\
       subsample=0.7), param_grid=param_test2, scoring='accuracy',\
       n_jobs=4,iid=False, verbose=True, cv=5)

gsearch1.fit(train[predictors], train[target])

#print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
print gsearch1.best_params_, "\n", gsearch1.best_score_

predictions1 = gsearch1.predict(test[predictors])

df_result = pd.DataFrame({ 'Loan_ID' : test['Loan_ID'], 'Loan_Status' : predictions1 })
df_result.to_csv('gboost/sol_gboost_py4.csv', index=False)

# ------------------------ Neural Network -------------------------------------

param_test = {'max_iter':(1000,1500,10)}
# hidden_layer_sizes=(2,10) 'hidden_layer_sizes': (2, 4, 6)

gsearch2 = GridSearchCV(estimator =\
       MLPClassifier(activation='identity', alpha=0.0001, batch_size='auto',\
       beta_1=0.1, beta_2=0.999, early_stopping=False,\
       epsilon=1e-08, learning_rate='adaptive', hidden_layer_sizes=(2,10),\
       learning_rate_init=0.001, max_iter=1000, momentum=0.01,\
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=False,\
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=True,\
       warm_start=False),\
       param_grid=param_test, scoring='accuracy',n_jobs=4,iid=False, verbose=True, cv=5)

gsearch2.fit(train[predictors], train[target])

#print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
print gsearch2.best_params_, "\n", gsearch2.best_score_

predictions2 = gsearch2.predict(test[predictors])

df_result = pd.DataFrame({ 'Loan_ID' : test['Loan_ID'], 'Loan_Status' : predictions2 })
df_result.to_csv('nnet/sol_neuralnet_py9.csv', index=False)

# ------------------------------- Ada Boost Classifier ------------------------------

param_test2 = {'n_estimators':range(200,300,20)}
#max_features=4, randomstate

gsearch3 = GridSearchCV(estimator = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),
                         algorithm="SAMME", n_estimators=250),
                         param_grid=param_test2, scoring='accuracy',\
                         n_jobs=4,iid=False, verbose=True, cv=5)

gsearch3.fit(train[predictors], train[target])
print gsearch3.best_params_, "\n", gsearch3.best_score_

predictions3 = gsearch3.predict(test[predictors])

df_result = pd.DataFrame({ 'Loan_ID' : test['Loan_ID'], 'Loan_Status' : predictions1 })
df_result.to_csv('zAdaBoost/sol_Adaboost_py3.csv', index=False)

#------------------------- Support Vetor Machine ----------------------------------------

param_test = {'coef0':(0.0,0.1,0.2), 'degree': range(1,4,1)}

gsearch4 = GridSearchCV(estimator = SVC(kernel='poly', probability=True), param_grid=param_test, scoring='accuracy',\
       n_jobs=4,iid=False, verbose=True, cv=3)

gsearch4.fit(train[predictors], train[target])

print gsearch4.best_params_, "\n", gsearch4.best_score_

predictions4 = gsearch4.predict(test[predictors])

df_result = pd.DataFrame({ 'Loan_ID' : test['Loan_ID'], 'Loan_Status' : predictions4 })
df_result.to_csv('svm/sol_svm_py1.csv', index=False)

# ------------------------ VotingClassifier ---------------

pred_df = pd.DataFrame({'p1' : [predictions1],
 'p2':[predictions2],
 'p3':[predictions3]})

#, weights=[5, 1, 1] -- soft voting
eclf = VotingClassifier(estimators=[('gb', gsearch1), ('nn', gsearch2), ('AdaBoost', gsearch3)], voting='soft', weights=[5, 1, 1])

c = train[target]
eclf = eclf.fit(train[predictors], c)

predictions = eclf.predict(test[predictors])

df_result = pd.DataFrame({ 'Loan_ID' : test['Loan_ID'], 'Loan_Status' : predictions })
df_result.to_csv('votingClassif/sol_votingclasiffier_py3.csv', index=False)














