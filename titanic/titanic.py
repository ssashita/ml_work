#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
from preproc.py import FullPipeline
#get_ipython().run_line_magic('run', 'preproc.ipynb')
#get_ipython().system('jupyter kernelspec list')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
import sys
print('Python3 executable is ',sys.executable)


data = pd.read_csv("datasets/titanic/train.csv")


# In[51]:


data.describe()

# ### Run the data through the preprocessing pipeline, to binarize categoricals, fill NaNs etc.

full_pipel = FullPipeline()
data_prepared = full_pipel.full_pipeline_apply_features(data,non_num_attrs=["Sex","Ticket","Cabin","Embarked"], num_attrs=["Pclass","Age","SibSp","Parch","Fare"])
labels_prepared = full_pipel.full_pipeline_apply_labels(data,label_data_non_num=["Survived"])
print('data_prepared.shape',data_prepared.shape)
print("labels_prepared.shape",labels_prepared.shape)

data_train,data_test,label_train,label_test = train_test_split(data_prepared,labels_prepared,test_size=0.2,random_state=42)


# In[52]:


label_train = label_train.ravel()
label_test= label_test.ravel()


# ### Quickly Try Decision tree

# In[53]:


dtree_classifier = DecisionTreeClassifier(random_state=42,max_depth=8)
dtree_classifier.fit(data_train,label_train)
dtree_classifier.score(data_test, label_test)


# ### Then RandomForest

# In[225]:


F= int(np.log2(data_prepared.shape[1])+1)
rndf_classifier = RandomForestClassifier(n_estimators=1000,max_features=F,random_state=42,oob_score=True, bootstrap=True, n_jobs=-1)
rndf_classifier.fit(data_train, label_train.ravel())
rndf_classifier.score(data_test, label_test.ravel())


# In[91]:


rndf_classifier.oob_score_


# ### Logistic Regression

# In[178]:


from sklearn.linear_model import LogisticRegressionCV

logit_classifier=LogisticRegressionCV(Cs=[1.0],max_iter=2000,cv=5,random_state=42,n_jobs=-1,penalty='l2',solver='newton-cg')
logit_classifier.fit(data_train, label_train)
print('score on test data',logit_classifier.score(data_test, label_test))


# ### SVM classification using kernel trick

# In[282]:


from sklearn.svm import SVC

svc_classifier=SVC(C=1000,degree=3,gamma='scale',kernel='poly',coef0=0.1,decision_function_shape='ovo')
svc_classifier.fit(data_train, label_train)
print(svc_classifier.score(data_test, label_test))


# ### Ada Boost 

# In[307]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error

ada_best = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7),learning_rate=0.70,n_estimators=150,random_state=49,algorithm="SAMME.R")
ada_best.fit(data_train,label_train)
ada_best.score(data_test,label_test)


# ### Voting Classifier

# In[310]:


from sklearn.ensemble import VotingClassifier
logit_classifier=LogisticRegressionCV(Cs=[1.0],max_iter=2000,cv=5,random_state=42,n_jobs=-1,penalty='l2',solver='newton-cg')
rndf_classifier = RandomForestClassifier(n_estimators=2000, max_depth=7,max_features=F,random_state=42,bootstrap=True, n_jobs=-1)
ada_best = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7),learning_rate=0.70,n_estimators=150,random_state=49,algorithm="SAMME.R")
svc_classifier=SVC(C=1000,degree=3,gamma='scale',kernel='poly',coef0=0.1,decision_function_shape='ovo',probability=True)

vote_classifier = VotingClassifier(estimators=[('ada',ada_best),('logist',logit_classifier),('svc', svc_classifier)], voting="soft")
vote_classifier.fit(data_train,label_train.ravel())
print(vote_classifier.score(data_test, label_test.ravel()))

#print(svc_classifier.fit(data_train,label_train).score(data_test, label_test.ravel()))


# In[323]:

# ### Code for First submission to kaggle

testdata = pd.read_csv("datasets/titanic/test.csv")
testdata_prepared = full_pipel.full_features_pipeline_.transform(testdata)
testdata['Survived']=vote_classifier.predict(testdata_prepared)
testdata[['PassengerId','Survived']].to_csv(path_or_buf="datasets/titanic/results.csv",header=True,index=False)


# In[324]:


vote_classifier.fit(data_prepared,labels_prepared.ravel())
print(vote_classifier.score(data_test, label_test.ravel()))


# In[325]:

# ### Code for second submission to kaggle

testdata = pd.read_csv("datasets/titanic/test.csv")
testdata_prepared = full_pipel.full_features_pipeline_.transform(testdata)
testdata['Survived']=vote_classifier.predict(testdata_prepared)
testdata[['PassengerId','Survived']].to_csv(path_or_buf="datasets/titanic/results2.csv",header=True,index=False)


# In[326]:


result1=pd.read_csv('datasets/titanic/results.csv',index_col='PassengerId')
result2=pd.read_csv('datasets/titanic/results2.csv',index_col='PassengerId')


# In[331]:


mean_squared_error(result1,result2)


# In[374]:

# ### Code for 4th submission to kaggle (This gave me best position on leaderboard so far)

#Use only ada boost
testdata = pd.read_csv("datasets/titanic/test.csv")
testdata_prepared = full_pipel.full_features_pipeline_.transform(testdata)
testdata['Survived']=ada_classifier.predict(testdata_prepared)
testdata[['PassengerId','Survived']].to_csv(path_or_buf="datasets/titanic/results4.csv",header=True,index=False)


# In[378]:


result1=pd.read_csv('datasets/titanic/results.csv',index_col='PassengerId')
result2=pd.read_csv('datasets/titanic/results2.csv',index_col='PassengerId')
result3=pd.read_csv('datasets/titanic/results4.csv',index_col='PassengerId')

mean_squared_error(result2,result1)


# # Dimension reduction (As it turned out, dimension reduction using both pca and kernel pca lead to worse results

#  ### Using PCA

# In[57]:


from sklearn.decomposition import PCA, KernelPCA, SparsePCA

pca=PCA(random_state=42, n_components=0.95)
X_reduced = pca.fit_transform(data_train.toarray())
X_test_reduced = pca.transform(data_test.toarray())


# In[58]:


print("Columns reduced to ",X_reduced.shape[1])


# ### Random Forest with reduced features

# In[60]:


F= int(np.log2(X_reduced.shape[1]))
rndf_classifier = RandomForestClassifier(n_estimators=2000, max_features=F,random_state=42,oob_score=True, bootstrap=True, n_jobs=-1)
rndf_classifier.fit(X_reduced, label_train.ravel())
rndf_classifier.score(X_test_reduced, label_test.ravel())


# ### Lasso logistic regression to further reduce features

# In[61]:


lasso_classifier=LogisticRegressionCV(Cs=[1.0],max_iter=2000,cv=5,random_state=42,n_jobs=-1,penalty='l1',solver='saga')
lasso_classifier.fit(X_reduced, label_train)
print('score on test data',lasso_classifier.score(X_test_reduced, label_test))


# In[73]:


X_reduced_after_lasso = X_reduced[:,lasso_classifier.coef_[0]!=0.0]
X_test_reduced_after_lasso = X_test_reduced[:,lasso_classifier.coef_[0]!=0.0]
print("Columns further reduced to",X_reduced_after_lasso.shape[1])


# ### Now Logistic with l2 penalty and reduced features - same answer for score as Lasso (l1)

# In[177]:


logit_classifier=LogisticRegressionCV(Cs=[1.0],max_iter=2000,cv=5,random_state=42,n_jobs=-1,penalty='l2',solver='newton-cg')
logit_classifier.fit(X_reduced_after_lasso, label_train)
print('score on test data',logit_classifier.score(X_test_reduced_after_lasso, label_test))


# ### Random Forest with lasso reduced features

# In[83]:


F= int(np.log2(X_reduced_after_lasso.shape[1]))
rndf_classifier = RandomForestClassifier(n_estimators=2000, max_depth=7,max_features=F,random_state=42,oob_score=True, bootstrap=True, n_jobs=-1)
rndf_classifier.fit(X_reduced_after_lasso, label_train.ravel())
rndf_classifier.score(X_test_reduced_after_lasso, label_test.ravel())
rndf_classifier.oob_score_


# ### Attempt voting classifier for the X_reduced_after_lasso data
# ### This was my 3rd attempt uploading to Kaggle, but didnt improve over previous

# In[361]:


from sklearn.svm import SVC

svc_classifier=SVC(C=1000,degree=4,gamma='scale',kernel='poly',coef0=0.1,decision_function_shape='ovo',probability=True)
svc_classifier.fit(X_reduced_after_lasso, label_train)
print(svc_classifier.score(X_test_reduced_after_lasso, label_test))


# In[358]:


ada_best = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),learning_rate=0.69,n_estimators=56,random_state=49,algorithm="SAMME.R")
ada_best.fit(X_reduced_after_lasso,label_train)
ada_best.score(X_test_reduced_after_lasso,label_test)


# In[371]:


#Before submitting to kaggle train on the entire train.csv data, do pca to reduce features, but I avoided further reduction via lasso reg
# Anyway this gave a poorer kaggle score

pca=PCA(random_state=42, n_components=0.95)
X_reduced = pca.fit_transform(data_prepared.toarray())
logit_classifier=LogisticRegressionCV(Cs=[1.0],max_iter=2000,cv=5,random_state=42,n_jobs=-1,penalty='l2',solver='newton-cg')
rndf_classifier = RandomForestClassifier(n_estimators=2000, max_depth=7,max_features=F,random_state=42,oob_score=True, bootstrap=True, n_jobs=-1)
svc_classifier=SVC(C=1000,degree=4,gamma='scale',kernel='poly',coef0=0.1,decision_function_shape='ovo',probability=True)
ada_best = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),learning_rate=0.69,n_estimators=56,random_state=49,algorithm="SAMME.R")

vote_classifier = VotingClassifier(estimators=[('ada',ada_best),('logist',logit_classifier),('svc', svc_classifier)], voting="soft")
vote_classifier.fit(X_reduced,labels_prepared.ravel())
print(vote_classifier.score(X_reduced, labels_prepared.ravel())
)


# In[372]:


testdata = pd.read_csv("datasets/titanic/test.csv")
testdata_prepared = full_pipel.full_features_pipeline_.transform(testdata)
testdata_reduced = pca.transform(testdata_prepared.toarray())
testdata['Survived']=vote_classifier.predict(testdata_reduced)
testdata[['PassengerId','Survived']].to_csv(path_or_buf="datasets/titanic/results3.csv",header=True,index=False)


# In[373]:


result1=pd.read_csv('datasets/titanic/results.csv',index_col='PassengerId')
result2=pd.read_csv('datasets/titanic/results2.csv',index_col='PassengerId')
result3=pd.read_csv('datasets/titanic/results3.csv',index_col='PassengerId')
print(mean_squared_error(result1,result2))
print(mean_squared_error(result2,result3))
print(mean_squared_error(result1,result3))


# ### Dimension Reduction using KernelPCA - this also gave poorer results, but of course reduced features quite a bit

# In[109]:


from sklearn.model_selection import GridSearchCV

pipeline = Pipeline(steps=[("kernel_pca",KernelPCA()),("dtree",DecisionTreeClassifier(max_depth=7, random_state=49))])
params_grid = [{"kernel_pca__n_components":[229,230,228],"kernel_pca__gamma":[0.03, 0.05,1.0],"kernel_pca__kernel":["rbf","sigmoid"]}]
grid_search = GridSearchCV(estimator=pipeline,cv=3,param_grid=params_grid)
grid_search.fit(data_train,label_train)
grid_search.best_params_


# In[144]:


pipeline = Pipeline(steps=[("kernel_pca",KernelPCA(gamma=0.055, kernel="sigmoid",n_components=230)),("dtree",DecisionTreeClassifier(max_depth=4, random_state=49))])
pipeline.fit(data_train, label_train)
pipeline.score(data_test,label_test)


# In[146]:


kpca = KernelPCA(gamma=0.055, kernel="sigmoid",n_components=230)
X_reduced = kpca.fit_transform(data_train)
X_test_reduced = kpca.transform(data_test)


# In[176]:


F= int(np.log2(X_reduced.shape[1]))
rndf_classifier = RandomForestClassifier(n_estimators=1500, max_depth=7,max_features=40,random_state=42,oob_score=True, bootstrap=True, n_jobs=-1)
rndf_classifier.fit(X_reduced, label_train.ravel())
rndf_classifier.score(X_test_reduced, label_test.ravel())
rndf_classifier.oob_score_


# # Ok let us take a time off here and look at the data harder

# In[93]:


data.info()


# In[ ]:


## Observer below the very slight negative correlation between Age and Survived - almost negligible.
## But strong negative correlation between class and survived
## Also 74% of the females survived but only 18% of the males


# In[179]:


data[['Age','Survived']].corr()


# In[180]:


data


# In[183]:


data[['Pclass','Survived']].corr()


# In[187]:


data[['Sex', 'Survived']].groupby('Sex').sum()/data[['Sex', 'Survived']].groupby('Sex').count()


# In[189]:


data[['SibSp','Survived']].corr()


# In[191]:


data[['Parch','Survived']].corr()


# In[194]:


data[['Fare','Survived']].corr()


# In[199]:


data[['PassengerId','Survived']].corr() #No correlation as expected


# In[200]:


data[['Fare','Pclass']].corr() #Higher the fare lower the class, of course hence high negative correlation


# In[201]:


data.corr()


# In[193]:


data.hist(bins=50,figsize=(20,15))


# In[382]:

# ### About to start looking at this animal call xgboost. They say it gives great results!

import xgboost as xgb


# In[384]:


xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(data_train,label_train)
print(xgb_classifier.score(data_test,label_test))


# In[385]:


param_grid={'max_depth': [2,4,6],
'n_estimators': [50,100,200]}
grid_search = GridSearchCV(estimator=xgb_classifier,cv=5,param_grid=param_grid,n_jobs=-1)
grid_search.fit(data_train, label_train)


# In[387]:


grid_search.best_params_


# In[ ]:




