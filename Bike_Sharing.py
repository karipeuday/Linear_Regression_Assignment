#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


bike_sharing=pd.read_csv('day (3).csv')
bike_sharing


# In[4]:


status=pd.get_dummies(bike_sharing['season'])
status.head()


# In[5]:


status=pd.get_dummies(bike_sharing['season'],drop_first=True)
status.head()


# In[6]:


bike_sharing=pd.concat([bike_sharing,status],axis=1)
bike_sharing.head()


# In[7]:


bike_sharing.rename(columns={2:'winter',3:'spring',4:'summer'},inplace=True)
bike_sharing


# In[8]:


statu=pd.get_dummies(bike_sharing['mnth'])
statu.head()


# In[9]:


statu=pd.get_dummies(bike_sharing['mnth'],drop_first=True)
statu.head()


# In[10]:


bike_sharing=pd.concat([bike_sharing,statu],axis=1)
bike_sharing.head()


# In[11]:


bike_sharing.rename(columns={2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'},inplace=True)
bike_sharing


# In[12]:


statu=pd.get_dummies(bike_sharing['weekday'])
statu.head()


# In[13]:


statu=pd.get_dummies(bike_sharing['weekday'],drop_first=True)
statu.head()


# In[14]:


bike_sharing=pd.concat([bike_sharing,statu],axis=1)
bike_sharing.head()


# In[15]:


bike_sharing.rename(columns={1:'tue',2:'wed',3:'thu',4:'fri',5:'sat',6:'sun'},inplace=True)
bike_sharing


# In[16]:


statu=pd.get_dummies(bike_sharing['weathersit'])
statu.head()


# In[17]:


statu=pd.get_dummies(bike_sharing['weathersit'],drop_first=True)
statu.head()


# In[18]:


bike_sharing=pd.concat([bike_sharing,statu],axis=1)
bike_sharing.head()


# In[19]:


bike_sharing.rename(columns={2:'nor',3:'hot'},inplace=True)
bike_sharing


# In[20]:


bike_sharing.drop(['season','weathersit'],axis=1,inplace=True)
bike_sharing.drop(['instant','dteday','mnth','weekday'],axis=1,inplace=True)
bike_sharing


# In[21]:


from sklearn.model_selection import train_test_split
df_train,df_test=train_test_split(bike_sharing,train_size=0.7,test_size=0.3,random_state=100)
print(df_train.shape)
print(df_test.shape)


# In[22]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

num_vars=['temp','atemp','hum','windspeed','casual','registered','cnt']

df_train[num_vars]=scaler.fit_transform(df_train[num_vars])

df_train.head()


# In[23]:


y_train=df_train.pop('cnt')
X_train=df_train


# In[24]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import sklearn as sl


# In[25]:


lm=LinearRegression()
lm.fit(X_train,y_train)
rfe=RFE(lm,step=2)
rfe=rfe.fit(X_train,y_train)


# In[26]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[27]:


col=X_train.columns[rfe.support_]
col


# In[28]:


X_train.columns[~rfe.support_]


# In[29]:


X_train_rfe=X_train[col]


# In[30]:


import statsmodels.api as sm
X_train_rfe=sm.add_constant(X_train_rfe)


# In[31]:


lm=sm.OLS(y_train,X_train_rfe).fit()


# In[32]:


print(lm.summary())


# In[33]:


X_train_rfe=X_train_rfe.drop(['const'],axis=1)


# In[34]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[35]:


vif=pd.DataFrame()
X=X_train_rfe
vif['Features']=X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[36]:


X_train_ac=X_train_rfe.drop(["atemp"],axis=1)


# In[37]:


import statsmodels.api as sm
X_train_lm=sm.add_constant(X_train_ac)


# In[38]:


lm=sm.OLS(y_train,X_train_lm).fit()


# In[39]:


print(lm.summary())


# In[40]:


vif=pd.DataFrame()
X=X_train_ac
vif['Features']=X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[41]:


X_train_ad=X_train_ac.drop(["oct"],axis=1)


# In[42]:


X_train_lm=sm.add_constant(X_train_ad)


# In[43]:


lm=sm.OLS(y_train,X_train_lm).fit()


# In[44]:


print(lm.summary())


# In[45]:


vif=pd.DataFrame()
X=X_train_ad
vif['Features']=X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[46]:


X_train_ae=X_train_ad.drop(["winter"],axis=1)
X_train_lm=sm.add_constant(X_train_ae)
lm=sm.OLS(y_train,X_train_lm).fit()
print(lm.summary())


# In[47]:


vif=pd.DataFrame()
X=X_train_ae
vif['Features']=X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[48]:


X_train_af=X_train_ad.drop(["temp"],axis=1)
X_train_lm=sm.add_constant(X_train_af)
lm=sm.OLS(y_train,X_train_lm).fit()
print(lm.summary())


# In[49]:


vif=pd.DataFrame()
X=X_train_af
vif['Features']=X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[50]:


X_train_ag=X_train_af.drop(["hum"],axis=1)


# In[51]:


X_train_lm=sm.add_constant(X_train_ag)


# In[52]:


lm=sm.OLS(y_train,X_train_lm).fit()


# In[53]:


print(lm.summary())


# In[54]:


vif=pd.DataFrame()
X=X_train_ag
vif['Features']=X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[55]:


X_train_ah=X_train_ag.drop(["registered"],axis=1)


# In[56]:


X_train_lm=sm.add_constant(X_train_ah)


# In[57]:


lm=sm.OLS(y_train,X_train_lm).fit()


# In[58]:


print(lm.summary())


# In[59]:


vif=pd.DataFrame()
X=X_train_ah
vif['Features']=X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[60]:


X_train_ai=X_train_ah.drop(["dec"],axis=1)


# In[61]:


X_train_lm=sm.add_constant(X_train_ai)


# In[62]:


lm=sm.OLS(y_train,X_train_lm).fit()


# In[63]:


print(lm.summary())


# In[64]:


vif=pd.DataFrame()
X=X_train_ai
vif['Features']=X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[65]:


X_train_aj=X_train_ai.drop(["winter"],axis=1)


# In[66]:


X_train_lm=sm.add_constant(X_train_aj)


# In[67]:


lm=sm.OLS(y_train,X_train_lm).fit()


# In[68]:


print(lm.summary())


# In[69]:


vif=pd.DataFrame()
X=X_train_aj
vif['Features']=X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[70]:


X_train_ak=X_train_aj.drop(["feb"],axis=1)


# In[71]:


X_train_lm=sm.add_constant(X_train_ak)


# In[72]:


lm=sm.OLS(y_train,X_train_lm).fit()


# In[73]:


print(lm.summary())


# In[74]:


vif=pd.DataFrame()
X=X_train_ak
vif['Features']=X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[75]:


y_train_cnt=lm.predict(X_train_lm)


# In[76]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[77]:


fig=plt.figure()
sns.distplot((y_train-y_train_cnt),bins=20)
fig.suptitle('Error Terms',fontsize=20)
plt.xlabel('Errors',fontsize=18)


# In[78]:


num_vars=['temp','atemp','hum','windspeed','casual','registered','cnt']
df_test[num_vars]=scaler.transform(df_test[num_vars])


# In[81]:


y_test=df_test.pop('cnt')
X_test=df_test


# In[83]:


X_test_new=X_test[X_train_ak.columns]


# In[84]:


X_test_new=sm.add_constant(X_test_new)


# In[85]:


y_pred=lm.predict(X_test_new)


# In[87]:


fig=plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred',fontsize=20)
plt.xlabel('y_test',fontsize=18)
plt.ylabel('y_pred',fontsize=16)


# In[ ]:




