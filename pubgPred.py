#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.filterwarnings('ignore')
import seaborn as sns
import random
random.seed(42)


# In[56]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import gc
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor


# In[3]:


def reduce_mem_usage(df):
    import numpy as np

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    return df


# In[4]:


df_train = pd.read_csv('train_V2.csv')
df_train.info()


# In[5]:


df_train = reduce_mem_usage(df_train)


# In[6]:


df_train.info()


# In[7]:


pd.set_option('display.max_columns', 500)


# In[8]:


df_train.head()


# In[9]:


df_test = pd.read_csv('test_V2.csv')


# In[11]:


df_test = reduce_mem_usage(df_test)


# In[12]:


#Removing NAN values from the dataset
df_train.isnull().sum()


# In[13]:


df_train.dropna(inplace=True)


# In[14]:


df_train.isnull().sum()


# In[16]:


print(len(df_train))
print(len(df_test))


# # visualizations and Removing Outliers

# In[17]:


def visualization (col, num_bin=10):
    title = col[0].upper() + col[1:]
    f,axes=plt.subplots()
    plt.xlabel(title)
    plt.ylabel('Log Count')
    axes.set_yscale('log')
    df_train.hist(column=col,ax=axes,bins=num_bin)
    plt.title('Histogram of ' + title)
    plt.show()
    
    tmp = df_train[col].value_counts().sort_values(ascending=False)

    print('Min value of ' + title + ' is: ',min(tmp.index))
    print('Max value of ' + title + ' is: ',max(tmp.index))


# In[21]:


visualization('assists')


# In[22]:


visualization('roadKills')


# In[18]:


# # since, most of the players have kills from  0 to 10 
# so to remove the outliers from my data, we drop all the players who have more than 10 roadkills.
 # drop all the road kills above 10.


#test

df_train.drop(df_train[df_train['roadKills']>=10].index,inplace=True)

#test 

df_test.drop(df_test[df_test['roadKills']>=10].index,inplace=True)


# In[57]:


visualization('kills')


# In[19]:


# dropping the outliers.

#train
df_train.drop(df_train[df_train['kills']>=35].index,inplace=True)

#test

df_test.drop(df_test[df_test['kills']>=35].index,inplace=True)


# In[23]:


visualization('killStreaks')


# In[32]:


visualization('teamKills')


# In[33]:


visualization('headshotKills', num_bin=40)


# In[34]:


visualization('vehicleDestroys',num_bin=5)


# In[35]:


visualization('revives',num_bin=50) 


# In[38]:


visualization('damageDealt', num_bin=1000)


# In[39]:


visualization('weaponsAcquired',num_bin=30)


# In[20]:


# removing the outliers.

#train
df_train.drop(df_train[df_train.weaponsAcquired>=50].index,inplace=True)

#test
df_test.drop(df_test[df_test.weaponsAcquired>=50].index,inplace=True)


# In[40]:


visualization('boosts',num_bin=30)


# In[41]:


visualization('heals', num_bin=100)


# In[21]:


# removing the outliers.

#train
df_train.drop(df_train[df_train.heals>=40].index,inplace=True)

#test

df_test.drop(df_test[df_test.heals>=40].index,inplace=True)


# In[42]:


visualization('walkDistance',num_bin=250)


# In[22]:


#Removing the outliers

#train
df_train.drop(df_train[df_train['walkDistance']>=10000].index,inplace=True)


#test
df_test.drop(df_test[df_test['walkDistance']>=10000].index,inplace=True)


# In[43]:


visualization('rideDistance',num_bin=500)


# In[23]:


#Removing the outliers.

#test

df_train.drop(df_train[df_train.rideDistance >=15000].index, inplace=True)

#test

df_test.drop(df_test[df_test.rideDistance >=15000].index, inplace=True)


# In[49]:


visualization('longestKill', num_bin=100) 


#  Most kills are made from a distance of 100 meters or closer. There are however some outliers who make a kill from more than 1km away. This is probably done by cheaters or game crackers. 

# In[24]:


# drop outliers.      


#train
df_train.drop(df_train[df_train['longestKill']>=1000].index,inplace=True)

#test

df_test.drop(df_test[df_test['longestKill']>=1000].index,inplace=True)


# In[25]:


df_train.shape


#  So the initial shape is (4446965, 29)And After removing the outliers the new shape is  (4445866, -) 
#  Something around 1100 rows have been removed until now. Which is nothing compared to the number of rows we have.

# In[66]:


#So the initial shape is (4446965, 29)And After removing the outliers the new shape is (4445866, -) 
#Something around 1100 rows have been removed until now. Which is nothing compared to the number of rows we have.


# In[26]:


# Creating a dummy variable for categorical variable present in our data set.
#matchType
#train

df_train=pd.get_dummies(df_train,columns=['matchType'])

#test

df_test=pd.get_dummies(df_test,columns=['matchType'])


# In[30]:


#Correlation Analysis
cols_to_drop = ['Id','matchId','groupId','matchType']
cols_to_fit = [col for col in df_train.columns if col not in cols_to_drop]
corr = df_train[cols_to_fit].corr()


# In[31]:


plt.figure(figsize=(9,7))
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values,linecolor='white',linewidths=0.1,cmap='RdBu')
plt.show()


# In[32]:


t = df_train
t =t.drop(['Id','groupId','matchId',],axis=1)
y = t['winPlacePerc']
X = t.drop(['winPlacePerc'],axis=1)
X_test = df_test.drop(['Id','groupId','matchId'],axis=1)


# In[34]:


#splitting the data into training and testing by using train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7)
len(X_train)

del train,test,X,y

#X_train = X
#y_train = y
gc.collect()


# # Model LightGBM

# In[38]:


import lightgbm as lgbm


# In[36]:


def calculate_error(cl,name):
  print(name)
  y_pre = cl.predict(X_val)
  print('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(y_val,y_pre)))
  print('R2 score is {:.2%}'.format(r2_score(y_val, cl.predict(X_val))))


# In[42]:


# Create parameters to search
params = {"objective" : "regression", "metric" : "mae", 'n_estimators':20000, 
              'early_stopping_rounds':200, "num_leaves" : 31, "learning_rate" : 0.05, 
              "bagging_fraction" : 0.7, "bagging_seed" : 0, "num_threads" : 4,
              "colsample_bytree" : 0.7
             }
lgbTrain = lgbm.Dataset(X_train, label=y_train)
lgbVal = lgbm.Dataset(X_val, label=y_val)
model = lgbm.train(params,lgbTrain,valid_sets=[lgbTrain, lgbVal],
                      early_stopping_rounds=200, verbose_eval=1000)


# In[45]:


calculate_error(model,"LGBM")


# In[47]:


y_predict = model.predict(X_test)


# In[52]:


y_predict[y_predict > 1] = 1
y_predict[y_predict < 0] = 0
df_test['winPlacePerc'] = y_predict
submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('Finalsubmission.csv', index=False)


# In[55]:


#find which feature importance
cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
cols_to_fit = [col for col in X_train.columns if col not in cols_to_drop]
feature_importance = pd.DataFrame(sorted(zip(model.feature_importance(), cols_to_fit)), columns=['Value','Feature'])
feature_importance = feature_importance.tail(15)


plt.figure(figsize=(10, 6))
sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig("lgbmfeatures.png",dpi=500)


# # Regression Models

# In[57]:


def runAllModels(X_train, Y_train):
        
    linear = LinearRegression(copy_X=True)
    linear.fit(X_train,Y_train)
    calculate_error(linear,"linear")

    ridge = Ridge(copy_X=True)
    ridge.fit(X_train,Y_train)
    calculate_error(ridge,"ridge")
    
    lasso = Lasso(copy_X=True)
    lasso.fit(X_train,Y_train)
    calculate_error(lasso,"lasso")
    
    elastic = ElasticNet(copy_X=True)
    elastic.fit(X_train,Y_train)
    calculate_error(elastic,"elastic")
    
    ada = AdaBoostRegressor(learning_rate=0.8)
    ada.fit(X_train,Y_train)
    calculate_error(ada,"Adaboost")
    
    GBR = GradientBoostingRegressor(learning_rate=0.8)
    GBR.fit(X_train,Y_train)
    calculate_error(GBR,"GBR")

    forest = RandomForestRegressor(n_estimators=10)
    forest.fit(X_train,Y_train)
    calculate_error(forest,"forest")
    
    tree = DecisionTreeRegressor()
    tree.fit(X_train,Y_train)
    calculate_error(tree,"tree")


# In[58]:


runAllModels(X_train,y_train)


# In[ ]:




