#!/usr/bin/env python
# coding: utf-8

# In[31]:


import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



# In[33]:


X_train = pd.read_csv("train.genotype.txt", sep = " ", header=None).values
X_test = pd.read_csv("test.genotype.txt", sep = " ", header=None).values

Y_train = pd.read_csv("train.phenotype.txt", sep = " ", header=None).values
 


X_train_df = pd.DataFrame(X_train)
Y_train_df = pd.DataFrame(Y_train)
Y_train_df = np.array(Y_train_df.iloc[:, 0])

#df.to_csv('output.csv', index=False)  # Use index=False to exclude row numbers in the output

#print(X_train_df)

X_train_df = X_train_df.values


# In[34]:


#need to somehow reduce dimensionality of non-causal snps, becuase this dataset is too large. 
print('Training data shape : ', X_train.shape)

print('Testing data shape : ', X_test.shape)

#right now we have 200 different snps 

#MUST do feature selection. 

plt.hist(Y_train, bins = 50) #normally distributed, try to do linear regression maybe OLS. 


# In[35]:


import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression

#tried this with threshold = 5,2 and 10 and the best one was this one where it's 3. 

k = 200  # select all the features 
threshold = 3 #set a threshold to account for features that are actually useful.  
f_selector = SelectKBest(f_regression, k=k)
X_kbest = f_selector.fit_transform(X_train, Y_train)

# Retrieve the indices of selected features and their scores
selected_indices = f_selector.get_support(indices=True)
feature_scores = f_selector.scores_[selected_indices]

# Filter selected indices and scores based on the threshold
indices_above_threshold = [index for index, score in zip(selected_indices, feature_scores) if score > threshold]
scores_above_threshold = [score for score in feature_scores if score > threshold]

#plot  
plt.figure(figsize=(10, 6))
plt.bar(range(len(indices_above_threshold)), scores_above_threshold)
plt.xlabel('Selected Feature Indices (Score > 3)')
plt.ylabel('Feature Score')
plt.title('Scores of Selected Features above Threshold from SelectKBest (f_regression)')
plt.xticks(range(len(indices_above_threshold)), indices_above_threshold)
plt.show()



# In[36]:


print(indices_above_threshold)
X_train_feature_selection = X_train_df[:,indices_above_threshold]
print(X_train_feature_selection)
      
#these are the selected features. 


# In[37]:


#try OLS


#load data and do whatever preprocessing stuff that is probably unnecessary and redundant 
X_test_final = pd.read_csv("test.genotype.txt", sep = " ", header=None).values
X_test_df_final = pd.DataFrame(X_test_final)

#choose the features from the feature selection thing above in the training dataset.
X_train_feature_selection = X_train_df[:,indices_above_threshold]

#choose the features from the feature selection thing for the test dataset. 
X_test_df_final = X_test_df_final.values[:,indices_above_threshold]


# In[38]:


#ask about submitting this score because I accidentally messed up. 

X_train, X_test, y_train, y_test = train_test_split(X_train_feature_selection, Y_train, test_size=0.2, random_state=42)

# Fit an Ordinary Least Squares (OLS) model

ols_model = sm.OLS(y_train, X_train)
ols_results = ols_model.fit()

# Predict on the validation set
y_pred = ols_results.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")



#predict on unseen data (test set)
ols_pred = ols_results.predict(X_test_df_final)

#save to dataframe 
pd.DataFrame(ols_pred).to_csv(f"predictions.csv", sep = " ", header = None, index = None)
os.system("zip -r predictions.zip predictions.csv")


# In[ ]:




