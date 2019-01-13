
#%%
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#%% [markdown]
# ## User-based Collaborative Filtering
# 
# - Input: Original data with missing rating
# 
# - Step1: Calculate the USER average for all other ITEMS
# 
# - Step2: Centralise each row for all other ITEMS
# 
# - Step3: Calculate the 2-norm of each USER with all other ITEMS
# 
# - Step4: Compute the Pearson correlation coefficient between USERS
# 
# - Output: The predicted rating as the sum of average rating of the target USER and the centralised rating on the target ITEM of the most similar USER

#%%
# input: original data with missing rating

# import user scores csv
# set index to column 0
user_scores = pd.read_csv('user_scores.csv', index_col=0)

print("\nThese are the original user scores: \n")
user_scores


#%%
# calculate the USER average for all other ITEMS

newvariable430 = user_scores['Avg 1-4']
newvariable430=user_scores[['Item 1', 'Item 2', 'Item 3', 'Item 4']].sum(axis=1)/4

print("\nThese are the original user scores w/ averages for items 1-4 included: \n")
user_scores


#%%
# centralise each row for all other ITEMS

cent_scores=user_scores.loc[:,'Item 1':'Item 4']

cent_scores['Item 1']=user_scores['Item 1']-newvariable430
cent_scores['Item 2']=user_scores['Item 2']-newvariable430
cent_scores['Item 3']=user_scores['Item 3']-newvariable430
cent_scores['Item 4']=user_scores['Item 4']-newvariable430

print("\nThese are the centralised scores: \n")
cent_scores


#%%
# calculate the 2-norm of each USER with all other ITEMS

cent_scores['2-norm']=np.linalg.norm(cent_scores[['Item 1','Item 2','Item 3','Item 4']].values,axis=1)

cent_scores


#%%
# compute the Pearson correlation coefficient between USERS

"""
the sum of item-wise products divided by 2-norm of user x and further divided by 2-norm of user 1
"""

headers=['Pearson', 'Max?', 'U1 Avg', 'Prediction']
rows=['User 2', 'User 3', 'User 4', 'User 5']
pearson = pd.DataFrame(index=rows,columns=headers)

pearson


#%%
# output: the predicted rating as the sum of average rating of the target USER and the centralised rating on the...
# target ITEM of the most similar USER

#%% [markdown]
# ## Item-based Collaborative Filtering
# 
# - Input: Original data with missing rating
# - Step1: Calculate the USER average for all other USERS
# - Step2: Centralise each column for all other USERS
# - Step3: Calculate the 2-norm of each ITEM
# - Step4: Compute the adjusted cosine similarity between ITEMS
# - Output: The predicted rating based on the rating of the ITEM with maximal similarity

#%%



