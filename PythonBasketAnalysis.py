
# coding: utf-8

# Association rules are normally written like this: {Diapers} -> {Beer} which means that there is a strong relationship between customers that purchased diapers and also purchased beer in the same transaction.
# 
# In the above example, the {Diaper} is the **antecedent** and the {Beer} is the **consequent**. Both antecedents and consequents can have multiple items. In other words, {Diaper, Gum} -> {Beer, Chips} is a valid rule.
# 
# **Support** is the relative frequency that the rules show up. In many instances, you may want to look for high support in order to make sure it is a useful relationship. However, there may be instances where a low support is useful if you are trying to find “hidden” relationships.
# 
# **Confidence** is a measure of the reliability of the rule. A confidence of .5 in the above example would mean that in 50% of the cases where Diaper and Gum were purchased, the purchase also included Beer and Chips. For product recommendation, a 50% confidence may be perfectly acceptable but in a medical situation, this level may not be high enough.
# 
# **Lift** is the ratio of the observed support to that expected if the two rules were independent (see wikipedia). The basic rule of thumb is that a lift value close to 1 means the rules were completely independent. Lift values > 1 are generally more “interesting” and could be indicative of a useful rule pattern.

# In[1]:

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv('Transaction_SKU_Test.csv')
df.head(10)


# In[2]:

pd.pivot_table(df, index = ['SKU'], aggfunc=np.sum).head()


# In[3]:

df['SKU'] = df['SKU'].astype('str')


# In[4]:

basket = (df.groupby(['Name', 'SKU'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('Name'))


# In[5]:

basket.head()


# In[6]:

# Show a subset of columns
basket.iloc[:,[0,1,2,3,4,5,6,7]].head()


# In[7]:

# Convert the units to 1 hot encoded values
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets.head()


# In[8]:

# Build up the frequent items
frequent_itemsets = apriori(basket_sets, min_support=0.001, use_colnames=True)


# In[13]:

frequent_itemsets.sort_values('support', ascending=True).tail()


# In[10]:

# Create the rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
rules


# In[16]:

writer = pd.ExcelWriter("Association Analysis Result.xlsx", engine='xlsxwriter')
rules.to_excel(writer, sheet_name="Sheet1")
writer.save()


# In[ ]:



