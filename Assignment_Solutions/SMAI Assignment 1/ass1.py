#!/usr/bin/env python
# coding: utf-8

# In[89]:


# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import numpy as np
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data = pd.read_csv("train.csv") 
# Preview the first 5 lines of the loaded data 
data.head()


# In[90]:


data.at[4,'number_project']
data.iat[4,2]


# In[91]:


len(data.columns)


# In[92]:


msk = np.random.rand(len(data)) < 0.8
train = data[msk]
val = data[~msk]
print(len(train))
print(len(val))
train.head()


# In[93]:


val.head()


# In[179]:


train = train.loc[:,['Work_accident', 'promotion_last_5years', 'department', 'salary','left'] ] 
print(train.head())
val = val.loc[:,['Work_accident', 'promotion_last_5years', 'department', 'salary','left'] ] 
print(val.head())


# In[95]:


#class DecisionTreeForDTClassifier:
#   def __init__(self,)
depAttValue = train["department"].unique()
print(depAttValue)
len(depAttValue)


# In[186]:


#Implement the solution using scikit learn algorithm
numberOfNodes = 0


# In[96]:


class Node(object):
    def __init__(self, label,name):
        self.name = name
        self.label = label
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)
        
    def __str__(self):
        return self.label +" "+ self.name


# In[187]:


targetAttribute = 'left'
Attributes = list(train.columns)
Attributes.remove(targetAttribute)
AllAttrExpTar = Attributes[:]
print(Attributes)
examples = list(train.index.values) 
len(examples)


# In[ ]:





# In[191]:


def getEntropyofAttribute(currentAttribute,examplesForCurAttr,targetAttribute):
    NumberOfexamplesForCurAttr = len(examplesForCurAttr)
    sum = 0.0
    fra1 = train.loc[examplesForCurAttr,[currentAttribute] ]
    currentAttributeValues = fra1[currentAttribute].unique()
    for i in currentAttributeValues:
        NumberOfPosForThisValue = 0
        NumberOfNegForThisValue = 0
        for j in examplesForCurAttr:
            if  train.at[int(j),targetAttribute] == 1:
                NumberOfPosForThisValue = NumberOfPosForThisValue+1
            else:
                NumberOfNegForThisValue = NumberOfNegForThisValue+1
        PropOfPos = NumberOfPosForThisValue/NumberOfexamplesForCurAttr
        PropOfNeg = NumberOfNegForThisValue/NumberOfexamplesForCurAttr
        sum = sum + (-1*PropOfPos*(np.log(PropOfPos)/np.log(2))) + (-1*PropOfNeg*(np.log(PropOfNeg)/np.log(2)))
    return sum


# In[99]:


(np.log(4)/np.log(2))


# In[120]:


def BestOfAttributes(Attributes,examplesForCurAttr,targetAttribute,method):
    if method == 'Entropy':
        EntropyForAttributes = {}
        for i in Attributes:
            EntropyForAttributes[i] = getEntropyofAttribute(i,examplesForCurAttr,targetAttribute)
        #findining the minimum among entropy of attributes
        BestOfAttributes = min(EntropyForAttributes, key=EntropyForAttributes.get)
        return BestOfAttributes        
    elif method == 'Gini':
        pass
    elif method == 'misClasifi':
        pass
    return BestOfAttributes


# In[167]:


train.at[2,'left']


# In[193]:


def DecisionTreeBuilder(examplesForCurAttr,targetAttribute,Attributes):
    # if all examplesForCurAttr are positive(i.e employee left the company,left =1 )
    global numberOfNodes
    allExPos = True
    for i in examplesForCurAttr:
        if train.at[i,targetAttribute] == 1:
            pass
        else:
            allExPos = False
            break
    if allExPos:
        r = Node('Positive','leaf Node')
        numberOfNodes  = numberOfNodes + 1
        print("********************************************************************")
        print("The label of current node is")
        print(r)
        print("number of examplesForCurAttr of this attribute is ")
        print(len(examplesForCurAttr))
        #print(examplesForCurAttr)
        print("Type1")
        print("********************************************************************")    
        return r
    
    # if all examplesForCurAttr are negative(i.e employee didnot leave the company,left =0 )
    allExNeg = True
    for j in examplesForCurAttr:
        if (train.at[j,targetAttribute] == 0):
            pass
        else:
            allExNeg = False
            break
    if allExNeg:
        r = Node('Negative','leaf Node')
        numberOfNodes  = numberOfNodes + 1
        print("********************************************************************")
        print("The label of current node is")
        print(r)
        print("number of examplesForCurAttr of this attribute is ")
        print(len(examplesForCurAttr))
        #print(examplesForCurAttr)
        print("Type2")
        print("********************************************************************")
        return r
    
    #if attributes are empty , then most common value of target Attribute in examplesForCurAttr
    if len(Attributes) == 0:
    #if len(examplesForCurAttr)< 100: 
        fra1 = train.loc[examplesForCurAttr,[targetAttribute] ]
        targetAttributesList = fra1[targetAttribute].values
        unique1, counts1 = np.unique(targetAttributesList, return_counts=True)
        dictOfTaAttr = dict(zip(unique1, counts1))
        posCount = dictOfTaAttr[1]
        negCount = dictOfTaAttr[0]
        if posCount>negCount:
            r = Node('Positive','leaf Node')
            numberOfNodes  = numberOfNodes + 1
        else:
            r = Node('Negative','leaf Node')
            numberOfNodes  = numberOfNodes + 1
        print("********************************************************************")
        print("The label of current node is")
        print(r)
        print("number of examplesForCurAttr of this attribute is ")
        print(len(examplesForCurAttr))
        #print(examplesForCurAttr)
        print("Type3")
        print("********************************************************************")
        return r
    #if len(Attributes) == 0:
        #Attributes = Attributes + AllAttrExpTar
    
    BestAttribute = BestOfAttributes(Attributes,examplesForCurAttr,targetAttribute,'Entropy')
    Attributes.remove(BestAttribute)    
    r = Node(BestAttribute,'internalNode')
    numberOfNodes  = numberOfNodes + 1
    BestAttributeValues = train[BestAttribute].unique()
    for iter in BestAttributeValues:
        examplesForCurAttrForThisChild = []
        for iter1 in examplesForCurAttr:
            if  train.at[int(iter1),str(BestAttribute)] == iter:
                examplesForCurAttrForThisChild.append(int(iter1)) 
        r.add_child(DecisionTreeBuilder(examplesForCurAttrForThisChild,targetAttribute,Attributes))
    print("********************************************************************")
    print("The label of current node is")
    print(r)
    print("the best decision attribute of current node is")
    print(BestAttribute)
    print("number of examplesForCurAttr of this attribute is ")
    print(len(examplesForCurAttr))
    #print(examplesForCurAttr)
    print("children of current nodes")
    plabo = range(0,len(r.children))
    for k in plabo:
        print(r.children[k])
    print("Type4")
    print("********************************************************************")
            
    return r


# In[194]:


RootNode = DecisionTreeBuilder(examples,'left',Attributes) 
numberOfNodes

