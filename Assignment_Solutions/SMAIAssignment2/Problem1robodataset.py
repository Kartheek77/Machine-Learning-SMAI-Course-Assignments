import pandas as pd
import numpy as np
robo1data = pd.read_csv('Robot1',delim_whitespace=True,index_col=False)
ran = np.random.rand(len(robo1data)) < 0.8
train = robo1data[ran]
val = robo1data[~ran]
train.to_csv('UsedTrainingData.csv')
val.to_csv('UsedValidationData.csv')
#train.head()
#val.head()
#print(len(train))
#print(len(val))

def NumberOfEachClass(GivenData):
    NumberOfClassZerosInGivenData = 0
    NumberOfClassOnesInGivenData = 0
    for i in list(GivenData.index.values):
        if GivenData.at[int(i),'class'] == 0:
            NumberOfClassZerosInGivenData += 1
        else:
            NumberOfClassOnesInGivenData += 1
    print('NumberOfClassZerosInGivenData '+str(NumberOfClassZerosInGivenData))
    print('NumberOfClassOnesInGivenData  '+str(NumberOfClassOnesInGivenData))
    print('Class Zeros Percentage '+str(NumberOfClassZerosInGivenData/(NumberOfClassZerosInGivenData+NumberOfClassOnesInGivenData)))
    print('Class Ones Percentage '+str(NumberOfClassOnesInGivenData/(NumberOfClassZerosInGivenData+NumberOfClassOnesInGivenData)))
#NumberOfEachClass(train)
#NumberOfEachClass(val)
import math 
def  getDistBetweenPoints(Point1,Point2,method):
    #print(Point1)
    #print(Point2)
    if method == 'Euclidean':
        sum = 0
        for i in range(0,len(Point1)-1):
            sum = sum + (abs(Point1[i] - Point2[i]))**0.5 
        return sum**(1/float(0.5))
    #if method ==
def getDistanceFromAllPoints(Point,distance_method):
    DistancesFromAllOtherPoints ={}
        #for i in list(train['Id']):
            #DistancesFromAllOtherPoints{i} = 
    for index,row in train.iterrows():
        if row[-1] != Point[-1]:
            DistancesFromAllOtherPoints[row[-1]] = [getDistBetweenPoints(list(row),Point,distance_method),row[0]]
    #print(DistancesFromAllOtherPoints)
    return DistancesFromAllOtherPoints
def getLabel(Point,distance_method,NumOfNeighbours):
    NearestNeighboursAndTheirDistances = {}  
    DistancesFromAllOtherPoints = getDistanceFromAllPoints(Point,distance_method)
    #print(DistancesFromAllOtherPoints)
    NearestNeighboursAndTheirDistances = {k: DistancesFromAllOtherPoints[k] for k in list(DistancesFromAllOtherPoints)[:NumOfNeighbours]}
    for key, value in DistancesFromAllOtherPoints.items():
        for key1, value1 in NearestNeighboursAndTheirDistances.items():
            if value[0] < value1[0]:
                del NearestNeighboursAndTheirDistances[key1]
                NearestNeighboursAndTheirDistances[key] = value
                break
    NumberOfClassZero = 0
    NumberOfClassOne = 0
    
    for key, value in NearestNeighboursAndTheirDistances.items():
        if NearestNeighboursAndTheirDistances[key1][1] == 1:
            NumberOfClassOne += 1
        else:
            NumberOfClassZero += 1    
    if(NumberOfClassOne>NumberOfClassZero):
        return 1
    else:
        return 0          
def compute(GivenDataSet,distance_method,NumOfNeighbours):
    GivenDataSet["predict"] = [0]*len(val)
    for index,row in GivenDataSet.iterrows():
        GivenDataSet.at[int(index),'predict'] = getLabel(row.tolist(),distance_method,NumOfNeighbours)
    #caluculate accuracy and precision
    #return GivenDataSet
def CalulatePrecsionRecallEtc(GivenTestData):
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    for i in list(GivenTestData.index.values):
        if str(GivenTestData.at[i,'class']) == str(1) and str(GivenTestData.at[i,'predict']) == str(1):
            TP = TP + 1
        elif str(GivenTestData.at[i,'class']) == str(1) and str(GivenTestData.at[i,'predict']) == str(0):
            FN = FN + 1
        elif str(GivenTestData.at[i,'class']) == str(0) and str(GivenTestData.at[i,'predict']) == str(1):
            FP = FP + 1
        elif str(GivenTestData.at[i,'class']) == str(0) and str(GivenTestData.at[i,'predict']) == str(0):
            TN = TN + 1
    print("Results On GivenTestData")
    print("-------------------------------------------------")
    print("True Positive are    "+ str(TP))
    print("True Negatives are   "+ str(TN))
    print("False Positive are   "+ str(FP))
    print("False Negatives are  "+str(FN))
    Precision = (TP/(TP+FP))
    Recall = (TP/(TP+FN))
    F1_Score = 2*((1/Recall)+(1/Precision))
    accuracy = (TN+TP)/(TN+TP+FP+FN)
    print("Precsion is          "+str(Precision))
    print("Recall is            "+str(Recall))
    print("F1_Score is          "+str(F1_Score))
    print("Accuracy is          "+str(accuracy))
    print("-------------------------------------------------")

NumOfNeighbours = 10
distance_method = 'Euclidean'
GivenDataSet = val.copy()
compute(GivenDataSet,distance_method,NumOfNeighbours)
CalulatePrecsionRecallEtc(GivenDataSet)
GivenDataSet
