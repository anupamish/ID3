#Team Members:
# Dhruv Bajpai - dbajpai - 6258833142
# Anupam Mishra - anupammi - 2053229568
import math
import numpy as np
import sys
from collections import Counter


def childEntropy(trainData):
    targetList = [row[-1] for row in trainData]
    l1 = list(Counter(targetList).keys())
    l2 = list(Counter(targetList).values())
    total = sum(l2)
    entropy = 0.0
    for freq in l2:
        p = float(freq/total)
        entropy += -1 *(p) * math.log(p,2)
    return entropy

#Returns best attribute name
def chooseBestAttribute(trainData,attributes):
    totalEntropy = childEntropy(trainData)
    maxGain = -sys.maxsize -1
    bestAttr = attributes[0]
    for attr in attributes[:-1]:
        allUnique = set([row[header.index(attr)] for row in trainData])
        lengths = [len([row for row in trainData if row[header.index(attr)]== d]) for d in allUnique]
        AllChildentropies = [childEntropy([row for row in trainData if row[header.index(attr)]== d]) for d in allUnique]
        tempGain = totalEntropy - sum([lengths[i]/sum(lengths) * AllChildentropies[i] for i in range(0,len(lengths))])
#        print (tempGain)
        maxGain, bestAttr = (tempGain, attr) if tempGain > maxGain else (maxGain,bestAttr)
    return bestAttr

# Returns all the unique values in a column (one attribute)
def getUniqueValues(data,col):
    return set(ele[header.index(col)] for ele in data)

# Returns the majority prediction. i.e. In this case "Yes" or "No".
def majorPrediction(a):
    return list(Counter(a).keys())[np.argmax(list(Counter(a).values()))]

# Recursively generates and returns a dictionary form of a decision tree classifier
def makeMyTree(trainData,attributes):
#     duplicating the dataset for this function call
    trainData = trainData[:]
    bestAttr = chooseBestAttribute(trainData,attributes)
    predictionVals = [row[-1] for row in trainData]
#     This is the case when all attributes have been exhausted
    defaultValue = majorPrediction(predictionVals)
#     This case applies when there are no further attributes to split on.
    if(len(attributes)<=1):
        return defaultValue
#   If the dataLeft is purely separated
    elif predictionVals.count(predictionVals[0])== len(predictionVals):
        return predictionVals[0]
    else:
#     get unique values of the attribute
        uniqueSplits = getUniqueValues(trainData,bestAttr)
        dTree = {bestAttr:{}}

        for val in uniqueSplits:
            rows = [ele for ele in trainData if ele[header.index(bestAttr)]==val]
            updatedAttributes = attributes[:]
            updatedAttributes.remove(bestAttr)
#             updatedAttributes.remove(bestAttr)
            dTree[bestAttr][val] = makeMyTree(rows,updatedAttributes)
    return dTree

def predict(treeDic, data):
    if type(treeDic)==type({}):
        key = list(treeDic.keys())[0]
#         print(list(dic.keys()))
#        print("Key: ",key)
        return predict(treeDic[key][data[header.index(key)]], data)
    else:
        return treeDic

def formatPrint(treeDic,recurse):
    if type(treeDic)==type({}):
        if recurse%2==0:
            print(" ",list(treeDic.keys())[0])
#             print(" "*recurse*2,list(treeDic.keys())[0])
            formatPrint(treeDic[list(treeDic.keys())[0]],recurse+1)
        else:
            for key in treeDic.keys():
                print(" "*recurse*2, key,":",end="")
                formatPrint(treeDic[key],recurse+1)
    else:
        print(" ",treeDic)
header = ["Occupied", "Price", "Music", "Location", "VIP", "Favorite Beer", "Enjoy"]    
def main():
    train_file = sys.argv[1]
    
# dataFile = open("/home/dhruv/Desktop/ML/training_data", "r")
    dataFile = open(train_file, "r")
    data = dataFile.read()

    trainingData = [[ele.strip(" ") for ele in row.split(", ")] for row in data.split("\n")]
    trainingData.pop(22)

    print ("---------------------TREE GENERATED-------------------")

    tree = makeMyTree(trainingData,header)
    formatPrint(tree,0)

    print ("\n---------------------OUTPUT PREDICTED-------------------")
    test = ["Moderate","Cheap","Loud","City-Center","No","No"]
    print("Predicted Output = ",predict(tree, test))

if __name__ == '__main__':
    main()
     

