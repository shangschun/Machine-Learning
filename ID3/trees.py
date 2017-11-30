#-*-coding:utf-8-*-

from math import log
import operator

# 创建数据集
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

# 计算香农熵
def calcShannonEnt(dataSet):

    numEntries = len(dataSet)

    numLabel = {}

    for key in dataSet:
        feture = key[-1]
        if feture not in numLabel.keys():
            numLabel[feture] = 0

        numLabel[feture] += 1
    shannonEnt = 0.0
    for key in numLabel:
        prob = float(numLabel[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

# 划分数据集
def splitDataSet(dataSet,axis,value):
    splitData = []

    for data in dataSet:
        dataSplit = []
        if data[axis] == value:
            dataSplit= data[:axis]
            dataSplit.extend(data[axis+1:])
            splitData.append(dataSplit)
    return splitData

# 计算信息增益
def chooseBestFeture(dataSet):
    numFeture = len(dataSet[0]) - 1
    baseEntire = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeture = -1

    for i in range(numFeture):
        FeatureList = [example[i] for example in dataSet]
        uniFeture = set(FeatureList)
        newEntrie = 0.0
        for values in uniFeture:
            newDataSet = splitDataSet(dataSet,i,values)
            prob = len(newDataSet)/float(len(dataSet))
            newEntrie += prob * calcShannonEnt(newDataSet)
        infoGain = baseEntire - newEntrie
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeture = i
    return bestFeture

# 多数投票
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeture(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabel = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabel)
    return myTree







if __name__ == "__main__":
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age','prescript','astigmatic','tearRate']
    myTree = createTree(lenses,lensesLabels)
    print myTree

