#-*-coding:utf-8-*-

from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def createDataset():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

'''
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sumDistance = sqDiffMat.sum(axis=1)
    distances = sumDistance ** 0.5
    sortDistance = distances.argsort()

    classCount = {}
    for i in range(k):
        voteLabel = labels[sortDistance[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
    
'''
def KNN(input,dataset,label,k):
    # 求出数据集的行数
    datasetSize = dataset.shape[0]
    # 求输入数据到每个数据的距离
    diffMat = tile(input,(datasetSize,1)) - dataset
    # 欧式距离
    sqDiffMat = diffMat ** 2
    sumSqDiffMat = sqDiffMat.sum(axis=1)
    distances = sumSqDiffMat ** 0.5
    sortDistance = distances.argsort()

    classCount = {}

    for i in range(k):
        # 选择标签
        voteLabel = label[sortDistance[i]]
        # 标签计数
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key = lambda x:x[1],reverse=True)
    return sortedClassCount[0][0]

# 加载文件并处理成矩阵格式
def file2matrix(filename):
    # 打开文件
    fr = open(filename)
    # 读取文件行数
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)
    matrix = zeros((numberOfLines,3))
    # 类别标签
    classLabel = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFormatLine = line.split("\t")
        matrix[index, :] = listFormatLine[0:3]
        classLabel.append(int(listFormatLine[-1]))
        index += 1
    return matrix,classLabel

#数据归一化
def NormData(dataSet):
    # 求出数据集中每一列的最小值
    minValue = dataSet.min(0)
    # 求出数据集中每一列的最大值
    maxValue = dataSet.max(0)
    # 求出列数
    m = dataSet.shape[0]
    # 求出范围
    ranges = maxValue - minValue

    NormMat = zeros(dataSet.shape)

    # 归一化
    NormMat = dataSet - tile(minValue,(m,1))
    NormMat = NormMat/tile(ranges,(m,1))

    return NormMat


#KNN的测试
def TestingKNN():
    # 测试百分比
    hoRatio = 0.1

    matrix,classLabel = file2matrix("datingTestSet.txt")

    normMat = NormData(matrix)

    m = NormMat.shape[0]

    testNum = int(m * hoRatio)

    errorCount = 0
    for i in range(testNum):
        classify = KNN(normMat[i,:],normMat[testNum:m,:],classLabel[testNum:m],3)
        print "the classifier came back with: %d,the real answer is: %d" % (classify,classLabel[i])
        if(classify != classLabel[i]):
            errorCount += 1
    print "the total error rate is :%f" %(errorCount/float(testNum))

#KNN预测
def PredPerson():
    resultList = ['not at all','in small doses','in large doses']

    percetageRate = float(raw_input("percentage of time spent playing video games?"))
    ffMile = float(raw_input("frequent fliter miles earns per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))

    datingMat,datingLabel = file2matrix("datingTestSet.txt")
    normMat = NormData(datingMat)

    arr = array([percetageRate,ffMile,iceCream])

    classResult = KNN(arr,normMat,datingLabel,3)

    print "You will probably like this person:",resultList[classResult-1]







if __name__ == "__main__":

    group,labels = createDataset()
    # label = classify0([0,0],group,labels,3)

    label = KNN([0,0],group,labels,3)
    matrix, classLabel = file2matrix("datingTestSet.txt")

    NormMat = NormData(matrix)
    PredPerson()
    # TestingKNN()
    # print NormMat

    # normData,ranges,minVals = autoNorm(matrix)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(matrix[:,1],matrix[:,2],15.0 * array(classLabel),15.0 * array(classLabel))
    #
    # plt.show()
