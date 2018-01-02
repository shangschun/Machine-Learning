# -*- coding: utf-8 -*- 2
from numpy import *


def loadDataSet(filename,delim='\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

# dataMat:进行PCA操作的数据集
# topNfeat:可选参数，即应用的N个特征
def pca(dataMat,topNfeat=9999999):
    # 求每个特征的均值
    meanVals = mean(dataMat,axis=0)
    # 去均值
    meanRemoved = dataMat - meanVals
    # 协方差矩阵
    covMat = cov(meanRemoved,rowvar = 0)
    # 特征值，特征向量
    eigVals,eigVects = linalg.eig(mat(covMat))
    # 特征值从小到大排列，返回索引位置
    eigValInd = argsort(eigVals)
    # 取最大的topNfeat个特征值位置的index
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    # 特征值所对应的特征向量
    redEigVects = eigVects[:,eigValInd]
    # 去均值数据，转化到新的低维空间
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat,reconMat

def pcaPlot(dataMat,recoMat):
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
    ax.scatter(recoMat[:,0].flatten().A[0],recoMat[:,1].flatten().A[0],marker='o',s=50,c='red')
    plt.show()

def replaceNanWithMean():
    datMat = loadDataSet('secom.data',' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        # 计算每列非NaN值的平均值
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])
        # 将所有NaN替换为该平均值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
    return datMat

def secomPlot():
    datMat = replaceNanWithMean()
    meanVals = mean(datMat,axis=0)
    meanRemoved = datMat - meanVals
    # 协方差矩阵
    covMat = cov(meanRemoved,rowvar=0)
    # 特征值，特征向量
    eigVals,eigVects = linalg.eig(mat(covMat))
    # 特征值从小到大排序
    eigValInd = argsort(eigVals)
    # 特征值从大到小排序
    eigValInd = eigValInd[::-1]
    sortedEigVals = eigVals[eigValInd]
    # 所有特征值求和，即总方差
    total = sum(sortedEigVals)
    # 各个方差百分比
    varPercentage = sortedEigVals/total*100
    # 累计能量求和
    accEigvals = zeros((20,1))
    for i in range(0,20):
        accEigvals[i] = sum(varPercentage[:(i+1)])
    # 绘制图像
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = 'SimHei'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1,21),varPercentage[:20],marker='^')
    ax.plot(range(1,21),accEigvals)
    plt.xlabel(u'主成分编号')
    plt.ylabel((u'方差比例'))
    plt.show()


if __name__=="__main__":
    # dataMat = loadDataSet('testSet.txt')
    # lowDMat,reconMat = pca(dataMat,1)
    # pcaPlot(dataMat,reconMat)
    # print shape(lowDMat)
    # PCA半导体制造数据降维
    secomPlot()
