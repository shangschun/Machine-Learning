# -*- coding: utf-8 -*- 2

from numpy import *
import matplotlib.pyplot as plt
import math

# 获取样本数据
def loadDataSet(filename):
    # 特征值数
    numFeat = len(open(filename).readline().split('\t')) - 1
    # 特征值
    dataMat = []
    # 类别标签
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        # 遍历每一维度
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        # 存所有行的数据的特征
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# 计算最佳拟合直线
# 最小二乘法（标准回归函数）：求拟合直线的参数w=(X.T*X).I*X.T*y
# xArr：样本特征数据
# yArr：样本目标值
# 回归系数：ws = (xTx)^(-1) * (xTy)
def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    # 判断行列式是否为0
    if linalg.det(xTx) == 0.0:
        print  "This matrix is singular,cannot do inverse"
        return
    # 拟合直线的参数
    ws = xTx.I * (xMat.T * yMat)
    return ws

# 原数据散点和拟合直线图（线性回归）
def standPlot(xArr,yArr,w):
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat * w
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * w
    ax.plot(xCopy[:, 1], yHat)
    plt.show()

# 预测值和真实值的匹配程度，可以通过计算相关系数
def correlation(xArr,yArr):
    ws = standRegres(xArr, yArr)
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat * ws
    return corrcoef(yHat.T,yMat)

# 局部加权线性回归
# 回归系数：ws = (xTwx)^(-1) * (xTwy)
# w：点的权重 一般使用高斯核
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        # 高斯核
        weights[j,j] = exp(diffMat * diffMat.T/(-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular,cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


# 局部加权线性回归散点图
def lwlrPlot(xArr,yArr,k):
    yHat = lwlrTest(xArr,xArr,yArr,k)
    xMat = mat(xArr)
    yMat = mat(yArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()


def rssError(yArr,yHatArr):
    return ((yArr -  yHatArr) ** 2).sum()

# 岭回归
# 回归系数：(w=xT*x + λI)^(-1) * (xTy)
# λ：用户自定义数据，用来限制所有w之和
# I：单位矩阵
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular,cannot do inverse"
        return
    ws = denom.T * (xMat.T * yMat)
    return ws

# 测试一组λ
# 数据标准化：对用特征减去各自的均值再除以方差
def ridgeTest(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # 对列求均值
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMeans = mean(xMat,0)
    # 求方差
    xVar = var(xMat,0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i - 10))
        wMat[i,:] = ws.T
    return wMat

# 逐步线性回归
# xArr：输入数据
# yArr：预测变量
# eps：每次迭代需要调整的步长
# numIt：迭代次数
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMat = (xMat - mean(xMat,0))/var(xMat,0)
    m,n = shape(xMat)
    returnMat = zeros((numIt,n))
    ws = zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat



if __name__ == "__main__":
    # 加载数据（散点）
    # xArr,yArr = loadDataSet("ex0.txt")

    # 预测鲍鱼年龄
    # 加载数据
    # abX,abY = loadDataSet("abalone.txt")
    # 分析预测误差
    # yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    # yHat1 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
    # yHat10 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
    # print rssError(abY[0:99],yHat01)
    # print rssError(abY[0:99],yHat1)
    # print rssError(abY[0:99],yHat10)

    # 岭回归，对应30个不同的λ值，并将其画出
    abX, abY = loadDataSet("abalone.txt")
    # ridgeWeight = ridgeTest(abX, abY)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridgeWeight)
    # plt.show()
    returnMat = stageWise(abX,abY,0.001,5000)
    # print returnMat