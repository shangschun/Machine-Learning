# -*- coding: utf-8 -*- 2
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

# 加载数据
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat
# 计算两个向量的欧式距离
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA - vecB,2)))

# 为给定数据集构建k个随机质心
def randCent(dataSet,k):
    # 特征维度
    n = shape(dataSet)[1]
    # 创建聚类中心矩阵
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        # 第j维，每次随机生成k个中心
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)
    return centroids

# kMeans算法
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    # 样本总数
    m = shape(dataSet)[0]
    # 分配样本到最近的簇，存[簇序号，距离的平方]
    clusterAssment = mat(zeros((m,2)))
    # step1：初始化聚类中心
    centroids = createCent(dataSet,k)
    clusterChanged = True
    # 所有样本分配结果不再改变，迭代停止
    while clusterChanged:
        clusterChanged = False
        # step2：分配到最近的聚类中心对应的簇中
        for i in range(m):
            # 对于每个样本定义最小距离
            minDist = inf
            minIndex = -1
            # 计算每个样本与k个中心的距离
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    # 获取最小距离及对应的簇号
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                # 分配样本到最近的簇
            clusterAssment[i,:] = minIndex,minDist ** 2
        print centroids
        # step：更新聚类中心
        for cent in range(k):
            # 获取该簇的所有样本点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            # 更新聚类中心，axis=0沿列方向求均值
            centroids[cent,:] = mean(ptsInClust,axis = 0)
    return centroids,clusterAssment
# 二分Kmeans
def biKmeans(dataSet,k,distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet,axis=0).tolist()[0]
    centList = [centroid0]
    # 计算初始总误差SSE
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:]) ** 2
    while (len(centList) < k):
        # 初始化SSE
        lowestSSE = inf
        for i in range(len(centList)):
            # 获取当前簇cluster=i内的数据
            ptsInCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]
            # 对cluster=i的簇进行kmeans划分
            centroidMat,splitClustAss = kMeans(ptsInCluster,2,distMeas)
            # cluster = i的簇被划分为两个子簇后的SSE
            sseSplit = sum(splitClustAss[:,1])
            # 除了cluster = i的簇，其他簇的SSE
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1])
            print "sseSplit, and notSplit:",sseSplit,sseNotSplit
            # 找最佳的划分簇，使得划分后 总SSE=sseSplit + sseNotSplit最小
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseNotSplit + sseSplit
        # 将最佳被划分簇的聚类结果为1的类别，更换类别为len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        # 将最佳被划分簇的聚类结果为0的类别，更换类别为bestCentToSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print "the bestCentToSplit is :",bestCentToSplit
        print "the len of bestClustAss is :",len(bestClustAss)
        # 将被划分簇的一个中心，替换为划分后的两个中心
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        # 更新整体的聚类效果clusterAssment(类别，SSE)
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return centList,clusterAssment

# 球面距离计算
# 点A ，纬度β1 ，经度α1 ；点B ，纬度β2 ，经度α2
# 则距离S=R·arccos[cosβ1cosβ2cos（α1-α2）+sinβ1sinβ2]，其中R为球体半径。
def distSLC(vecA,vecB):
    a = sin(vecA[0,1] * pi/180) * sin(vecB[0,1] * pi/180)
    b = cos(vecA[0,1] * pi/180) * cos(vecB[0.1] * pi/180) * cos(pi * (vecB[0,0] - vecA[0,0])/180)
    return arccos(a + b) * 6371.0

def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        # 取数据集中的纬度和经度
        datList.append([float(lineArr[4]),float(lineArr[3])])
    datMat = mat(datList)
    myCentroids,clustAssing = biKmeans(datMat,numClust)
    fig = plt.figure()
    rect = [0.1,0.1,0.8,0.8]
    scatterMarkers = ['s','o','^','8','p','d','v','h','>','<']
    axprops = dict(xticks = [],yticks = [])
    ax0 = fig.add_axes(rect,label = 'ax0', **axprops)
    # imread()基于一幅图像创建矩阵
    imgP = plt.imread('Portland.png')
    # imshow()绘制该矩阵
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect,label = 'ax1', frameon = False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A == i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0],ptsInCurrCluster[:,1].flatten().A[0],marker = markerStyle,s = 90)
    for i in range(len(myCentroids)):
        ax1.scatter(myCentroids[i][:,0].flatten().A[0],myCentroids[i][:,1].flatten().A[0],marker = '+',s = 300)
    plt.show()

if __name__ == "__main__":
    # kmeans
    # dataMat = mat(loadDataSet('testSet.txt'))
    # centroids = randCent(dataMat,2)
    # dist = distEclud(dataMat[0],dataMat[1])
    # kmean = kMeans(dataMat,4)
    # 二分k均值
    # dataMat = mat(loadDataSet('testSet2.txt'))
    # centList,myNewAssments = biKmeans(dataMat,3)
    # print centList
    # 例子：对地图上的点进行聚类
    clusterClubs()
