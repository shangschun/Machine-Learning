# -*- coding: utf-8 -*- 2
from numpy import *
from numpy import linalg as la

def svdTest():
    # SVD小测试
    data = [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]
    # 左奇异矩阵，sigma，右奇异矩阵
    U, sigma, VT = linalg.svd(data)
    print sigma
    UID = mat(U[:, :3])
    sigmaID = eye(3) * sigma[:3]
    VTID = mat(VT[:3:])
    # 将原始数据data分解为（7*3）*（3*3）*（3*5）的三个矩阵的成绩
    dataID = UID * sigmaID * VTID
    return dataID

def loadExData():
    return [[0, 0, 0, 2, 2],
             [0, 0, 0, 3, 3],
             [0, 0, 0, 1, 1],
             [1, 1, 1, 0, 0],
             [2, 2, 2, 0, 0],
             [5, 5, 5, 0, 0],
             [1, 1, 1, 0, 0]]

def loadExData2():
    return [[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
             [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
             [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
             [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
             [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
             [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
             [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
             [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]

def euclidSim(inA,inB):
    return (1.0/(1.0 + la.norm(inA - inB)))

def pearsSim(inA,inB):
    corrcoefMat = corrcoef(inA,inB,rowvar=0)
    return 0.5+0.5*corrcoefMat[0][1]

def cosSim(inA,inB):
    num = float(inA.T*inB)
    inALength = sqrt(float(inA.T*inA))
    inBLength = sqrt(float(inB.T*inB))
    denom = inALength * inBLength
    # denom = la.norm(inA) * la.norm(inB)
    return 0.5+0.5*(num/denom)

# 基于物品相似度的推荐引擎
# 参数：数据矩阵、用户编号、相似度计算方法、物品编号
def standEst(dataMat,user,simMeas,item):
    # 物品数
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        # 用户user对商品j的评分
        userRating = dataMat[user,j]
        if userRating == 0:
            continue
        # 统计对物品item和j都评分的用户编号
        overLap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        # 若没有用户都对item和j评分，则相似度为0
        if len(overLap) == 0:
            similarity = 0
        # 若有，则抽取出来，计算相似度
        else:
            similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])
        # 相似度求和
        simTotal += similarity
        # 预测用户user对物品item的评分总和
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        # 归一化预测评分
        return ratSimTotal/simTotal

# 参数：数据、用户编号、选择预测评分最高的N个结果、相似度计算方法、用户对物品的预测估分方法
def recommand(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
    # 找没有被用户user评分的物品
    unratedItems = nonzero(dataMat[user,:].A==0)[1]
    # 若都评分则退出，不需要再推荐
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    # 遍历未评分的物品
    for item in unratedItems:
        # 预测用户user对为评分物品item的估分
        estimatedScore = estMethod(dataMat,user,simMeas,item)
        # 存（物品编号，对应估分值）
        itemScores.append((item,estimatedScore))
    # 选择最高的估分结果:从高到低排序
    return sorted(itemScores,key=lambda x:x[1],reverse=True)[:N]

# 基于SVD的评分估计
# 参数：用户数据、用户编号、相似度计算函数、物品编号
def svdEst(dataMat,user,simMeas,item):
    # 物品数
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    # SVD分解
    U,sigma,VT = la.svd(dataMat)
    # 构建对角矩阵，取前3个奇异值，确保总能量>90%
    sig3 = mat(eye(3) * sigma[:3])
    # SVD降维，重构低维空间的物品
    xformedItems = dataMat.T * U[:,:3] * sig3.I
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j == item:continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        print "the %d and %d similarity is: %f" % (item,j,similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

def printMat(inMat,thresh=0.0):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print 1,
            else:print 0,
        print " "

def imgCompress(numSV=3,thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print "****original matrix****"
    printMat(myMat,thresh)
    U,sigma,VT = la.svd(myMat)
    sigRecon = mat(zeros((numSV,numSV)))
    for k in range(numSV):
        sigRecon[k,k] = sigma[k]
    reconMat = U[:,:numSV] * sigRecon * VT[:numSV,:]
    print("****reconstructed matrix using %d singular values******" % numSV)
    # 输出阈值处理后的重构图像
    print(printMat(reconMat, thresh))

if __name__=="__main__":
    # dataID = svdTest()
    # myMat = mat(loadExData())
    # U,sigma,VT = la.svd(myMat)
    # recom = recommand(myMat,1,estMethod=svdEst)
    # print recom
    imgCompress(2)