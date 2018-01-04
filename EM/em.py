# -*- coding: utf-8 -*- 2
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

isdebug = True
# 准备数据
# Sigma:方差
# Mu1,Mu2:k个分布的均值
# N:每个分布下总的样本数
def ini_data(Sigma,Mu1,Mu2,k,N):
    # 保存生成的随机样本
    global X
    # 求类别的均值
    global Mu
    # 保存样本属于某类的概率
    global Expectations
    # 1*N的矩阵，随机生成N个样本
    X = np.zeros((1,N))
    # 初始均值，两个分布的均值
    Mu = np.random.random(2)
    print Mu
    Expectations = np.zeros((N,k))
    for i in range(0,N):
        # 在大于0.5的在第一个分布，小于0.5的在第二个分布
        if np.random.random(1) > 0.5:
            X[0,i] = np.random.normal() * Sigma + Mu1
        else:
            X[0,i] = np.random.normal() * Sigma + Mu2
    if isdebug:
        print ("****************")
        print (u"初始观测数据X：")
        print X

# E步：计算每个样本属于男女各自的概率
# Sigma:方差
# k:类别
# N:样本数
def e_step(Sigma,k,N):
    # 样本属于某类的概率
    global Expectations
    # 两类均值
    global Mu
    # 样本
    global X
    # 遍历所有样本点，计算属于每个类别的概率
    for i in range(0,N):
        # 分母，用于归一化
        Denom = 0
        # 遍历男女两类，计算各自归一化分母
        for j in range(0,k):
            # 计算分母
            Denom += math.exp((-1/(2*(float(Sigma**2)))) * (float(X[0,i]-Mu[j])) **2)
        # 遍历男女两类，计算各自分子部分
        for j in range(0,k):
            Numer = math.exp((-1/(2*(float(Sigma**2)))) * (float(X[0,i]-Mu[j])) **2)
            # 每个样本属于该类别的概率
            Expectations[i,j] = Numer/Denom
    if isdebug:
        print ("**************")
        print (u"隐藏变量E（Z）：")
        print (len(Expectations))
        print (Expectations.size)
        print (Expectations.shape)
        print (Expectations)
# M步：期望最大化
def m_step(k,N):
    # 样本属于某类概率P(k|xi)
    global Expectations
    # 样本
    global X
    # 计算两类的均值
    for j in range(0,k):
        Numer = 0
        Denom = 0
        # 计算该类别下的均值和方差
        for i in range(0,N):
            Numer += Expectations[i,j] * X[0,i]
            Denom += Expectations[i,j]
        # 计算每个类别各自的均值uk
        Mu[j] = Numer/Denom


def run(Sigma,Mu1,Mu2,k,N,iter_num,Epsilon):
    # 初始化训练数据
    ini_data(Sigma,Mu1,Mu2,k,N)
    print (u"初始<u1,u2>:",Mu)
    for i in range(iter_num):
        # 保存上次两类均值
        Old_Mu = copy.deepcopy(Mu)
        # E步
        e_step(Sigma,k,N)
        # M步
        m_step(k,N)
        # 输出当前迭代次数及当前估计的值
        print (i,Mu)
        # 判断误差
        if sum(abs(Mu - Old_Mu) < Epsilon):
            break

if __name__=="__main__":
    ini_data(6,40,20,2,1000)
    plt.hist(X[0,:],100)
    plt.show()

    run(6,40,20,2,1000,1000,0.0001)
    plt.hist(X[0,:],100)
    plt.show()