# -*- coding: utf-8 -*- 2
def loadDataSet():
    return [[1,3,4,],[2,3,5],[1,2,3,5],[2,5]]

# 构建集合C1，C1是大小为1的所有候选项集的集合
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset,C1)

# 用来将C1生成L1，也就是将满足最低支持度的候选项集构成集合L1
# D：数据集
# Ck：候选项集
# minSupport：最小支持度
def scanD(D,Ck,minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItem = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItem
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList,supportData

# 由频繁k-1项集LK1，生成候选项集Ck
# Lk：频繁项集列表
# k：项集元素个数
def aprioriGen(Lk,k):
    # 保存新的候选集
    retList = []
    # 频繁项集记录数
    lenLk = len(Lk)
    # 比较两个子集的项集
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            # 若两个子集的前k-2个元素相同，就将这两个子集合并
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet,minSupport=0.5):
    # 生成1-项集
    C1 = createC1(dataSet)
    D = map(set,dataSet)
    # 候选项集C1->频繁1-项集L1
    L1,supportData = scanD(D,C1,minSupport)
    # 存放所有频繁项集
    L = [L1]
    # 由L1->C2
    k = 2
    while (len(L[k - 2]) > 0 ):
        Ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L,supportData

# 根据最小置信度过滤候选的关联规则，并返回过滤后的规则的后件
# freqSet：某频繁项集
# H：关联规则的后件
# supportData：所有项集的支持度
# brl：填充关联规则：前件、后件、置信度
# minConf：最小置信度
def calcConf(freqSet,H,supportData,brl,minConf=0.7):
    # 满足最小置信度要求的后件
    prunedH = []
    for conseq in H:
        # 使用支持度计算置信度
        conf = supportData[freqSet]/supportData[freqSet - conseq]
        if conf >= minConf:
            print freqSet - conseq,'--->',conseq,'conf:',conf
            # 保存关联规则
            brl.append((freqSet - conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

# 基于某个频繁项集，生成关联规则
# 输入：频繁项集、关联规则后件列表H、支持度列表、填充的规则列表、最小置信度
def rulesFromConseq(freqSet,H,supportData,brl,minConf=0.7):
    # 后件中项的个数
    m =len(H[0])
    # 后件中只有一个项
    if m == 1:
        H = calcConf(freqSet,H,supportData,brl,minConf)
    if(len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H,m + 1)
        Hmp1 = calcConf(freqSet,Hmp1,supportData,brl,minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet,Hmp1,supportData,brl,minConf)

# 产生关联规则
# 输入：频繁项集列表L、支持度列表，最小置信度
# 输出：满足最小置信度的关联规则列表
def generateRules(L,supportData,minConf=0.7):
    # 置信度规则列表
    bigRuleList = []
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if(i > 1):
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList


if __name__ == "__main__":
    dataSet = loadDataSet()
    L,supportData = apriori(dataSet,minSupport=0.5)
    rules = generateRules(L,supportData,minConf=0.5)
    print rules