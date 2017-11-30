#-*-coding:utf-8-*-
from numpy import *

def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 朴素贝叶斯分类器训练函数
def trainNB0(dataSetMat,label):

    dataNum = len(dataSetMat)

    chacNum = len(dataSetMat[0])

    p1 = sum(label)/float(dataNum)

    # p1Num = zeros(chacNum)
    # 拉普拉斯平滑
    p1Num = ones(chacNum)
    # p0Num = zeros(chacNum)
    p0Num = ones(chacNum)
    # p1Total = 0.0
    p1Total = 2.0
    # p0Total = 0.0
    p0Total = 2.0

    for data in range(dataNum):
        if label[data] == 1:
            p1Num += dataSetMat[data]
            p1Total += sum(dataSetMat[data])
        else:
            p0Num += dataSetMat[data]
            p0Total += sum(dataSetMat[data])

    p1V = log(p1Num/p1Total)
    p0V = log(p0Num/p0Total)

    return p0V,p1V,p1

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 -pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as ',classifyNB(thisDoc,p0V,p1V,pAb)


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open(r'email/span/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open(r'email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is :',float(errorCount)/len(testSet)




if __name__ == "__main__":
    dataSet,label = loadDataSet()
    vocaSet = createVocabList(dataSet)

    # trainMat = []
    # for i in dataSet:
    #     trainMat.append(setOfWords2Vec(vocaSet,i))
    # p0V,p1V,p1 = trainNB0(trainMat,label)
    # print p1
    # print len(trainMat)
    # testingNB()
    spamTest()