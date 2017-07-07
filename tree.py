# *-* coding:utf-8 *-*
from math import log
import os
import matplotlib.pyplot as plt

def createDataSet():
	dataSet = [ [1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
	labels = ["No sufing", "Filper"]
	return dataSet, labels

def calcShannonEnt(dataSet):
	numDataSet = len(dataSet)
	dic = {}
	for feature in dataSet:
		label = feature[-1]
		if(label  not in dic.keys() ):
			dic[label] = 0
		dic[label] += 1
	shannon = 0.0
	for key in dic.keys():
		prob = float(dic[key])/numDataSet
		shannon -= prob*log(prob, 2)
	return shannon

# print calcShannonEnt( createDataSet() )

def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for feature in dataSet:
		if(feature[axis] == value):
			featureVec = feature[:axis]
			featureVec.extend(feature[axis+1:])
			retDataSet.append(featureVec)
	return retDataSet
#print splitDataSet(createDataSet(), 0, 1)

def chooseBestSplitMethod(dataSet):
	numberFeature = len(dataSet[0])-1
	baseShannon = calcShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = -1
	for i in range(numberFeature):
		featureCol = [example[i] for example in dataSet]
		uniqueVals = set(featureCol)
		NewEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float( len(dataSet) )
			NewEntropy += prob*calcShannonEnt(subDataSet)
		infoGain = baseShannon-NewEntropy
		if(infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

#print chooseBestSplitMethod(createDataSet())

def majorityCnt(classList):
	classCount = {}
	for value in classList:
		if value not in classCount.keys():
			classCount[value] = 0
		classCount[value] += 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return  sortedClassCount[0][0]

def createTree(dataSet, labels):
	#归类
    resultList = [example[-1] for example in dataSet]
    if( len(set(resultList)) == 1):
		return resultList[0]
    if( len(dataSet[0]) == 1 ):
    	return majorityCnt(resultList)
    bestFeat = chooseBestSplitMethod(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree  = { bestFeatLabel:{} }
    #del( labels[bestFeat] )
    featureValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featureValues)
    for value in uniqueVals:
    	subLabels = labels[:bestFeat]
    	subLabels.extend(labels[bestFeat+1:])
    	myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

#dataSet, labels = createDataSet()
#print createTree(dataSet, labels)

'''
decisionNode = dict(boxstyle = "sawtooth", fc="0.8")
leafNode = dict(boxstyle = "round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
createPlot()
'''

def plotNode(nodeText, centerPt, parentPt, nodeType):
	createPlot.ax1.annotate(nodeText, xy= parentPt, xycoords='axes fraction', 
	xytext=centerPt,textcoords='axes fraction',	va='center', ha='center',
	bbox=nodeType, arrowprops = arrow_args)
'''
def createPlot():
	fig = plt.figure(1, facecolor = 'white')
	fig.clf()
	createPlot.ax1 = plt.subplot(111, frameon =False)
	plotNode('decisionNode', (0.5, 0.1), (0.1, 0.5), decisionNode)
	plotNode('leafNode', (0.8, 0.1), (0.3, 0.8), leafNode)
        plt.show()
'''


def getNumberLeaf(myTree):
	numberLeaf = 0
	if( type(myTree).__name__ == 'dict'):
		for key in myTree.keys():
			numberLeaf += getNumberLeaf( myTree[key] )
	else:
		return 1

	return numberLeaf

def getTreeDepth(myTree):   
	maxDepth = 0
	if( type(myTree).__name__ == 'dict'):
		for key in myTree.keys():
			maxDepth = max(maxDepth, getTreeDepth(myTree[key]) )
	return maxDepth+1
def calcTreeDepth(myTree):
	return (getTreeDepth(myTree)-1)/2

'''
dataSet, labels = createDataSet()
myTree = createTree(dataSet, labels)
print calcTreeDepth(myTree)
'''

def plotMidText(cntrpt, parentPt, txtString):
	xMid = (parentPt[0]-cntrpt[0])/2+cntrpt[0]
	yMid = (parentPt[1]-cntrpt[1])/2+cntrpt[1]
	createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeText):
	numLeafs = getNumberLeaf(myTree)
	depth = getTreeDepth(myTree)
	firstStr = myTree.keys()[0]
	cntrpt = (plotTree.x0ff+(1.0+float(numLeafs))/2.0/plotTree.totalW ,plotTree.y0ff)
	plotMidText(cntrpt, parentPt, nodeText)
	plotNode(firstStr, cntrpt, parentPt, decisionNode)
	secondDict = myTree[firstStr]
	plotTree.y0ff = plotTree.y0ff - 1.0/plotTree.totalD
	for key in secondDict.keys():
		if( type(secondDict[key]).__name__ == 'dict'):
			plotTree(secondDict[key], cntrpt, str(key) )
		else:
			plotTree.x0ff += 1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.x0ff, plotTree.y0ff), cntrpt, leafNode)
			plotMidText( (plotTree.x0ff, plotTree.y0ff), cntrpt, str(key))
	plotTree.y0ff += 1.0/plotTree.totalD



def createPlot(inTree):
	fig = plt.figure(1, facecolor = 'white')
	fig.clf()
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
	plotTree.totalW = float( getNumberLeaf(inTree))
	plotTree.totalD = float( getTreeDepth(inTree))
	plotTree.x0ff = -0.5/plotTree.totalW
	plotTree.y0ff = 1.0
	plotTree(inTree, (0.5, 1.0), '')
	plt.show()

'''draw the tree
decisionNode = dict(boxstyle = "sawtooth", fc="0.8")
leafNode = dict(boxstyle = "round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
dataSet, labels = createDataSet()
myTree = createTree(dataSet, labels)
createPlot(myTree)
#print myTree
'''

def classify(inTree, featLabels, testVec):
	firstStr = inTree.keys()[0]
	secondDict = inTree[firstStr]
	index = featLabels.index(firstStr)
	for key in secondDict.keys():
		if(key == testVec[index]):
			if( type(secondDict[key]).__name__ == 'dict'):
				return classify(secondDict[key], featLabels, testVec)
			else:
				return secondDict[key]
'''predict
dataSet, labels = createDataSet()
myTree = createTree(dataSet, labels)
print myTree
print labels
print classify(myTree, labels, [1, 1])
'''

decisionNode = dict(boxstyle = "sawtooth", fc="0.8")
leafNode = dict(boxstyle = "round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
fr = open('lenses.txt')
lenses = [ inst.strip().split('\t') for inst in fr.readlines() ]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses, lensesLabels)
print classify(lensesTree, lensesLabels, ['young', 'hyper', 'yes', 'normal'])
createPlot(lensesTree)































