# -*- coding: UTF-8 -*-
import sys
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.mllib.evaluation import BinaryClassificationMetrics

impurityParmX =0 
maxDepthParmX=0
maxBinsParmX=0
eva_auc=0
test_auc=0

def SetLogger( sc ):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)    

def SetPath(sc):
    global Path
    if sc.master[0:5]=="local" :
        Path="file:/home/harry/project/"
    else:   
        Path="hdfs://master:9000/user/hduser/"


def get_mapping(rdd, idx):
    return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()

def extract_label(record):
    label=(record[-1])
    return float(label)

def extract_features(fields,featureEnd):
    numericalFeatures=[convert_float(field)  for  field in fields[5: featureEnd]]    
    return numericalFeatures

def convert_float(x):
    return (0 if x=="?" else float(x))

def PrepareData(sc): 
    #----------------------1.匯入並轉換資料------------------------------------
    print("開始匯入資料...")
    rawDataWithHeader = sc.textFile("s3n://buck20170603/train268.csv",9)
    header = rawDataWithHeader.first() 
    rawData = rawDataWithHeader.filter(lambda x:x !=header)    
    lines = rawData.map(lambda x: x.split(","))
    print("Total count：" + str(lines.count()))
    #----------------------2.建立訓練評估所需資料 RDD[LabeledPoint]-------------
    labelpointRDD = lines.map( lambda r:LabeledPoint(
                                  extract_label(r), 
                                  extract_features(r,len(r) - 1)))
    #print "labelpointRDD=",labelpointRDD.first(),"\n"
    
    #----------------------3.以隨機方式將資料分為3部份並且回傳-------------
    (trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
    print("將資料分trainData:" + str(trainData.count()) + 
              "   validationData:" + str(validationData.count()) +
              "   testData:" + str(testData.count()))
    return (trainData, validationData, testData) #回傳資料

    
def PredictData(sc,model): 
    
    i = 0
    c = 0
    w = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    p = []
    global impurityParmX
    global maxDepthParmX
    global maxBinsParmX
    global eva_auc
    global test_auc
    print("importing data...")
    rawDataWithHeader = sc.textFile("s3n://buck20170603/train2652.csv",9)
    header = rawDataWithHeader.first() 
    rawData = rawDataWithHeader.filter(lambda x:x !=header)    
    lines = rawData.map(lambda x: x.split(","))
    print("Count：" + str(lines.count()))
    number = 500000 #lines.count()
    print("Predict: "+str(number))
    dataRDD = lines.map(lambda r: (extract_label(r),extract_features( r,len(r)-1 )))
    #(trainData, validationData, testData) = dataRDD.randomSplit([8, 1, 1])                                                
    
    for data in dataRDD.take(number):
	i = i + 1
	predictResult = model.predict(data[1])
	p.append(predictResult)
	label = (data[0])
	print("No."+str(i))
        print("predictResult: "+str(predictResult))
	print("label: "+str(data[0]))
	if(predictResult == label):
		#print("Correct!! ")
		c = c +1
		if(predictResult == 1):
			tp = tp +1
    		else:
			tn = tn +1
			
	else:
		#print("Wrong!! ")
		w = w +1
                if(predictResult == 1):
			fp = fp +1
		else:
			fn = fn +1
			
	#print("Predicted: "+str(predictResult)+", Label: "+str(label))
	#print("Features: "+str(data[1]))
	print("")
    testerror = float(fp+fn)/number
    print("Total Correct:"+str(c)+", Wrong:"+str(w))
    print("TP:"+str(tp)+", FN:"+str(fn))
    print("FP:"+str(fp)+", TN:"+str(tn))
    precision = round(tp / float(tp + fp),3)
    recall = round(tp / float(tp + fn),3)
    fm = round((2 *precision * recall) / float(precision + recall),3)
    print("Precision: "+str(precision)+" Recall: "+str(recall))
    print("F-measure: "+str(fm))
    print("Test error:"+str(testerror))
    #print("impurityParm: "+str(impurityParmX)+" ,maxDepthParm: "+str(maxDepthParmX)+" ,maxBinsParm: "+str(maxBinsParmX))
    print("Eva auc: "+str(eva_auc)+" ,Test auc: "+str(test_auc))
    form ={
        "error":[testerror],
        "TP":[tp],
	"FN":[fn],
	"FP":[fp],
	"TN":[tn]        
        }
    form2 ={
        "prediction":p        
        }
    f1 = pd.DataFrame(form)
    f2 = pd.DataFrame(form2)
    f1.to_csv('pred51.csv', sep=',',index=False)
    f2.to_csv('pred52.csv', sep=',',index=False)
 
def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels=score.zip(validationData.map(lambda p: p.label))
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    AUC=metrics.areaUnderROC
    
    return(AUC)

def trainEvaluateModel(trainData,validationData,
                                        impurityParm, maxDepthParm, maxBinsParm):
    startTime = time()
    model = DecisionTree.trainClassifier(trainData,
                numClasses=2, categoricalFeaturesInfo={},
                impurity=impurityParm,
                maxDepth=maxDepthParm, 
                maxBins=maxBinsParm)
    
    AUC = evaluateModel(model, validationData)
    global eva_auc
    eva_auc = AUC

    duration = time() - startTime
    print    ("訓練評估：使用參數" + \
                " impurity="+str(impurityParm) +\
                " maxDepth="+str(maxDepthParm) + \
                " maxBins="+str(maxBinsParm) +\
                 " 所需時間="+str(duration) + \
                 " 結果AUC = " + str(AUC) )
    return (AUC,duration, impurityParm, maxDepthParm, maxBinsParm,model)


def evalParameter(trainData, validationData, evalparm,
                  impurityList, maxDepthList, maxBinsList):
    
    metrics = [trainEvaluateModel(trainData, validationData,  
                                impurity,maxDepth,  maxBins  ) 
                       for impurity in impurityList
                       for maxDepth in maxDepthList  
                       for maxBins in maxBinsList ]
    
    if evalparm=="impurity":
        IndexList=impurityList[:]
    elif evalparm=="maxDepth":
        IndexList=maxDepthList[:]
    elif evalparm=="maxBins":
        IndexList=maxBinsList[:]
    
    df = pd.DataFrame(metrics,index=IndexList,
            columns=['AUC', 'duration','impurity', 'maxDepth', 'maxBins','model'])
    showchart(df,evalparm,'AUC','duration',0.5,0.7 )
    


    
def evalAllParameter(trainData, validationData, 
                     impurityList, maxDepthList, maxBinsList):    
    metrics = [trainEvaluateModel(trainData, validationData,  
                            impurity,maxDepth,  maxBins  ) 
                      for impurity in impurityList 
                      for maxDepth in maxDepthList  
                      for  maxBins in maxBinsList ]
    
    Smetrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParameter=Smetrics[0]
    
    global impurityParmX
    global maxDepthParmX
    global maxBinsParmX
    impurityParmX = str(bestParameter[2])
    maxDepthParmX = str(bestParameter[3])
    maxBinsParmX = str(bestParameter[4])
    print("調校後最佳參數：impurity:" + str(bestParameter[2]) + 
                                      "  ,maxDepth:" + str(bestParameter[3]) + 
                                     "  ,maxBins:" + str(bestParameter[4])   + 
                                      "  ,結果AUC = " + str(bestParameter[0]))
    
    return bestParameter[5]



def CreateSparkContext():
    sparkConf = SparkConf()                                                       \
                         .setAppName("RunDecisionTreeBinary")                         \
                         .set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    print ("master="+sc.master)    
    SetLogger(sc)
    SetPath(sc)
    return (sc)

if __name__ == "__main__":
    print("RunDecisionTreeBinary")
    sc=CreateSparkContext()
    print("==========phrase Data preparation===============")
    (trainData, validationData, testData) =PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    print("==========phrase evaluate trainging===============")
    (AUC,duration, impurityParm, maxDepthParm, maxBinsParm,model)= \
        trainEvaluateModel(trainData, validationData, "entropy", 10, 10)
        
    if (len(sys.argv) == 2) and (sys.argv[1]=="e"):
        parametersEval(trainData, validationData)
    elif   (len(sys.argv) == 2) and (sys.argv[1]=="a"): 
        print("----to find the best combination of parameters---------")  
        model=evalAllParameter(trainData, validationData,
                          ["gini", "entropy"],
                          [3, 5, 10, 15, 20, 25], 
                          [3, 5, 10, 50, 100, 200 ])
    print("==========phrase Testing===============")
    auc = evaluateModel(model, testData)
    global test_auc
    test_auc = auc
    print("使用test Data測試最佳模型,結果 AUC:" + str(auc))
    print("==========Data prediction===============")
    PredictData(sc, model)
    print (model.toDebugString())
    model.save(sc, "target/BDT_click5")
    print ("Done")

