# -*- coding: UTF-8 -*-
import sys
import os
from time import time
from time import sleep
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.feature import StandardScaler


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
        Path="hdfs://localhost:9000/user/harry/"
#如果您要在cluster模式執行(hadoop yarn 或Spark Stand alone)，請依照書上說明，先上傳檔案至HDFS目錄
#test booking rate by adding click rate features

def get_mapping(rdd, idx):
    return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()

def extract_label(record):
    label=(record[-2])
    return float(label)

def extract_features(fields,featureEnd):
    numericalFeatures=[convert_float(field)  for  field in fields[4: featureEnd]]    
    return numericalFeatures

def convert_float(x):
    return (0 if x=="?" else float(x))

def PrepareData(sc): 
    #----------------------1.匯入並轉換資料-------------
    print("開始匯入資料...")
    rawDataWithHeader = sc.textFile(Path+"data/train28.csv")
    header = rawDataWithHeader.first() 
    rawData = rawDataWithHeader.filter(lambda x:x !=header)    
    lines = rawData.map(lambda x: x.split(","))
    print("共計：" + str(lines.count()) + "筆")
    #----------------------2.建立訓練評估所需資料 RDD[LabeledPoint]-------------
    print "標準化之前：",
    labelRDD = lines.map(lambda r:  extract_label(r))
    featureRDD = lines.map(lambda r:  extract_features(r,len(r) - 2))
    print(featureRDD.first())
    print ""       
    print "標準化之後：",       
    stdScaler = StandardScaler(withMean=True, withStd=True).fit(featureRDD)
    ScalerFeatureRDD=stdScaler.transform(featureRDD)
    print(ScalerFeatureRDD.first())
    labelpoint=labelRDD.zip(ScalerFeatureRDD)
    labelpointRDD=labelpoint.map(lambda r: LabeledPoint(r[0], r[1]))
    #----------------------3.以隨機方式將資料分為3部份並且回傳-------------
    (trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
    print("將資料分trainData:" + str(trainData.count()) + 
              "   validationData:" + str(validationData.count()) +
              "   testData:" + str(testData.count()))
    return (trainData, validationData, testData) #回傳資料

    
def PredictData(sc,model): 
    i = 0  #做confusion matrix
    c = 0
    w = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    number = 20000
    print("開始匯入資料...")
    rawDataWithHeader = sc.textFile(Path+"data/train281.csv")
    header = rawDataWithHeader.first() 
    rawData = rawDataWithHeader.filter(lambda x:x !=header)    
    lines = rawData.map(lambda x: x.split(","))
    print("共計：" + str(lines.count()) + "筆")
    dataRDD = lines.map(lambda r: (extract_label(r),extract_features( r,len(r)-2) ))
                                                    

    for data in dataRDD.take(number):
	i = i + 1
	predictResult = int(model.predict(data[1]))
	label = int(data[0])
	print("No."+str(i))
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
			
	print("")
        testerror = float(fp+fn)/number
    print("Total Correct:"+str(c)+", Wrong:"+str(w))
    print("TP:"+str(tp)+", FN:"+str(fn))
    print("FP:"+str(fp)+", TN:"+str(tn))
    print("Test error:"+str(testerror))

def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels=score.zip(validationData \
                                   .map(lambda p: p.label))  \
                                   .map(lambda (x,y): (float(x),float(y)) )
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    AUC=metrics.areaUnderROC
    return( AUC)


def trainEvaluateModel(trainData,validationData,
                                        numIterations, stepSize, miniBatchFraction):
    startTime = time()
    model = LogisticRegressionWithSGD.train(trainData, 
                                        numIterations, stepSize, miniBatchFraction)
    AUC = evaluateModel(model, validationData)
    duration = time() - startTime
    print    "訓練評估：使用參數" + \
                " numIterations="+str(numIterations) +\
                " stepSize="+str(stepSize) + \
                " miniBatchFraction="+str(miniBatchFraction) +\
                 " 所需時間="+str(duration) + \
                 " 結果AUC = " + str(AUC) 
    return (AUC,duration, numIterations, stepSize, miniBatchFraction,model)


def evalParameter(trainData, validationData, evalparm,
                  numIterationsList, stepSizeList, miniBatchFractionList):
    
    metrics = [trainEvaluateModel(trainData, validationData,  
                                numIterations,stepSize,  miniBatchFraction  ) 
                       for numIterations in numIterationsList
                       for stepSize in stepSizeList  
                       for miniBatchFraction in miniBatchFractionList ]
    
    if evalparm=="numIterations":
        IndexList=numIterationsList[:]
    elif evalparm=="stepSize":
        IndexList=stepSizeList[:]
    elif evalparm=="miniBatchFraction":
        IndexList=miniBatchFractionList[:]
    
    df = pd.DataFrame(metrics,index=IndexList,
            columns=['AUC', 'duration','numIterations', 'stepSize', 'miniBatchFraction','model'])
    
    showchart(df,evalparm,'AUC','duration',0.5,0.7 )
    
def showchart(df,evalparm ,barData,lineData,yMin,yMax):
    ax = df[barData].plot(kind='bar', title =evalparm,figsize=(10,6),legend=True, fontsize=12)
    ax.set_xlabel(evalparm,fontsize=12)
    ax.set_ylim([yMin,yMax])
    ax.set_ylabel(barData,fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[[lineData ]].values, linestyle='-', marker='o', linewidth=2.0,color='r')
    plt.show()
        
def evalAllParameter(trainData, validationData, 
                     numIterationsList, stepSizeList, miniBatchFractionList):    
    metrics = [trainEvaluateModel(trainData, validationData,  
                            numIterations,stepSize,  miniBatchFraction  ) 
                      for numIterations in numIterationsList 
                      for stepSize in stepSizeList  
                      for  miniBatchFraction in miniBatchFractionList ]
    
    Smetrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParameter=Smetrics[0]
    
    print("調校後最佳參數：numIterations:" + str(bestParameter[2]) + 
                                      "  ,stepSize:" + str(bestParameter[3]) + 
                                     "  ,miniBatchFraction:" + str(bestParameter[4])   + 
                                      "  ,結果AUC = " + str(bestParameter[0]))
    
    return bestParameter[5]

def parametersEval(trainData, validationData):
    print("----- 評估numIterations參數使用 ---------")
    evalParameter(trainData, validationData,"numIterations", 
                              numIterationsList=[5, 15, 20, 60, 100],   
                              stepSizeList=[10],  
                              miniBatchFractionList=[1 ])  
    print("----- 評估stepSize參數使用 ---------")
    evalParameter(trainData, validationData,"stepSize", 
                              numIterationsList=[100],                    
                              stepSizeList=[10, 50, 100, 200],    
                              miniBatchFractionList=[1])   
    print("----- 評估miniBatchFraction參數使用 ---------")
    evalParameter(trainData, validationData,"miniBatchFraction", 
                              numIterationsList=[100],      
                              stepSizeList =[100],        
                              miniBatchFractionList=[0.5, 0.8, 1 ])



def CreateSparkContext():
    sparkConf = SparkConf()                                                       \
                         .setAppName("LogisticRegressionWithSGD")                         \
                         .set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    print ("master="+sc.master)    
    SetLogger(sc)
    SetPath(sc)
    return (sc)

if __name__ == "__main__":
    print("RunLogisticRegressionWithSGDBinary")
    sc=CreateSparkContext()
    print("HOW")
    
    print("==========資料準備階段===============")
    (trainData, validationData, testData) =PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    print("==========訓練評估階段===============")
    #(AUC,duration, numIterationsParm, stepSizeParm, miniBatchFractionParm,model)= \
    #      trainEvaluateModel(trainData, validationData, 25, 10, 0.5)
    if (len(sys.argv) == 2) and (sys.argv[1]=="e"):
        parametersEval(trainData, validationData)
    elif   (len(sys.argv) == 2) and (sys.argv[1]=="a"): 
        print("-----所有參數訓練評估找出最好的參數組合---------")  
        model=evalAllParameter(trainData, validationData,
                         [70,80,100],   #自己列出幾種可能的參數組合
                         [10,20,50,70],
                          [0.5, 0.8, 1 ])
    print("==========測試階段===============")
   
    auc = evaluateModel(model, testData)
    print("使用test Data測試最佳模型,結果 AUC:" + str(auc))
    sleep(6)
    print("==========預測資料===============")
    
    PredictData(sc, model)
    #將訓練好的model儲存
    #model.save(sc, "target/pythonLogisticRegressionWithLSGDSModel3")  
    print("done")
    
