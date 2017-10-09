##library import
from pyspark.sql.types import DoubleType
import pandas as pd
import numpy as np
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import *
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml import Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import os

#load data from s3 path
s3Path = "s3a://{0}:{1}@{2}".format("...", 
                                    "...", 
                                    "...")
mntPath = "/mnt/.../"
try:
  dbutils.fs.mount(s3Path, mntPath)
except:
  pass

  %fs
ls /mnt/.../

#convert text to pyspark dataframe
ProViewrdd=sc.textFile("/mnt/.../Program_Viewing_data.txt")
demo_datardd=sc.textFile("/mnt/.../demo_data.txt")
programrdd=sc.textFile("/mnt/.../program_data.txt")

PVheader=ProViewrdd.first()
demoheader=demo_datardd.first()
PGheader=programrdd.first()

ProView_rdd = ProViewrdd.map(lambda k: k.split("\t"))
demo_rdd=demo_datardd.map(lambda k:k.split("\t"))
program_rdd=programrdd.map(lambda k:k.split("\t"))

ProView_df=ProView_rdd.toDF(PVheader.split("\t"))
program_df=program_rdd.toDF(PGheader.split("\t"))
demoheader1=demoheader.split("\t")
demo_df=demo_rdd.toDF(demoheader1)

print ProView_df.count(),len(ProView_df.columns)
print demo_df.count(),len(demo_df.columns)
print program_df.count(),len(program_df.columns)

ProView_df.printSchema()
demo_df.printSchema()
program_df.printSchema()


#change the type of dataframe 
doubleprodf=['TELECAST_KEY','PROGRAM_ID','TELECAST_REPORT_DURATION']
doubledemo=['HOUSEHOLD_ID','PERSON_ID','DAILY_PERSON_WEIGHT','AGE','KIDSUNDER18','KIDSUNDER12','KIDSUNDER6','KIDSUNDER3','TEENS',\
           'STATE_CODE','HHINCOME','INTERNETHOURSUSAGEHOME','INTERNETHOURSUSAGEWORK']
doubleproview=['HOUSEHOLD_ID', 'PERSON_ID', 'TELECAST_KEY', 'PROGRAM_ID','LIVEMINSVIEWEDWITHINTHEHHR', 'LIVEPLUSSDMINSIEWEDWITHINTHEHHR',\
               'LIVEPLUS3MINSVIEWEDWITHINTHEHHR', 'LIVEPLUS7MINSVIEWEDWITHINTHEHHR']
for i in doubleprodf:
  program_df=program_df.withColumn(i,program_df[i].cast(DoubleType()).alias(i))
for i in doubledemo:
  demo_df=demo_df.withColumn(i,demo_df[i].cast(DoubleType()).alias(i))
for i in doubleproview:
  ProView_df=ProView_df.withColumn(i,ProView_df[i].cast(DoubleType()).alias(i))


#join dataframe
demoPVJ=ProView_df.join(demo_df, (ProView_df.HOUSEHOLD_ID == demo_df.HOUSEHOLD_ID) & \
	(ProView_df.PERSON_ID == demo_df.PERSON_ID)).drop(demo_df.HOUSEHOLD_ID).drop(demo_df.PERSON_ID)


programPVJ=ProView_df.join(program_df, (ProView_df.PROGRAM_ID == program_df.PROGRAM_ID) & \
	(ProView_df.TELECAST_KEY == program_df.TELECAST_KEY)).drop(program_df.PROGRAM_ID).drop(program_df.TELECAST_KEY)


allJ=demoPVJ.join(program_df,(program_df.PROGRAM_ID==demoPVJ.PROGRAM_ID) & \
	(program_df.TELECAST_KEY==demoPVJ.TELECAST_KEY)).drop(program_df.PROGRAM_ID).drop(program_df.TELECAST_KEY)

print demoPVJ.count(),len(demoPVJ.columns)
print programPVJ.count(),len(programPVJ.columns)
print allJ.count(), len(allJ.columns)

#output analysis
inputcols=['VIEWINGSOURCETYPE', 'VIEWINGSOURCE',  'DAILY_PERSON_WEIGHT', 'GENDER', 'AGE', \
'EDUCATION', 'OCCUPATION', 'HISPANIC_FLAG', 'RACE', 'LANGUAGE_SPOKEN', 'INDIVIDUAL_EDUCATION_BREAK', \
'RELATIONTOHOH', 'HHSIZE', 'HOHRACE', 'HOHORIGIN', 'HHLANGUAGE_SPOKEN', 'KIDSUNDER18', 'KIDSUNDER12', 'KIDSUNDER6',\
 'KIDSUNDER3', 'TEENS', 'STATE_CODE', 'STATE_DESC', 'MARITAL_STATUS', 'CABLESTATUS', 'OWNSPC', 'HHINCOME', 'HASINTERNET', \
 'INTERNETHOURSUSAGEHOME', 'INTERNETHOURSUSAGEWORK', 'PROGRAM_DISTRIBUTOR_TYPE_DESC', 'PROGRAM_DISTRIBUTOR_NAME', \
 'PROGRAM_LONG_NAME', 'EPISODE_LONG_NAME', 'BROADCAST_DATE', 'TELECAST_REPORT_START_TIME', 'TELECAST_REPORT_DURATION', \
 'DAYPART_CLASS_DESCRIPTION', 'PROGRAM_DETAIL_TYPE_DESC', 'PROGRAM_STANDARD_TYPE_DESC']


for i in range(40):
  allJ.groupby(inputcols[i]).agg(mean(allJ['LIVEMINSVIEWEDWITHINTHEHHR']).alias('Average')).orderBy('Average',ascending=False).show()

#data visulization plot
def pltBar(xVal, yVal, data, title):
  vizData = data.select(xVal, yVal)
  grpViz = vizData\
            .groupBy(xVal)\
            .agg(mean(vizData[yVal]).alias("Avg_{0}".format(yVal)), \
                  stddev(vizData[yVal]).alias("Std_Dev_{0}".format(yVal)),\
                  count(vizData[xVal]).alias('Count'),\
                  min(vizData[yVal]).alias('Min_{0}_value'.format(yVal)),\
                  max(vizData[yVal]).alias('Max_{0}_value'.format(yVal)))\
            .orderBy(xVal, ascending=True)
  grpViz = grpViz.toPandas()
  plt.clf()
  fig = plt.figure(1, figsize=(9, 6))
  ax = sns.barplot(x=xVal, y=("Avg_{0}".format(yVal)), data=grpViz, color='blue', alpha=.2)
  plt.tick_params(labelsize=8)
  plt.title(title, fontsize = 24)
  plt.xlabel(xVal, fontsize = 14)
  plt.ylabel(yVal, fontsize = 14)
  display(fig)
  
  def pltLine(xVal, yVal, data, title):
  vizData = data.select(xVal, yVal)
  grpViz = vizData\
            .groupBy(xVal)\
            .agg(mean(vizData[yVal]).alias("Avg_{0}".format(yVal)), \
                  stddev(vizData[yVal]).alias("Std_Dev_{0}".format(yVal)),\
                  count(vizData[xVal]).alias('Count'),\
                  min(vizData[yVal]).alias('Min_{0}_value'.format(yVal)),\
                  max(vizData[yVal]).alias('Max_{0}_value'.format(yVal)))\
            .orderBy(xVal, ascending=True)
  grpViz = grpViz.toPandas()
  plt.clf()
  fig = plt.figure(1, figsize=(9, 6))
  ax = sns.regplot(x=xVal, y=("Avg_{0}".format(yVal)), data=grpViz, ci = False,scatter_kws={"color":"darkred","alpha":0.3,"s":90},\
                   line_kws={"color":"g","alpha":0.5,"lw":4},marker="x")
  plt.tick_params(labelsize=8)
  plt.title(title, fontsize = 24)
  plt.xlabel(xVal, fontsize = 14)
  plt.ylabel(yVal, fontsize = 14)
  display(fig)

  pltBar('DAILY_PERSON_WEIGHT','LIVEMINSVIEWEDWITHINTHEHHR',allJ,'Average Live View for daily person weight')
  pltLine('DAILY_PERSON_WEIGHT','LIVEMINSVIEWEDWITHINTHEHHR',allJ,'Average Live View for daily person weight')
  pltBar('AGE','LIVEMINSVIEWEDWITHINTHEHHR',allJ,'Average Live View for different age')

#visulize all the output analysis
plt.clf()
figall=plt.figure()
for i in range(12):
  xVal=inputcols[i]
  yVal='LIVEMINSVIEWEDWITHINTHEHHR'
  allfori = allJ.groupBy(xVal).agg(mean(allJ[yVal]).alias("Avg_{0}".format(yVal))).orderBy(xVal, ascending=True)
  allfori = allfori.toPandas()
  x=allfori[xVal]
  y=allfori[("Avg_{0}".format(yVal))]
  ax1 = figall.add_subplot(4,3,i+1)
  ax1.bar(x.index,y)
  ax1.set_xticklabels(x.values, rotation=20,fontsize=5)
  ax1.set_title(xVal,fontsize=10)

display(figall)

plt.clf()
figall=plt.figure()
for i in range(12,24):
  xVal=inputcols[i]
  yVal='LIVEMINSVIEWEDWITHINTHEHHR'
  allfori = allJ.groupBy(xVal).agg(mean(allJ[yVal]).alias("Avg_{0}".format(yVal))).orderBy(xVal, ascending=True)
  allfori = allfori.toPandas()
  x=allfori[xVal]
  y=allfori[("Avg_{0}".format(yVal))]
  ax1 = figall.add_subplot(4,3,i-11)
  ax1.bar(x.index,y)
  ax1.set_xticklabels(x.values, rotation=20,fontsize=5)
  ax1.set_title(xVal,fontsize=10)

display(figall)

plt.clf()
figall=plt.figure()
for i in range(24,40):
  xVal=inputcols[i]
  yVal='LIVEMINSVIEWEDWITHINTHEHHR'
  allfori = allJ.groupBy(xVal).agg(mean(allJ[yVal]).alias("Avg_{0}".format(yVal))).orderBy(xVal, ascending=True)
  allfori = allfori.toPandas()
  x=allfori[xVal]
  y=allfori[("Avg_{0}".format(yVal))]
  ax1 = figall.add_subplot(4,4,i-23)
  ax1.bar(x.index,y)
  ax1.set_xticklabels(x.values, rotation=20,fontsize=5)
  ax1.set_title(xVal,fontsize=10)

display(figall)




#####start machine learning 
modelMVi=[1,2,4,5,6,8,9,10,11,13,15,16,17,20,21,24,26,28,29,31,32,33,35,36,37,38,39]
modelMVinput=list()
for i in modeli:
  modelMVinput.append(inputcols[i])
modelMVoutput=['LIVEMINSVIEWEDWITHINTHEHHR']
modelMV_df=allJ.select(modelMVinput+modelMVoutput)
print len(modelMV_df.columns),modelMV_df.count()

#one hot encoded
Categoricols = [item[0]+"ClassVec" for item in modelMV_df.dtypes if item[1].startswith('string')]
print Categoricols

Numericols=[item[0] for item in modelMV_df.dtypes if item[1].startswith('double')]
Numericols=Numericols[:-1]
print Numericols

modelMV_df_encoded=modelMV_df
for categoricalCol in Categoricols:
  indexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"Index")
  indexTransformer = indexer.fit(modelMV_df_encoded)
  indexed = indexTransformer.transform(modelMV_df_encoded)
  encoder = OneHotEncoder(inputCol = categoricalCol+"Index", outputCol = categoricalCol+"ClassVec")
  modelMV_df_encoded = encoder.transform(indexed)
print len(modelMV_df_encoded.columns),modelMV_df_encoded.count()

indexStages = [] 
for categoricalCol in Categoricols:
  stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"Index")
  indexStages.append(stringIndexer)
print 'StringIndexer stages:', indexStages

encodeStages = []
for categoricalCol in Categoricols:
  encoder = OneHotEncoder(inputCol=categoricalCol+"Index", outputCol=categoricalCol+"ClassVec")
  encodeStages.append(encoder)
print 'OneHotEncoder stages:', encodeStages

assembler = VectorAssembler(inputCols=Categoricols+Numericols, 
                            outputCol="features")
output = assembler.transform(modelMV_df_encoded)

#traing and test dataset
training, test = modelMV_df.randomSplit([0.7, 0.3])
training.cache()
test.cache()

lr = LinearRegression(featuresCol="lmFeatures", labelCol="LIVEMINSVIEWEDWITHINTHEHHR")
lmAssembler = VectorAssembler(inputCols=Categoricols+Numericols, 
                              outputCol="lmFeatures")

lrPipeline = Pipeline(stages = indexStages + encodeStages + [lmAssembler, lr])

lrModel = lrPipeline.fit(training)

#make prediction on test dataset
lrPredict = lrModel.transform(test)
lrPredict.printSchema()
evaluator = RegressionEvaluator(labelCol="LIVEMINSVIEWEDWITHINTHEHHR", 
                                predictionCol="prediction", 
                                metricName="rmse")
rmse=evaluator.evaluate(lrPredict)
print 'Root Mean Squared Error (RMSE) on test data = %g' % rmse





