
######################################################################################9.3

## Mount S3 bucket nycdsabootcamp to the Databricks File System
s3Path = "s3a://{0}:{1}@{2}".format("...", 
                                    "...", 
                                    "...")
mntPath = "/mnt/.../"
try:
  dbutils.fs.mount(s3Path, mntPath)
except:
  pass

  %fs
ls /mnt/.../consumer_hh_data/


part_0 = spark.read.parquet('/mnt/.../...gz.parquet')

#####rbind dataframe together

data = spark.read.parquet('/mnt/.../...gz.parquet')
for i in range(1,6):
  a=('/mnt/.../...gz.parquet' % (i))
  part_0=spark.read.parquet(a)
  data=data.unionAll(part_0)
  print(data.count())


 ####understand the dataset 

part_0.show()
len(part_0.columns)


######### select the categorical content 
import pandas as pd
import numpy as np
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import *


a.filter("avg(F7101) is null").show()


numlist=dict()
for index,i in enumerate(data.columns):
  if data.select(avg(i)).filter('avg(%s) is null' % (i)).count()==0:
    numlist[index]=i



######find the missing value in every column
missingvalue=dict()
for i in part_0.columns:  
  a=part_0.filter("%s is null" % (i)).count()
  missingvalue[i]=a


####### convert string to double
w=0
for i in part_20.columns:
  part_20=part_20.withColumn(i,part_20[i].cast(DoubleType()).alias(i))
  w=w+1
  print(w)

#####drop the column with a lot of none 
for i in part_20.columns:
	a=part_20.filter('%s is null' % (i)).count()
	if a > 200000:
		part_20.drop('%s' % (i))


w=0
for i in part_20.columns:
  w=w+1
  print(w)
  a=part_20.filter('%s is null' % (i)).count()
  if a > 200000:
    part_20.drop('%s' % (i))



##### scale
scaler = MinMaxScaler(inputCol="Revenue", outputCol="scaledRevenue")
scalerModel = scaler.fit(dataFrame)
scaledData = scalerModel.transform(dataFrame)


#### change na to 0
part_20_0=part_20_0.fillna(0)


##### drop the column+ use assembler
part_20_0=part_20.drop('loc')


from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=part_20_0.columns, 
                            outputCol="features")
output = assembler.transform(part_20_0)


####scale the whole dataset

from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(output)

# Normalize each feature to have unit standard deviation.
scaledData = scalerModel.transform(output)
scaledData.show(10)


##### PCA 
PCA on spark

from pyspark.ml.feature import *
from pyspark.ml.linalg import Vectors, VectorUDT
data = [(Vectors.dense([0.0, 1.0, 0.0, 7.0, 0.0]),),
         (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
         (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df = sqlContext.createDataFrame(data,["features"])
pca_extracted = PCA(k=2, inputCol="features", outputCol="pca_features")
model = pca_extracted.fit(df)
model.transform(df).collect()


#####S3 path 
part_20.write.csv('s3n://...',header=True)

AKIAJMP6X3UIRUOD533A
QvKT5PTkg8PmR/Arz+Gqv/CISor6aJPVwkOOb2Ig


#####andra code for num and categoric column


numList=dict()
catList=dict()
for index, i in enumerate(data.columns):
  avg_i=data.select(avg(i))
  if (avg_i.filter(‘avg(%s) = 0’ %(i)).count()==1):
    catList[index]=i
  elif (avg_i.filter((‘avg(%s) is null’) %(i)).count()==1):
    catList[index]=i
  else:
    numList[index]=i

#check  the result, numeric: 680, categoric: 320


######################### fit the model change the value of k
from pyspark.ml.feature import *
from pyspark.ml.linalg import Vectors, VectorUDT
pca_extractedk203 = PCA(k=203, inputCol="scaledFeatures", outputCol="pca_features")
pcamodel = pca_extractedk203.fit(scaledData)
pcaresult=pcamodel.transform(scaledData).select('pca_features')
pcaresult.show(25,truncate=False)


##############for imputation
from pyspark.ml.feature import Imputer

imputer = Imputer(
    inputCols=df.columns, 
    outputCols=["{}_imputed".format(c) for c in df.columns]
)
part20imput=imputer.fit(part20num).transform(part20num)


###################################################################### for clustering
from pyspark.ml.clustering import KMeans


# Trains a k-means model.
kmeans = KMeans().setK(200).setSeed(1)
model = kmeans.fit(scaledDatapart20)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(scaledDatapart20)
print("Within Set Sum of Squared Errors = " + str(wssse))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


################################################################### for different k and plot
import numpy as np
import matplotlib.pyplot as plt
ylist=[]
for i in klist:
  kmeans = KMeans().setK(i)
  model = kmeans.fit(scaledDatapart20)
  wssse = model.computeCost(scaledDatapart20)
  ylist.append(wssse)

fig, ax = plt.subplots()
x=klist
y=ylist
ax.plot(x, y)
ax.plot(x, y, 'ro')
display(fig)











