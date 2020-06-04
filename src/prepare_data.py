import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, OneHotEncoderEstimator
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes,GBTClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = SparkSession.builder.appName('Airbnb').getOrCreate()


#########################################################
def get_data():
    
    train = spark.read.csv('../Data/train_users_2.csv',
                       header=True,
                       multiLine=True,
                       inferSchema=True)
    test = spark.read.csv('../Data/test_users.csv',
                       header=True,
                       multiLine=True,
                       inferSchema=True)
    age_gender_bkts = spark.read.csv('../Data/age_gender_bkts.csv',
                       header=True,
                       multiLine=True,
                       inferSchema=True)

    countries = spark.read.csv('../Data/countries.csv',
                       header=True,
                       multiLine=True,
                       inferSchema=True)

    sessions = spark.read.csv('../Data/sessions.csv',
                       header=True,
                       multiLine=True,
                       inferSchema=True)

    return train, test, age_gender_bkts, countries, sessions


########################################################
def combine_train_test(train, test,col_diff_list,test_col_drop = None):
    
    for col in col_diff_list:
        test = test.withColumn(col, F.lit(None).cast(StringType()))
        
    if (test_col_drop == None):    
        df = train.unionByName(test)
    else:
        df = train.unionByName(test.drop(test_col_drop))
 
    return df


########################################################
def print_missing(df):
    
    for i in df.columns:
        print(i, df.where(F.col(i).isNull()).count())
    
    return 0


########################################################
def parse_date(df, cols):
    
    for col in cols:
        df = df.withColumn(col+'year', F.year(F.col(col)).cast(IntegerType()))
        df = df.withColumn(col+'month', F.month(F.col(col)).cast(IntegerType()))
        df = df.withColumn(col+'dayofmonth', F.dayofmonth(F.col(col)).cast(IntegerType()))
        df = df.withColumn(col+'dayofweek', F.dayofweek(F.col(col)).cast(IntegerType()))
        df = df.drop(col)
        
    return df


#########################################################
def get_datefromtimestamp_first_active(df):
    
    df = df.withColumn('timestamp_first_active', df['timestamp_first_active'].cast(StringType()))
    df = df.withColumn('first_active_year', df.timestamp_first_active.substr(0,4).cast(IntegerType()))
    df = df.withColumn('first_active_month', df.timestamp_first_active.substr(5,2).cast(IntegerType()))
    df = df.withColumn('first_active_day', df.timestamp_first_active.substr(7,2).cast(IntegerType()))
    df = df.drop('timestamp_first_active')
    
    return df


########################################################
def encoding(df,incol,outcol):
    
    encoder = OneHotEncoderEstimator(
        inputCols= [incol],
        outputCols=[outcol]
    )
    encoder = encoder.fit(df)
    df = encoder.transform(df)
    
    return df


##########################################################
def one_hot_col(df,col):
    
    indexer = StringIndexer(inputCol= col, outputCol=col+"Indexed")
    model = indexer.fit(df).transform(df)
    encoded = (
        OneHotEncoder(inputCol= col+"Indexed", outputCol= col + "Vec")
        .transform(model)
    )
    
    return encoded


########################################################
def one_hot_encoding(df, categorical_features):
    
    indexers = [StringIndexer
            (inputCol=column, outputCol=column+"_index")
            .fit(df) for column in categorical_features
           ]


    pipeline = Pipeline(stages=indexers)
    df = pipeline.fit(df).transform(df)
    
    for i in range(len(categorical_features)):
        categorical_features[i] = categorical_features[i] +"_index"
    
    for col in categorical_features:
        df = encoding(df, incol = col, outcol = col+"onehot")
    
    
    return df


###############################
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(20, 18))
    
    sns.heatmap(
        df.corr(), 
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    


########################################
def build_models(i,labelCol,features_list, train, test):
    
    model_list = [
      LogisticRegression(featuresCol='features', labelCol = labelCol),
      RandomForestClassifier(featuresCol='features', labelCol = labelCol),
      NaiveBayes(featuresCol='features',labelCol=labelCol), 
      GBTClassifier(featuresCol='features',labelCol=labelCol)
    ]
    assembler = VectorAssembler(inputCols=features_list, outputCol='features')
    pipeline = Pipeline(stages=[assembler, model_list[i]])
    model = pipeline.fit(train)
    prediction = model.transform(test)
    
    return prediction


########################################    
def type_to_integer(df, split_col):
    for col in df.drop(split_col).columns:
        df = df.withColumn(col, df[col].astype('int'))
    
    return df



######################################
def pred_precision_kaggle(prediction,NumCluster):
    
    pred_label = prediction.rdd.map(lambda
                                    x: (float(np.argsort(-1*x.prediction)[:1]),
                                        float((x.country_destination_indexed))))
    metrics = MulticlassMetrics(pred_label)
    avg_precision = metrics.precision()
    
    for i in range(1,NumCluster):
        pred_label = prediction.rdd.map(lambda
                                        x: (float(np.argsort(-1*x.probability)[i:(i+1)]),
                                            float(x.country_destination_indexed)))
        metrics = MulticlassMetrics(pred_label)
        avg_precision += metrics.precision()
        
    return avg_precision


##########################################



#            Pandas


######################################
def get_data_pd():
    
    train = pd.read_csv('../Data/train_users_2.csv',
                        parse_dates=['date_account_created','date_first_booking'])
    test = pd.read_csv('../Data/test_users.csv',
                        parse_dates=['date_account_created','date_first_booking'])
    
    countries = pd.read_csv('../Data/countries.csv')
    age_gender_bkts = pd.read_csv('../Data/age_gender_bkts.csv')
    
    
    return train, test, countries, age_gender_bkts