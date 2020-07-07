import pyspark
from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *

spark = SparkSession.builder.master("local[2]").appName("jessica_ai").config("spark.driver.memory", "50g").config("spark.driver.maxResultSize", "50g").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)
spark.sparkContext._conf.getAll()
