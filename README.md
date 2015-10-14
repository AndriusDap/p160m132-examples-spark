# pySpark pavyzdžiai

**_Pastaba:_** pasidaryti repozitorijos _Fork_'ą(https://confluence.atlassian.com/bitbucket/forking-a-repository-221449527.html). Tuomet vietoje _daraliu/p160m132-examples-spark_ repozitorijos pavadinimas taps _studentoBitbucketVartotojoVardas/p160m132-examples-spark_. Tuomet ją reikėtų klonuoti į savo _Ubuntu Linux_ virtualią mašiną, pvz. `/home/vagrant/labs` direktoriją.

## examples_datafiles.ipynb

- `.xls`, `.xlsx`, `.sas7bdat` formato failų skaitymas ir konvertavimas į `.csv` formato failus naudojant _Python_.

- `.csv` formato failų skaitymas ir rašymas naudojant _Apache Spark_.

## examples_kmeans.ipynb

- K-vidurkių modelio apmokymas panaudojant [`pyspark.mllib.clustering.KMeans`](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.clustering.KMeans) ir [`pyspark.clustering.KMeansModel`](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.clustering.KMeansModel).

- K-vidurkių modelio apmokymas panaudojant [`pyspark.ml.clustering.KMeans`](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.clustering.KMeans) ir naudojant [`pyspark.ml.pipeline.Pipeline`](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Pipeline) bei kitus [`pyspark.ml`](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html) komponentus.