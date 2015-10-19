
# coding: utf-8

# # Klasterizavimas.  K-vidurkių metodas
# ## _Apache Spark Mllib KMeans_

# Šiame bloknote duomenis klasterizuosime naudodami K-vidurkių metodą. _Apache Spark_ šiuo metu yra 2 mašinų mokymo bibliotekos: `pyspark.ml` ir `pyspark.mllib`.
# 
# `pyspark.ml` yra naujesnė biblioteka, joje naudojami `DataFrame` objektai, tačiau šiuo metu joje yra ne visi `pyspark.mllib` esantys modeliai. `ml` bibliotekos [projektavimo dokumente](https://docs.google.com/document/d/1rVwXRjWKfIb-7PI6b86ipytwbUH7irSNLF1_6dLmh8o/edit#) rašoma, kad kai iš `mllib` bibliotekos modeliai bus pilnai perkelti į `ml` biblioteka, tik pastaroji ir bus naudojama:
# 
# 
# >MLlib now covers a basic selection of machine learning algorithms, e.g., logistic regression, decision trees, alternating least squares, and k-means. The current set of APIs contains several design flaws that prevent us moving forward to address practical machine learning pipelines, make MLlib itself a scalable project.
# 
# >The new set of APIs will live under org.apache.spark.ml, and org.apache.spark.mllib will be deprecated once we migrate all features to org.apache.spark.ml.
# 
# Kol taip nėra, priklausomai nuo uždavinių, naudosime tiek `mllib`, tiek `ml` bibliotekas.

# ### 1 pavyzdys
# 
# #### CA1 duomenų failas

# Šiame pavyzdyje paprastumo dėlei naudosime $\mathbb{R}^2$ erdvės duomenis.

# Konvertuojame _MS Excel_ `.xls` formato failą į `.csv` formato failą.

# In[1]:

get_ipython().system(' in2csv --sheet "CA1" data/CA1.xls > data/CA1.csv')


# Išvedame failo pradžią su stulpelių pavadinimais.

# In[2]:

get_ipython().system(' head data/CA1.csv')


# Komandos `head` parinktis `-n <skaičius>` nuo failo pradžios nuskaitomų eilučių skaičių:

# In[3]:

get_ipython().system(' head -n 5 data/CA1.csv')


# Kaip ir daugumai _Linux_ komandų, `head` parinktis `--help` išveda naudojimo instrukciją.

# In[4]:

get_ipython().system(' head --help')


# Įkeliame `.csv` formato duomenų failą. Įkėlimo metu galime pakeisti stulpelių pavadinimus naudodami `pyspark.sql.DataFrame` metodą `withColumnRenamed` arba iš įkelto `pyspark.sql.DataFrame` gauti `RDD` (`pyspark.sql.DataFrame` atributas `rdd`) ir vėl paversti į `pyspark.sql.DataFrame` panaudojant `RDD` metodą `toDF` su naujų stulpelių pavadinimų sąrašo argumentu.

# In[69]:

ca1DF = (
    sqlContext.read.load("data/CA1.csv", format="com.databricks.spark.csv", header=True, inferSchema=True)
    .rdd.toDF(["id", "x", "y", "cluster"])
).cache()

ca1DF.show(5)


# arba

# In[70]:

ca1DF = (
    sqlContext.read.load("data/CA1.csv", format="com.databricks.spark.csv", header=True, inferSchema=True)
    .withColumnRenamed("ID", "id")
    .withColumnRenamed("X", "x")
    .withColumnRenamed("Y", "y")
    .withColumnRenamed("klasteris", "cluster")
).cache()

ca1DF.show(5)


# Importuojame _Apache Spark Mllib_ K-vidurkių klasteriavimo klasę ir vektorių klasę.

# In[7]:

from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import DenseVector


# Pamename, kad `pyspark.sql.DataFrame` atributo `rdd` reikšmė `pyspark.sql.Row` tipo objektų `RDD` objektas:

# In[8]:

ca1DF.rdd.take(5)


# `pyspark.mllib` paketas yra pritaikytas dirbti su `RDD` tipo objektais, todėl iš turimo `pyspark.sql.DataFrame` sudarome požymių `RDD` iš `x` ir `y` stulpelių. 

# In[9]:

featuresRDD = ca1DF.rdd.map(lambda row: DenseVector([row.x, row.y]))
featuresRDD.take(5)


# Parenkame parametrą $k=2$ ir apmokome pirmą K-vidurkių modelį. `pyspark.mllib.clustering.KMeans` metodo `train` argumentų paaiškinimas yra [čia](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html?highlight=mllib%20kmeans#pyspark.mllib.clustering.KMeans).

# In[10]:

firstModel = KMeans.train(
    featuresRDD, k=2, maxIterations=20, 
    runs=10, initializationMode="k-means||")


# Apmokyto modelio klasterių centrus galime pasiekti per modelio (kurio tipas `pyspark.mllib.clustering.KMeansModel`) atributą `centers`:

# In[11]:

type(firstModel)


# In[12]:

firstModel.centers


# Klasterių prasmę patogiausia interpretuoti pagal jų centrų koordinates.

# `pyspark.mllib.clustering.KMeansModel` metodas `computeCost` apskaičiuoja stebėjimų Euklido atstumų nuo savo klasterių  centrų sumą $S_K$ (angl. _Within Set Sum of Squared Error (WSSSE)_):
# 
# $I_k = \sum_{\mathbf{x}_i \in C_k} \| \mathbf{x}_i - \mathbf{\overline{x}}_k \|$
# 
# $S_K = \sum_{k}^{K} I_k$
# 
# čia 
# 
# $k$ - klasterio indeksas,
# 
# $C_k$ - $k$-asis klasteris
# 
# $K$ - klasterių skaičius,
# 
# $N_k$ - $k$-jam klasteriui priklausančių stebėjimų skaičius,
# 
# $\mathbf{x_i}$ - $i$-tojo stebėjimo vektorius,
# 
# $\mathbf{\overline{x}}_k$ - $k$-otojo klasterio vidurinio taško (centro) vektorius,
# 
# $\|\mathbf{x}\|$ - vektoriaus Euklido norma, t.y. kvadratinė šaknis iš jo komponenčių kvadratų sumos.

# In[13]:

firstModel.computeCost(featuresRDD)


# * * *
# **_Pastaba:_** $S_K$ galime apskaičiuoti ir patys:

# In[14]:

featuresRDD.map(lambda point: point.squared_distance(firstModel.centers[firstModel.predict(point)])).sum()


# Aiškumo dėlei išskaidome šią išraišką į keletą funkcijų:

# In[15]:

def computeWSSSE(vectorRDD, model):
    return vectorRDD.map(lambda point: squaredDistance(point, model)).sum()

def squaredDistance(point, kmeansModel):
    center = kmeansModel.centers[kmeansModel.predict(point)]
    return point.squared_distance(center)


# In[16]:

computeWSSSE(featuresRDD, firstModel)


# 
# * * *

# Modelio parametrą $K = 2$ parinkome intuityviai, tačiau ar galime vien iš šios metrikos pasakyti ar tai optimali $K$ reikšmė?

# Kadangi naudojame $\mathbb{R}^2$ duomenis, nubraižysime sklaidos diagramą. Grafikų braižymui naudosime [_R_](https://www.r-project.org/) paketo [`ggplot2`](http://docs.ggplot2.org/) _Python_ atitikmenį [`ggplot`](http://ggplot.yhathq.com/).
# 
# Pirmiausia prie turimų stebėjimų prijungsime ką tik apmokyto modelio prognozuojamų klasterių indeksus.

# In[17]:

from pyspark.sql import Row


# Naudodami apmokytą modelį atliekame stebėjimų priklausymo klasteriams prognozavimą ir patogumo dėlei sukuriame naują `pyspark.sql.DataFrame` su prognozavimo rezultatais.

# Tai galima atlikti naudojant vieną išraišką, todėl užtenka vienos anoniminės funkcijos, kurią perduodame `RDD` metodui `map`:

# In[18]:

ca1WithPredictionDF = (
    ca1DF.rdd.map(lambda r: Row(*(tuple(r) + (firstModel.predict(DenseVector([r.x, r.y])),))))
).toDF(ca1DF.columns + ["prediction"])


# Kita vertus, išraiška `Row(*(tuple(r) + (bestModel.predict(DenseVector([r.x, r.y])),)))` nėra labai lengvai skaitoma bei joje naudojame konkrečius stulpelių pavadinimus `x` ir `y`. Šios išraišką galime iškelti į atskirą funkciją. Funkcijoje taip pat naudosime požymių stulpelių pavadinimus, o tai leis funkciją naudoti duomenų rinkiniams su bet kokiais stulpelių pavadinimais.

# In[19]:

def addPredictionColumn(row, model, featureColumns):
    rowDict = row.asDict()
    features = DenseVector([rowDict[c] for c in featureColumns])
    predictedValue = model.predict(features)
    allValues = tuple(row) + (predictedValue,)
    return Row(*allValues)

ca1WithPredictionDF = (
    ca1DF.rdd.map(lambda row: addPredictionColumn(row, firstModel, ["x", "y"]))
).toDF(ca1DF.columns + ["prediction"])


# Pastebime, kad `RDD` metodo `toDF` argumente pradinio `pyspark.sql.DataFrame` stulpelius apjungiame su prognozavimo stulpeliu, kurį pavadiname `prediction`. Tokiu būdu mums nereikia rankiniu būdu nurodyti pradinių stulpelių pavadinimų.

# In[20]:

ca1WithPredictionDF.take(10)


# Klasterių indeksai stulpelyje `prediction` atitinka klasterio centrų indeksus:

# In[21]:

list(enumerate(firstModel.centers))


# _Python_ `ggplot` paketas naudoja _Python_ `pandas` paketo duomenų rinkinius `pandas.DataFrame`. Iškvietus _Apache Spark_ `pyspark.sql.DataFrame` metodą `toPandas` gauname `pandas.DataFrame` objektą kurį naudojame sklaidos diagramos braižymui.

# In[22]:

# grafikų braižymui Jupyter Notebook ląstelėse
get_ipython().magic('matplotlib inline')
import ggplot as gg


# In[23]:

(
    gg.ggplot(gg.aes(x="x", y="y", color="prediction"), data=ca1WithPredictionDF.toPandas()) + 
    gg.geom_point() + 
    gg.ggtitle("K = 2")
)


# Matome, kad $K = 2$ tikrai nėra optimalus klasterių kiekis. $\mathbb{R}^2$ atveju galime nubraižyti sklaidos diagramą, tačiau ką daryti $\mathbb{R}^p$, kai $p > 2$ atveju? 
# Vienas iš būdų yra apmokyti keletą modelių su skirtingomis $K$ reikšmėmis ir juos palyginti pagal iš anksto apsibrėžtų charakteristikų reikšmes.

# Apmokome keletą modelių su skirtingomis $K$ reikšmėmis ir apskaičiuojame jų $S_K$ reikšmes.

# In[24]:

kValues = [2, 3, 4, 5, 6, 7]
models = [KMeans.train(featuresRDD, k=k) for k in kValues]
WSSSEs = [m.computeCost(featuresRDD) for m in models]

rowsWSSSSE = list(zip(kValues, map(float, WSSSEs)))
WSSSEDF = sc.parallelize(rowsWSSSSE).toDF(["K", "WSSSE"])


# Aiškumo dėlei išvedame kiekvieno kintamojo reikšmes

# In[25]:

print("K reikšmės")
print(kValues)
print("\nModelių objektai")
print(models)
print("\nS_k reikšmės")
print(WSSSEs)
print("\nRezultatų eilutės su K ir S_k reikšmėmis")
print(rowsWSSSSE)
print("\npyspark.sql.DataFrame turinys su rezultatais")
WSSSEDF.show()


# Galime nubraižyti $S_K$ priklausomybės nuo $K$ grafiką.

# In[26]:

gg.ggplot(gg.aes(x="K", y="WSSSE"), data=WSSSEDF.toPandas()) + gg.geom_line()


# Aiškiai matome, kad $K = 2$ nėra optimali reikšmė.

# Nubraižome grafiką su reikšmėmis $K > 2$

# In[27]:

gg.ggplot(gg.aes(x="K", y="WSSSE"), data=WSSSEDF.where(WSSSEDF["K"] > 2).toPandas()) + gg.geom_line()


# Su $K = 4$ matomas ženklus pagerėjimas. $K=5$ taip pat suteikia pastebimą pagerėjimą. Sprendžiant tikrą uždavinį tokiu atveju interpretuotume klasterių centrus su $K = 4$ ir $K = 5$, tuomet parinktume prasmingesnę $K$ reikšmę.

# $K = 4$ centrai

# In[28]:

models[kValues.index(4)].centers


# $K = 5$ centrai

# In[29]:

models[kValues.index(5)].centers


# Tolesniam darbui galime sukurti funkciją, apmokančią keletą K-vidurkių modelių su nurodytomis $K$ reikšmėmis ir grąžinančią WSSSE reikšmę kiekvienai $K$ reikšmei. Joje patalpinsime jau naudotus teiginius (angl. _statement_). Patogumo dėlei panaudodami  `collections.namedtuple` sukursime rezultatų klasę. `namedtuple` objektą galima naudoti ir kaip paprastą `tuple`, ir kaip objektą.

# In[30]:

import collections


# In[31]:

kmeansWSSSEResults = collections.namedtuple("KmeansWSSSEResults", ["ks", "WSSSEs", "models"])

def kmeansWSSSEsByK(featuresRDD, kValues):
    models = [KMeans.train(featuresRDD, k=k) for k in kValues]
    WSSSEs = [m.computeCost(featuresRDD) for m in models]
    return kmeansWSSSEResults(kValues, WSSSEs, models)


# Rezultatų priskyrimas atskiriems kintamiesiems naudojant `tuple` „išpakavimą“ (angl. _tuple unpacking_):

# In[32]:

ks, wssses, models = kmeansWSSSEsByK(featuresRDD, [2, 3, 4, 5])


# In[33]:

print(ks, "\n")
print(wssses, "\n")
print(models, "\n")


# Rezultatų priskyrimas vienam kintamajam:

# In[34]:

results = kmeansWSSSEsByK(featuresRDD, [2, 3, 4, 5])


# In[35]:

print(results.ks, "\n")
print(results.WSSSEs, "\n")
print(results.models, "\n")


# Kadangi turime $\mathbb{R^2}$ duomenis, galime nubraižyti sklaidos diagramas su $K=4$ ir $K=5$

# In[36]:

ca1K4PredictionDF = (
    ca1DF.rdd.map(lambda row: addPredictionColumn(row, models[ks.index(4)], ["x", "y"]))
).toDF(ca1DF.columns + ["prediction"])

(
    gg.ggplot(gg.aes(x="x", y="y", color="prediction"), data=ca1K4PredictionDF.toPandas()) + 
    gg.geom_point() + 
    gg.ggtitle("K = 4")
)


# In[37]:

ca1K5PredictionDF = (
    ca1DF.rdd.map(lambda row: addPredictionColumn(row, models[ks.index(5)], ["x", "y"]))
).toDF(ca1DF.columns + ["prediction"])

(
    gg.ggplot(gg.aes(x="x", y="y", color="prediction"), data=ca1K5PredictionDF.toPandas()) + 
    gg.geom_point() + 
    gg.ggtitle("K = 5")
)


# ## 2 pavyzdys

# Kita iš galimų charakteristikų nustatyti optimalią $K$ reikšmę yra aprašyta [čia](http://www.ee.columbia.edu/~dpwe/papers/PhamDN05-kmeans.pdf) ir [čia](https://datasciencelab.wordpress.com/2014/01/21/selection-of-k-in-k-means-clustering-reloaded/). Jos reikšmė $f(K)$ apskaičiuojama tokiu būdu:
# 
# <img src="https://datasciencelab.files.wordpress.com/2014/01/fk.png?w=359&h=192">
# 
# Optimilaus $K$ yra ties "alkūnine" $f(K)$ reikšme.

# Pastebime, kad $\alpha_K$ yra rekurentinis sąryšis.

# Pagal $f(K)$ išraišką matome, kad kiekvienam $K$ reikia turėti ir $S_{K-1}$ reišmes. Sukuriame reikalingas funkcijas.

# In[38]:

import collections
Pham2004Results = collections.namedtuple("Pham2004Results", ["ks", "fKs", "models"])

def computePham2004(featuresRDD, kValues):
    allKs = makeAllKs(kValues)
    _, WSSSEs, models = kmeansWSSSEsByK(featuresRDD, allKs)
    
    kToWSSSEMap = dict(zip(allKs, WSSSEs))
    kToModelMap = dict(zip(allKs, models))
    
    ksWithPairedWSSSEs = [
        (k, kToWSSSEMap[k], kToWSSSEMap.get(k - 1, 0),)
        for k in kValues]
    dimension = len(featuresRDD.first())
    
    fKs = [
        computeFK(k, Sk, prevSk, dimension) 
        for (k, Sk, prevSk) in ksWithPairedWSSSEs]
    return Pham2004Results(kValues, fKs, [kToModelMap[k] for k in kValues])

def makeAllKs(initialKs):
    allKs = [k - 1 for k in initialKs if k - 1 > 0] + initialKs
    return sorted(list(set(allKs)))

def computeFK(k, SSE, prevSSE, dim):
    if k == 1 or prevSSE == 0:
        return 1
    weight = weightFactor(k, dim)
    return SSE / (weight * dim)

# calculating alpha_k in functional style with tail recursion -- which is not optimized in Python :(
def weightFactor(k, dim):
    if not k >= 2:
        raise ValueError("k should be greater than 1.")
        
    def weightFactorAccumulator(acc, k):
        if k == 2:
            return acc
        return weightFactorAccumulator(acc + (1 - acc) / 6, k - 1)
        
    k2Weight = 1 - 3 / (4 * dim)
    return weightFactorAccumulator(k2Weight, k)


# $\alpha_K$ skaičiavome naudodami rekursiją (funkcija `weightFactor`). Deja, _Python_ programavimo kalboje „uodegos rekursija“ (angl. _tail recursion_) [nėra optimizuota](http://kachayev.github.io/talks/uapycon2012/#/46) ir funkcija neveiks kai $K > 975$. Kai $K > 975$, _Python_'e $\alpha_K$ reikėtų skaičiuoti imperatyviai:

# In[39]:

def weightFactor(k, dim):
    if not k >= 2:
        raise ValueError("k must be greater than 1.")
        
    weight = 1 - 3 / (4 * dim)
    i = 2
    while i < k:
        weight_prev = weight
        weight = weight_prev + (1 - weight_prev) / 6
        i += 1
    return weight


# Apskaičiuojame $f(K)$ statistikos reikšmes, kai $K \in \{1, 2, 3, 6, 9\}$:

# In[40]:

ks, fKs, _ = computePham2004(featuresRDD, [1, 2, 3, 6, 9])
print(ks)
print(fKs)


# Sukuriame funkciją, kuri iš dviejų sąrašų ($K$ reikšmių ir $f(K)$ reikšmių) sukurs `pyspark.sql.DataFrame`:

# In[41]:

Pham2004Row = Row("K", "fK")

def makePham2004DF(ks, fKs):
    rowContents = list(zip(ks, fKs))
    return sc.parallelize(
        [Pham2004Row(k, float(fK))
         for (k, fK) in rowContents]).toDF()


# In[42]:

pham2004DF = makePham2004DF(ks, fKs)
pham2004DF.show()


# Nubraižome $f(K)$ pagal $K$ grafiką.

# In[43]:

gg.ggplot(gg.aes(x="K", y="fK"), data=pham2004DF.toPandas()) + gg.geom_line() + gg.ylab("f(K)")


# Matome, kad „alkūnė“ yra kai $K = 3$, tačiau iš lentelės matome, kad $f(K)$ skirtumas tarp $K = 3$ ir $K = 6$ taip pat žymus. Taip galima matyti ir iš grafiku su $K >= 3$ reikšmėmis:

# In[44]:

pham2004DF.where(pham2004DF["K"] > 2).show()


# In[45]:

gg.ggplot(gg.aes(x="K", y="fK"), data=pham2004DF.where(pham2004DF["K"] > 2).toPandas()) + gg.geom_line() + gg.ylab("f(K)")


# Parinkime daugiau $K$ reikšmių itervale $[3; 9]$.

# In[46]:

moreKs = [3,4,5,6,7,8,9]

_, morefKs, moreModels = computePham2004(featuresRDD, moreKs)
pham2004moreKsDF = makePham2004DF(moreKs, morefKs)
pham2004moreKsDF.show()


# In[47]:

gg.ggplot(gg.aes(x="K", y="fK"), data=pham2004moreKsDF.toPandas()) + gg.geom_line() + gg.ylab("f(K)")


# Matome, kad „alkūninė“ reikšmė yra kai $K=4$, tačiau ir su $K=5$ turi matomas $k(K)$ sumažėjimas.

# Nubraižome sklaidos diagramas šiems atvejams.

# In[48]:

kToModelMap = dict(zip(moreKs, moreModels))
kToModelMap


# In[57]:

ca1K4WithPredDF = (
    ca1DF.rdd.map(lambda row: addPredictionColumn(row, kToModelMap[4], ["x", "y"]))
).toDF(ca1DF.columns + ["prediction"])

gg.ggplot(gg.aes(x="x", y="y", color="prediction"), data=ca1K4WithPredDF.toPandas()) + gg.geom_point() + gg.ggtitle("K = 4")


# In[58]:

ca1K5WithPredDF = (
    ca1DF.rdd.map(lambda row: addPredictionColumn(row, kToModelMap[5], ["x", "y"]))
).toDF(ca1DF.columns + ["prediction"])

gg.ggplot(gg.aes(x="x", y="y", color="prediction"), data=ca1K5WithPredDF.toPandas()) + gg.geom_point() + gg.ggtitle("K = 5")


# ## 3 Pavyzdys

# Šį kartą naudosime `pyspark.ml` paketą. K-vidurkių modelių sudarymui naudosime [`pyspark.ml.clustering.Kmeans`](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.clustering), duomenų paruošimui naudosime [`pyspark.ml.feature.VectorAssembler`](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler) ir gausime [`pyspark.ml.PipelineModel`](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.PipelineModel) objektą, kuriuo galėsime atlikti prognozes iš pradinių duomenų.

# Nuskaitome duomenų failą į `pyspark.sql.DataFrame`:

# In[72]:

ca1MlDF = (
    sqlContext.read.load("data/CA1.csv", format="com.databricks.spark.csv", header=True, inferSchema=True)
    .rdd.toDF(["id", "x", "y", "cluster"])
).cache()

ca1MlDF.show(5)


# Sukuriame [`pyspark.ml.feature.VectorAssembler`](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler), kuriuo iš nurodytų stulpelių (`x` ir `y`) sukursime modeliui reikalingą  vektorių stulpelį:

# In[60]:

from pyspark.ml.feature import VectorAssembler

vecAssembler = VectorAssembler(inputCols=["x", "y"], outputCol="features")
type(vecAssembler)


# Sukurtas `vecAssembler` objektas (kurio tipas yra `pyspark.ml.feature.VectorAssembler`) turi `transform` metodą, kuriuo iš stulpelių `x` ir `y` bus pagamintas vektorių stulpelis `features` ir iš jam perduodamo `DataFrame` objekto bus sukurtas naujas `DataFrame` su `features` stulpeliu:

# In[61]:

ca1mlFeaturizedDF = vecAssembler.transform(ca1MlDF)
ca1mlFeaturizedDF.show(5)


# Sukursime `pyspark.ml.clustering.KMeans` objektą (atminkime, kad 1 pavyzdyje naudojome `pyspark.mllib.clustering.KMeans`), kuriuo apmokysime `pyspark.ml.clustering.KMeansModel`. Duomenis pateiksime nebe `RDD`, o `DataFrame` objektu.

# In[62]:

from pyspark.ml.clustering import KMeans as MlKMeans

firstMlKMeans = MlKMeans(
    featuresCol="features", predictionCol="prediction", k=2, 
    initMode="k-means||", maxIter=20)
type(firstMlKMeans)


# `pyspark.ml` paketo modelių klasės turi `explainParams` metodą, kuruo išvedami modelio parametrų paaiškinimai.

# In[63]:

print(firstMlKMeans.explainParams())


# Apmokykime modelį.

# In[64]:

firstMlModel = firstMlKMeans.fit(ca1mlFeaturizedDF)
type(firstMlModel)


# In[65]:

firstMlModel.clusterCenters()


# Sudarome `Pipeline` žingsnių seką iš `vecAssembler` ir `kmeans` komponentų.

# In[66]:

from pyspark.ml.pipeline import Pipeline

firstPipeline = Pipeline(stages=[vecAssembler, firstMlKMeans])


# In[67]:

firstPipelineModel = firstPipeline.fit(ca1MlDF)


# In[73]:

firstPipelineModel.transform(ca1MlDF).show(5)


# In[74]:

MlKmeansWSSSEResults = collections.namedtuple("MlKmeansWSSSEResults", ["ks", "WSSSEs", "pipelineModels"])

def mlKmeansWSSSEsByK(initialDF, kValues):
    vecAssembler = VectorAssembler(inputCols=["x", "y"], outputCol="features")
    pipelineModels = [Pipeline(stages=[vecAssembler, MlKMeans(k=k)]).fit(initialDF) 
                      for k in kValues]
    mlWSSSEs = [computeMlWSSSE(initialDF, pm) for pm in pipelineModels]
    return MlKmeansWSSSEResults(kValues, mlWSSSEs, pipelineModels)

def computeMlWSSSE(initialDF, pipelineModel):
    centers = pipelineModel.stages[-1].clusterCenters()
    predictionDF = pipelineModel.transform(initialDF)
    
    squaredDistancesRDD = predictionDF.rdd.map(
        lambda row: row.features.squared_distance(centers[row.prediction]))
    return squaredDistancesRDD.sum()


# In[75]:

computeMlWSSSE(ca1MlDF, firstPipelineModel)


# Naudosime $f(K)$ charakteristiką.

# In[76]:

MlPham2004Results = collections.namedtuple("Pham2004Results", ["ks", "fKs", "pipelineModels"])

def computeMlPham2004(initialDF, kValues):
    allKs = [k for k in makeAllKs(kValues) if k > 1]
    _, WSSSEs, pipelineModels = mlKmeansWSSSEsByK(initialDF, allKs)
    
    kToWSSSEMap = dict(zip(allKs, WSSSEs))
    kToPipelineModelMap = dict(zip(allKs, pipelineModels))
    
    ksWithPairedWSSSEs = [
        (k, kToWSSSEMap.get(k, 0), kToWSSSEMap.get(k - 1, 0),)
        for k in kValues]
    dimension = len(featuresRDD.first())
    
    fKs = [
        computeFK(k, Sk, prevSk, dimension) 
        for (k, Sk, prevSk) in ksWithPairedWSSSEs]
    return Pham2004Results(kValues, fKs, [kToPipelineModelMap.get(k) for k in kValues])


# In[82]:

ks, fKs, pipelineModels = computeMlPham2004(ca1MlDF, [3, 4, 5, 6])


# In[83]:

rowsWSSSE = list(zip(ks, map(float, fKs)))
WSSSEDF = sc.parallelize(rowsWSSSE).toDF(["K", "fK"])
WSSSEDF.show()


# In[84]:

gg.ggplot(gg.aes(x="K", y="fK"), data=WSSSEDF.toPandas()) + gg.geom_line()


# ## 3 pavyzdys

# Šiame pavyzdyje naudosime `pyspark.ml` paketą, $S_k$ charakteristiką ir laikysime, kad neturime anksčiau rašyto kodo.

# In[85]:

ls data


# In[86]:

get_ipython().system(' in2csv data/CA2.xls > data/CA2.csv')


# In[87]:

get_ipython().system(' head data/CA2.csv')


# In[88]:

ca2DF = (
    sqlContext.read.load("data/CA2.csv", format="com.databricks.spark.csv", header=True, inferSchema=True)
    .rdd.toDF(["id", "x", "y", "cluster"])
)


# In[89]:

from pyspark.ml.clustering import KMeans as MlKMeans
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.feature import VectorAssembler
import collections

MlKmeansWSSSEResults = collections.namedtuple("MlKmeansWSSSEResults", ["ks", "WSSSEs", "pipelineModels"])

vecAssembler = VectorAssembler(inputCols=["x", "y"], outputCol="features")

def mlKmeansWSSSEsByK(initialDF, kValues):
    pipelineModels = [Pipeline(stages=[vecAssembler, MlKMeans(k=k)]).fit(initialDF) 
                      for k in kValues]
    mlWSSSEs = [computeMlWSSSE(initialDF, pm) for pm in pipelineModels]
    return MlKmeansWSSSEResults(kValues, mlWSSSEs, pipelineModels)

def computeMlWSSSE(initialDF, pipelineModel):
    centers = pipelineModel.stages[-1].clusterCenters()
    predictionDF = pipelineModel.transform(initialDF)
    
    squaredDistancesRDD = predictionDF.rdd.map(
        lambda row: row.features.squared_distance(centers[row.prediction]))
    return squaredDistancesRDD.sum()


# In[90]:

ca2ks, ca2WSSSEs, ca2PiplelineModels = mlKmeansWSSSEsByK(ca2DF, [2, 4, 6, 8])


# In[91]:

print(ca2ks)
print(ca2WSSSEs)
print(ca2PiplelineModels)


# In[92]:

rowsCa2WSSSE = list(zip(ca2ks, map(float, ca2WSSSEs)))
rowsCa2WSSSE


# In[93]:

ca2WSSSEDF = sc.parallelize(rowsCa2WSSSE).toDF(["K", "WSSSE"])
ca2WSSSEDF.show()


# In[94]:

gg.ggplot(gg.aes(x="K", y="WSSSE"), data=ca2WSSSEDF.toPandas()) + gg.geom_line()


# In[95]:

zoomedCa2ks, zoomedCa2WSSSEs, zoomedCa2PiplelineModels = mlKmeansWSSSEsByK(ca2DF, [3, 4, 5, 6])
zoomedWSSSEDF = sc.parallelize(list(zip(zoomedCa2ks, map(float, zoomedCa2WSSSEs)))).toDF(["K", "WSSSE"])
zoomedWSSSEDF.show()


# In[96]:

gg.ggplot(gg.aes(x="K", y="WSSSE"), data=zoomedWSSSEDF.toPandas()) + gg.geom_line()


# In[97]:

kToPipelineModelMap = dict(zip(zoomedCa2ks, zoomedCa2PiplelineModels))
kToPipelineModelMap


# In[98]:

K = 4
pipelineModel = kToPipelineModelMap[K]
transformedDF = pipelineModel.transform(ca2DF)
(
    gg.ggplot(gg.aes(x="x", y="y", color="prediction"), data=transformedDF.toPandas()) + 
    gg.geom_point() + 
    gg.ggtitle("K = {}".format(K))
)


# In[99]:

K = 5
(
    gg.ggplot(gg.aes(x="x", y="y", color="prediction"), 
              data=kToPipelineModelMap[K].transform(ca2DF).toPandas()) + 
    gg.geom_point() + 
    gg.ggtitle("K = {}".format(K))
)


# ## 4 Pavyzdys

# Šiame pavyzdyje naudosime `pyspark.ml` paketą, tačiau rezultatų braižymui iškart sukursime `pandas.DataFrame`, o ne `pyspark.sql.DataFrame` objektą.

# In[100]:

get_ipython().system(' head data/CA2.csv')


# In[101]:

csvFilename = "data/CA2.csv"

initialDF = (
    sqlContext.read.load(csvFilename, format="com.databricks.spark.csv", header=True, inferSchema=True)
    .rdd.toDF(["id", "x", "y", "cluster"])
)


# In[102]:

kValues = [3, 4, 5, 6, 7]
ks, WSSSEs, pipelineModels = mlKmeansWSSSEsByK(initialDF, kValues)


# In[103]:

rowsWSSSSE = list(zip(kValues, map(float, WSSSEs)))
rowsWSSSSE


# In[104]:

import pandas as pd


# In[105]:

WSSSEPdDF = pd.DataFrame(rowsWSSSSE, columns=["K", "WSSSE"])
WSSSEPdDF


# In[106]:

import ggplot as gg


# In[107]:

get_ipython().magic('matplotlib inline')
gg.ggplot(gg.aes(x="K", y="WSSSE"), data=WSSSEPdDF) + gg.geom_line()


# In[108]:

bestK = 4
bestPipelineModel = pipelineModels[kValues.index(bestK)]


# In[109]:

bestPipelineModel.stages[-1].clusterCenters()


# In[110]:

transformedDF = bestPipelineModel.transform(initialDF)
transformedDF.show(5)


# In[111]:

get_ipython().magic('matplotlib inline')
gg.ggplot(gg.aes(x="x", y="y", color="prediction"), data=transformedDF.toPandas()) + gg.geom_point()


# ## 5 pavyzdys

# Šiame pavyzdyje laikysime, kad neturime anksčiau rašyto kodo, naudosime `pyspark.mllib` paketą, $S_k$ charakteristiką, tačiau rezultatų braižymui iškart sukursime `pandas.DataFrame`, o ne `pyspark.sql.DataFrame` objektą.

# In[112]:

csvFilename = "data/CA2.csv"

initialDF = (
    sqlContext.read.load(csvFilename, format="com.databricks.spark.csv", header=True, inferSchema=True)
    .rdd.toDF(["id", "x", "y", "cluster"])
)


# In[113]:

from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import DenseVector
import pandas as pd
import ggplot as gg

featuresRDD = initialDF.map(lambda r: DenseVector([r.x, r.y]))

kValues = [3, 4, 5, 6, 7]
models = [KMeans.train(featuresRDD, k=k) for k in kValues]
WSSSEs = [m.computeCost(featuresRDD) for m in models]

rowsWSSSSE = list(zip(kValues, map(float, WSSSEs)))
WSSSEPdDF = pd.DataFrame(rowsWSSSSE, columns=["K", "WSSSE"])


# In[114]:

WSSSEPdDF


# In[115]:

get_ipython().magic('matplotlib inline')
gg.ggplot(gg.aes(x="K", y="WSSSE"), data=WSSSEPdDF) + gg.geom_line()


# In[116]:

bestK = 4
bestModel = models[ks.index(bestK)]


# In[117]:

def addPredictionColumn(row, model, featureColumns):
    rowDict = row.asDict()
    features = DenseVector([rowDict[c] for c in featureColumns])
    predictedValue = model.predict(features)
    allValues = tuple(row) + (predictedValue,)
    return Row(*allValues)

pointsWithPredictionDF = (
    initialDF.rdd.map(lambda row: addPredictionColumn(row, bestModel, ["x", "y"]))
).toDF(initialDF.columns + ["prediction"])


# In[118]:

pointsWithPredictionDF.show()


# In[119]:

get_ipython().magic('matplotlib inline')
gg.ggplot(gg.aes(x="x", y="y", color="prediction"), data=pointsWithPredictionDF.toPandas()) + gg.geom_point()

