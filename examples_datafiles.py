
# coding: utf-8

# # Įvairių formatų duomenų failų įkėlimas, konvertavimas ir išsaugojimas

# ## Python

# ### SAS `.sas7bdat` formato failų skaitymas panaudojant _Python_ paketą `sas7bdat`

# Importuojame _Python_ paketą `sas7bdat`:

# In[1]:

import sas7bdat


# _SAS_ `.sas7bdat` formato failą nuskaitome ir paverčiame į _Python_ [`pandas`](http://pandas.pydata.org/) [`DataFrame`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) objektą:

# In[2]:

sasFilename = 'data/CAB1.sas7bdat'
with sas7bdat.SAS7BDAT(sasFilename) as f:
    pandasDf = f.to_data_frame()


# In[3]:

pandasDf


# `pandas.DataFrame` objekto turinį įrašome į `.csv` formato failą.

# In[4]:

csvFilename = sasFilename.replace('.sas7bdat', '.csv')
csvFilename


# In[5]:

pandasDf.to_csv(csvFilename)


# Išvedame `.csv` failo turinį panaudodami _linux_ komandą `cat`:

# In[6]:

get_ipython().system(' cat data/CAB1.csv')


# Pastebime, kad pirmasis išsaugotas stulpelis neturi pavadinimo. Jo reikšmės yra `pandas.DataFrame` indeksas. Šis indeksas buvo sukurtas duomenų įkėlimo metu ir saugojant duomenis mums nėra reikalingas. 
# 
# Parametru `index=None` nurodome, kad nenorime saugoti indekso stulpelio:

# In[7]:

pandasDf.to_csv(csvFilename, index=None)


# Dar kartą išvedame `.csv` failo turinį panaudodami _linux_ komandą `cat`:

# In[8]:

get_ipython().system(' cat data/CAB1.csv')


# Dabar išsaugojome tik duomenis.

# **_Pastaba_**: _Jupyter Notebook_ _Code_ rūšies ląstelėje įrašę `!` simbolį galime naudoti _linux_ komandas, pvz:

# Išvesti darbinės direktorijos turinį:

# In[9]:

get_ipython().system(' ls -a')


# **_Pastaba:_** _Jupyter Notebook_ darbinė direktorija yra direktorija, kurioje patalpintas dabartinis bloknoto `.ipynb` failas:

# In[10]:

get_ipython().system(' pwd')


# Sukurti naują direktoriją `nauja_direktorija` ir išvesti darbinės direktorijos turinį:

# In[11]:

get_ipython().system(' mkdir nauja_direktorija')
get_ipython().system(' ls')


# Ištrinti direktoriją `nauja_direktorija` ir išvesti darbinės direktorijos turinį:

# In[12]:

get_ipython().system(' rm -rf nauja_direktorija')
get_ipython().system(' ls')


# Nadojant simbolį `|` Sukurti komandų seką, kai vienos komandos išvestis tampa kitos komandos įvestimi:

# In[13]:

get_ipython().system(' ls data | grep .xls')


# ### Excel _.xls_ ir _.xlsx_ formato failų skaitymas

# Toliau pateikiama keletas būdų kaip įkelti _Excel_ `.xls` ir `.xlsx` formatų failus ir juos išsaugoti `.csv` formatu.

# #### Python `pandas` paketas

# Importuojame _Python_ paketą [`pandas`](http://pandas.pydata.org/) ir pakeičiame jo pavadinimą į `pd`:

# In[14]:

import pandas as pd


# In[15]:

pdDf = pd.read_excel("data/CA1.xls")
pdDf.head()


# Galime išsaugoti `df` turinį į `.csv` formato failą kaip tai darėme su įkeltu `.sas7bdat` failu:

# In[16]:

pdDf.to_csv("data/CA1.csv", index=None)


# Išvedame pirmas 10 failo `CA1.csv` eilučių:

# In[17]:

get_ipython().system(' head data/CA1.csv')


# #### Python `xlrd` paketas

# `xlrd` paketas leidžia dirbti su _Excel_ failais itin žemu abstrakcijos lygmeniu. Jeigu _Excel_ failai yra tvarkingos lentelės, prasidedančios viršutiniame kairiame _Excel_ darbo knygos kampe, jų įkėlimui patogiau naudoti kitus metodus.

# Importuojame reikalingus _Python_ paketus ir sukuriame funkciją, kuri nuskaito _Excel_ failą ir jo darbinius lakštus (angl. worksheet) išsaugo `.csv` failų formatu:

# In[18]:

import xlrd
import csv
import os

def write_csv_from_excel(excel_filepath):
    dir_path = os.path.dirname(excel_filepath)
    workbook = xlrd.open_workbook(excel_filepath)
    for worksheet_name in workbook.sheet_names():
        worksheet = workbook.sheet_by_name(worksheet_name)
        if not worksheet.nrows:
            continue
        csv_filename = os.path.join(dir_path, worksheet_name) + '.csv'
        with open(csv_filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            for rownum in range(worksheet.nrows):
                csvwriter.writerow(worksheet.row_values(rownum))


# In[19]:

write_csv_from_excel("data/CA2.xls")


# In[20]:

get_ipython().system(' ls data')


# In[21]:

get_ipython().system(' ls data | grep CA2')


# In[22]:

get_ipython().system(' head data/CA2.csv')


# #### Python `csvkit` paketas ir _Linux_ komandinė eilutė

# Jeigu sistemoje yra įdiegtas _Python_ paketas [`csvkit`](http://csvkit.readthedocs.org/en/0.9.1/), galime pasinaudoti komandine eilutės programa `in2csv`:

# In[23]:

get_ipython().system(' in2csv data/CA1.xls > data/CA1_csvkit.csv')


# In[24]:

get_ipython().system(' head data/CA1_csvkit.csv')


# Detali programos `in2csv` naudojimo dokumentacija:

# In[25]:

get_ipython().system(' in2csv -h')


# ## pySpark

# Duomenys į _Apache Spark_ paprastai įkeliami iš duomenų bazių arba tekstinių failų, esančių lokalioje _Linux_ failų sistemoje arba _Hadoop_ paskirstytoje failų sistemoje (angl. Hadoop Distributed File System - [HDFS](http://hortonworks.com/hadoop/hdfs/)). Studijų modulyje **P160M132** apsiribosime darbu su tekstiniais failais. 
# 
# Toliau pateikiama keletas metodų kaip įkelti ir išsaugoti tekstinius failus su _Apache Spark_.

# ### Tekstiniai failai lokalioje _Linux_ failų sistemoje 

# #### `sc.textFile` metodas

# In[26]:

csvRDD = sc.textFile("./data/CA1.csv")
print(type(csvRDD))
csvRDD.take(5)


# Matome, kad `sc.textFile` metodas grąžina tipo `RDD` objektą. Pažiūrime kokio tipo yra `RDD` turinys.

# In[27]:

csvRDD.map(type).take(5)


# Nieko nuostabaus, kad įkėlę tekstinį failą, turime [`str`](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str) (tekstinių simbolių sekos) tipo objektų rinkinį. Kadangi failą skaitome naudomi _Apache Spark_, tikėtina, kad norėsime atlikti kažką daugiau, negu išvesti pirmas **N** jo eilučių. Pastarajam veiksmui puikiai tinka _Linux_ komanda `head -n` **N**:

# In[28]:

get_ipython().system(' head -n 5 data/CA1.csv')


# Norėdami atlikti skaičiavimus, turime atitinkamus stulpelius konvertuoti į reikiamo tipo reikšmes, tačiau mūsų pirmoji eilutė yra stupelių pavadinimai. Pirmąją eilutę paimame naudodami `RDD` metodą `first`:

# In[29]:

header = csvRDD.first()
header


# Atskiriame pirmąją eilutę nuo `RDD`. Nors skamba paprastai, tai nėra visiškai trivialus veiksmas.

# In[30]:

import itertools as it

rowsRDD = csvRDD.mapPartitionsWithIndex(lambda idx, gen: it.islice(gen, 1, None) if idx == 0 else gen)

rowsRDD.take(5)


# Prieš aptardami ką tik naudotą anoniminę funkciją, panagrinėkime kokių tipų elementai pasiekiami šiai funkcijai:

# In[31]:

csvRDD.mapPartitionsWithIndex(lambda elem1, elem2: (str(type(elem1)), str(type(elem2)))).take(5)


# `mapPartitionsWithIndex` metodo argumentas yra funkcija, kuriai pateikiami 2 argumentai: `RDD` particijos indeksas ir particijos [`generator`](https://docs.python.org/3/library/stdtypes.html#generator-types) tipo objektas. _Python_ generatoriai nesaugo visų savo elementų atmintyje iš karto, tačiau juos generuoja po vieną, tik tada kai jų paprašoma.
# 
# Todėl panaudojame _Python_ standartinės bibliotekos modulį `itertools`. Panaudodami [`itertools.islice`](https://docs.python.org/3/library/itertools.html?highlight=itertools%20islice#itertools.islice) klasę pirmosios particijos generatorių pakeičiame generatoriumi be pirmojo elemento.
# 
# **_Pastaba_**: Atminkime, kad _Python_ indeksai prasideda nuo 0.

# Kadangi turime stulpelių pavadinimus ir duomenų eilutes atskirai, pastarąsias galime konvertuoti į skaitines reikšmes. Kad būtų aiškiau, po kiekvieno žingsnio pateiksime `RDD` elementų tipus.

# In[32]:

splitRowsRDD = rowsRDD.map(lambda line: line.split(","))
splitRowsRDD.take(5)


# `splitRowsRDD` elementų tipas:

# In[33]:

splitRowsRDD.map(type).take(5)


# `splitRowsRDD` elementų tipas yra _Python_ `list` (sąrašas). Kokie yra šių sąrašų tipai?

# In[34]:

splitRowsRDD.map(lambda list_: list(map(type, list_))).take(5)


# Matome, kad `splitRowsRDD` sudaro sąrašai su `str` tipo elementais.

# Prieš atliekant skaičiavimus `str` tipo elementus reikia pavesti skaitinių tipų kintamaisiais:

# In[35]:

typedRowsRDD = splitRowsRDD.map(lambda vals: [int(vals[0]), float(vals[1]), float(vals[2]), int(vals[3])])
typedRowsRDD.take(5)


# Ar pavyko?

# In[36]:

typedRowsRDD.map(lambda list_: [type(element) for element in list_]).take(5)


# Panašu, kad payko.

# _Apache Spark_ turi ir aukštesnio lygio abstrakciją - [`DataFrame`](http://spark.apache.org/docs/latest/sql-programming-guide.html#dataframes). Prieš turimą `RDD` paverčiant į `DataFrame` tipo objektą, `RDD` sąrašus pakeisime _pySpark_ `Row` tipo objektais.

# Eilutę su stulpelių pavadinimais atskyrėme anksčiau. Eilutė:

# In[37]:

header


# Stulpelių pavadnimų sąrašas:

# In[38]:

headerColumns = header.split(",")
headerColumns


# Pagal turimus stulpelius sukuriame savo duomenims pritaikytą `Row`.

# In[39]:

import pyspark.sql


# In[40]:

ca1Row = pyspark.sql.Row("ID", "X", "Y", "klasteris")
ca1Row


# arba

# In[41]:

ca1Row = pyspark.sql.Row(headerColumns[0], headerColumns[1], headerColumns[2], headerColumns[3])
ca1Row


# arba

# In[42]:

ca1Row = pyspark.sql.Row(*headerColumns)
ca1Row


# Sudarome `RDD` su `Row` tipo objektais:

# In[43]:

ca1RowsRDD = typedRowsRDD.map(lambda values: ca1Row(values[0], values[1], values[2], values[3]))
ca1RowsRDD.take(5)


# arba

# In[44]:

ca1RowsRDD = typedRowsRDD.map(lambda values: ca1Row(*values))
ca1RowsRDD.take(5)


# Ar tikrai pavyko?

# In[45]:

ca1RowsRDD.map(type).take(5)


# Dabar galima `RDD` paversti į `DataFrame`:

# In[46]:

ca1DF = ca1RowsRDD.toDF()
ca1DF


# In[47]:

ca1DF.show(5)


# Puiku! `DataFrame` objektai bus naudojami su _Apache Spark_ _Machine Learning_ metodais.

# Reikėjo atlikti nemažai darbo, kol iš tekstinio failo gavome `DataFrame` objektą. Galbūt šį procesą galime automatizuoti? Atliktus veiksmus sudedame į _Python_ funkciją:

# In[48]:

from pyspark.sql import Row
from itertools import islice

def parseCsvToDf(csvFilePath):
    rawRdd = sc.textFile(csvFilePath)
    header = rawRdd.first().split(",")
    DataRow = Row(*[h.lower() for h in header])
    rowsRdd = (
        rawRdd
        .mapPartitionsWithIndex(lambda idx, gen: islice(gen, 1, None) if idx == 0 else gen)
        .map(lambda line: DataRow(*map(float, line.split(","))))
    )
    rowDf = rowsRdd.toDF()
    return rowDf


# Pasinaudojame ką tik aprašyta funkcija:

# In[49]:

ca1ParsedDF = parseCsvToDf("data/CA1.csv")
ca1ParsedDF


# Aprašydami funkciją padarėme prielaidą, kad visos `.csv` failo duomenų eilučių reikšmės yra skaitinės. Ar veiks ši funkcija, jeigu bent vienas stulpelis turės tekstines reikšmes? Pamėgikime. Įrašome testinį failą ir jį nuskaitome:

# In[50]:

test_filename = "data/csv_file_with_strings.csv"
with open(test_filename, "w") as f:
    f.writelines(["stulpelis_a,stulpelis_b\n", "simbolinė_reikšmė_1,10\n", "simbolinė_reikšmė_2,20\n"])


# Patikrininame įrašyto failo turinį:

# In[51]:

get_ipython().system(' cat data/csv_file_with_strings.csv')


# Bandome įkelti failą naudodami savo aprašytą funkciją (įvyks klaida):

# In[52]:

testDF = parseCsvToDf(test_filename)


# Klaidos pranešime matome eilutę:
# 
# `ValueError: could not convert string to float: 'simbolinė_reikšmė_1'`
# 
# Ši klaida reikškia, kad simbolių eilutės tipo (angl. string) reikšmės  `simbolinė_reikšmė_1` nepavyko paveiksti į skaitinio tipo `float` reikšmę.

# Norėdami aprašyti funkciją, kuri patikrintų galimus variantus ir stulpeliams parinktų teisingus tipus, turėtume daryti panašiai kaip daroma [čia](https://github.com/seahboonsiew/pyspark-csv/blob/master/pyspark_csv.py). Tada turėtume universalią funkciją tekstinio `RDD` pavertinmui į `DataFrame`. Tačiau atminkime, kad _Python_ yra interpretuojama programavimo kalba, todėl gerokai lėtesnė už kompiliuojamą [_Scala_](http://www.scala-lang.org/) programavimo kalbą, kuria ir parašytas _Apache Spark_. Žemiau pateiktas šių kalbų spartos palyginimas darbui su _Apache Spark_ paimtas iš [šio šaltinio](https://databricks.com/blog/2015/04/24/recent-performance-improvements-in-apache-spark-sql-python-dataframes-and-more.html).

# <img src="https://databricks.com/wp-content/uploads/2015/02/Screen-Shot-2015-02-16-at-9.46.39-AM.png">

# Matome, kad naudojant `DataFrame` objektų metodus skirtumo tarp kalbų nėra, tačiau jis labai ryškus `RDD` objektų metodams. Žemiau pateikiame _Scala_ kalba parašytą _Apache Spark_ paketą, kurį galima naudoti iš _Python_ kalbos _Apache Spark_ sąsajos.

# #### [spark-csv](https://github.com/databricks/spark-csv) paketas iš http://spark-packages.org/

# ##### `.csv` failo įkėlimas

# In[53]:

ca1EasyDF = sqlContext.read.format("com.databricks.spark.csv").options(header=True, inferSchema=True).load("data/CA1.csv")
ca1EasyDF


# In[54]:

ca1EasyDF.show(5)


# arba

# In[55]:

ca1EasyDF = sqlContext.read.load("data/CA1.csv", format="com.databricks.spark.csv", header=True, inferSchema=True)
ca1EasyDF


# In[56]:

ca1EasyDF.show(5)


# Gavome ir stulpelių pavadinimus, ir jų tipus. Patikriname su testiniu failu:

# In[57]:

testEasyDF = sqlContext.read.load(
    "data/csv_file_with_strings.csv", 
    format="com.databricks.spark.csv", 
    header=True, inferSchema=True)
testEasyDF


# In[58]:

testEasyDF.show()


# Puiku!

# Tikrai rekomenduotina `.csv` failų įkėlimui naudoti pastarąjį metodą :) Parinkčių paaiškinimai pateikti [čia](https://github.com/databricks/spark-csv#features).

# ##### `DataFrame` išsaugojimas `.csv` formatu

# In[59]:

testEasyDF.write.format("com.databricks.spark.csv").save("data/csv_test_write")


# **_Pastaba:_** _Apache Spark_ duomenis saugoja į direktoriją atskirais failais. Jeigu duomenų išsaugojimas atliktas sėktingai, direktorijoje sukuriamas tuščias failas `_SUCCESS`. Saugojimas negalimas, jeigu egzistuoja direktorija tokiu pačiu pavadinimu ir įvyks klaida 
# 
# `java.lang.RuntimeException: path data/csv_test_write already exists.`
# 
# Išsaugoti galima tik duomenų eilutes (be stulpelių pavadinimų eilutės).

# In[60]:

# java.lang.RuntimeException: path data/csv_test_write already exists.
testEasyDF.write.format("com.databricks.spark.csv").save("data/csv_test_write")


# Žemiau pateikiama keletas komandų rezultatų peržiūrai.

# In[61]:

# išvesti direktorijos turinį
get_ipython().system(' ls data/csv_test_write/')


# In[62]:

#išvesti pirmasias 2 failo eilutes
get_ipython().system(' head -n 2 data/csv_test_write/part-00000')


# In[63]:

#išvesti paskutinę failo eilutę
get_ipython().system(' tail -n 1 data/csv_test_write/part-00000')


# In[64]:

# failas _SUCCESS tikrai tuščias
get_ipython().system(' cat data/csv_test_write/_SUCCESS')


# In[65]:

# išvesti sujuntą visų direktorijos failų turinį
get_ipython().system(' cat data/csv_test_write/*')


# In[66]:

# išvesti kiekvieno direktorijos failo pirmąją eilutę
get_ipython().system(' head -n 1 data/csv_test_write/*')

