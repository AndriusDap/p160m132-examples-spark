
import sas7bdat

sasFilename = 'data/CAB1.sas7bdat'
with sas7bdat.SAS7BDAT(sasFilename) as f:
    pandasDf = f.to_data_frame()

pandasDf

csvFilename = sasFilename.replace('.sas7bdat', '.csv')
csvFilename

pandasDf.to_csv(csvFilename)

! cat data/CAB1.csv

pandasDf.to_csv(csvFilename, index=None)

! cat data/CAB1.csv

! ls -a

! pwd

! mkdir nauja_direktorija
! ls

! rm -rf nauja_direktorija
! ls

! ls data | grep .xls

import pandas as pd

pdDf = pd.read_excel("data/CA1.xls")
pdDf.head()

pdDf.to_csv("data/CA1.csv", index=None)

! head data/CA1.csv

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

write_csv_from_excel("data/CA2.xls")

! ls data

! ls data | grep CA2

! head data/CA2.csv

! in2csv data/CA1.xls > data/CA1_csvkit.csv

! head data/CA1_csvkit.csv

! in2csv -h

csvRDD = sc.textFile("./data/CA1.csv")
print(type(csvRDD))
csvRDD.take(5)

csvRDD.map(type).take(5)

! head -n 5 data/CA1.csv

header = csvRDD.first()
header

import itertools as it

rowsRDD = csvRDD.mapPartitionsWithIndex(lambda idx, gen: it.islice(gen, 1, None) if idx == 0 else gen)

rowsRDD.take(5)

csvRDD.mapPartitionsWithIndex(lambda elem1, elem2: (str(type(elem1)), str(type(elem2)))).take(5)

splitRowsRDD = rowsRDD.map(lambda line: line.split(","))
splitRowsRDD.take(5)

splitRowsRDD.map(type).take(5)

splitRowsRDD.map(lambda list_: list(map(type, list_))).take(5)

typedRowsRDD = splitRowsRDD.map(lambda vals: [int(vals[0]), float(vals[1]), float(vals[2]), int(vals[3])])
typedRowsRDD.take(5)

typedRowsRDD.map(lambda list_: [type(element) for element in list_]).take(5)

header

headerColumns = header.split(",")
headerColumns

import pyspark.sql

ca1Row = pyspark.sql.Row("ID", "X", "Y", "klasteris")
ca1Row

ca1Row = pyspark.sql.Row(headerColumns[0], headerColumns[1], headerColumns[2], headerColumns[3])
ca1Row

ca1Row = pyspark.sql.Row(*headerColumns)
ca1Row

ca1RowsRDD = typedRowsRDD.map(lambda values: ca1Row(values[0], values[1], values[2], values[3]))
ca1RowsRDD.take(5)

ca1RowsRDD = typedRowsRDD.map(lambda values: ca1Row(*values))
ca1RowsRDD.take(5)

ca1RowsRDD.map(type).take(5)

ca1DF = ca1RowsRDD.toDF()
ca1DF

ca1DF.show(5)

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

ca1ParsedDF = parseCsvToDf("data/CA1.csv")
ca1ParsedDF

test_filename = "data/csv_file_with_strings.csv"
with open(test_filename, "w") as f:
    f.writelines(["stulpelis_a,stulpelis_b\n", "simbolinė_reikšmė_1,10\n", "simbolinė_reikšmė_2,20\n"])

! cat data/csv_file_with_strings.csv

testDF = parseCsvToDf(test_filename)

ca1EasyDF = sqlContext.read.format("com.databricks.spark.csv").options(header=True, inferSchema=True).load("data/CA1.csv")
ca1EasyDF

ca1EasyDF.show(5)

ca1EasyDF = sqlContext.read.load("data/CA1.csv", format="com.databricks.spark.csv", header=True, inferSchema=True)
ca1EasyDF

ca1EasyDF.show(5)

testEasyDF = sqlContext.read.load(
    "data/csv_file_with_strings.csv", 
    format="com.databricks.spark.csv", 
    header=True, inferSchema=True)
testEasyDF

testEasyDF.show()

testEasyDF.write.format("com.databricks.spark.csv").save("data/csv_test_write")

# java.lang.RuntimeException: path data/csv_test_write already exists.
testEasyDF.write.format("com.databricks.spark.csv").save("data/csv_test_write")

# išvesti direktorijos turinį
! ls data/csv_test_write/

#išvesti pirmasias 2 failo eilutes
! head -n 2 data/csv_test_write/part-00000

#išvesti paskutinę failo eilutę
! tail -n 1 data/csv_test_write/part-00000

# failas _SUCCESS tikrai tuščias
! cat data/csv_test_write/_SUCCESS

# išvesti sujuntą visų direktorijos failų turinį
! cat data/csv_test_write/*

# išvesti kiekvieno direktorijos failo pirmąją eilutę
! head -n 1 data/csv_test_write/*
