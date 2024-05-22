import pyspark
from pyspark.sql import SparkSession
import sys
import time


import os
import sys
# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
# os.environ["SPARK_HOME"] = "/content/spark-3.2.1-bin-hadoop3.2"


import findspark
findspark.init()
findspark.find()



from pyspark.sql import DataFrame, SparkSession
from typing import List
import pyspark.sql.types as T
import pyspark.sql.functions as F

spark= SparkSession \
       .builder \
       .appName("Our First Spark Example") \
       .getOrCreate()

spark

from pyspark.sql import SparkSession
from pyspark import SparkContext
import math
import sys
import time

spark = SparkSession.builder.appName("Outlier Detection").getOrCreate()
sc = spark.sparkContext

def pairwise_distances(points):
    distances = []
    for i in range(len(points)):
        row = []
        for j in range(len(points)):
            dist = math.sqrt((points[i][0] - points[j][0]) ** 2 +
                             (points[i][1] - points[j][1]) ** 2)
            row.append(dist)
        distances.append(row)
    return distances

def ExactOutliers(points, D, M, K):
    distances = pairwise_distances(points)
    outlier_count = 0
    outliers = []

    for i in range(len(points)):
        neighbor_count = 0
        for j in range(len(points)):
            if distances[i][j] <= D:
                neighbor_count += 1
        if neighbor_count <= M:
            outliers.append((points[i], neighbor_count))
            outlier_count += 1

    # Sort
    outliers.sort(key=lambda x: x[1])

    print("Number of Outliers =", outlier_count)
    for i in range(min(K, outlier_count)):
        print("Point:", outliers[i][0])

def getNeighbors(cellId, radius):
    i, j = cellId
    neighbors = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
                neighbors.append((i + dx, j + dy))
    return neighbors

def MRApproxOutliers(inputPoints, D, M, K):
    Lambda = D /(2 * math.sqrt(2) )

    def mapToCell(point):
        i = int(point[0] / Lambda)
        j = int(point[1] / Lambda)
        return (i, j), 1

    cellsRDD = inputPoints.map(mapToCell).reduceByKey(lambda a, b: a + b)

    def processCells(partitionIterator):
      cellData = dict(partitionIterator)
      outlierCount = 0
      uncertainCount = 0

      for cellId, size in cellData.items():
          neighbors_R3 = getNeighbors(cellId, radius=1)
          neighbors_R7 = getNeighbors(cellId, radius=3)

          N3 = sum(cellData.get(n, 0) for n in neighbors_R3)
          N7 = sum(cellData.get(n, 0) for n in neighbors_R7)

          if N3 > M:
              # Non outlier
              continue
          elif N7 <= M:
              # Sure outlier
              outlierCount += size
          else:
              # Uncertain
              uncertainCount += size

      return [(outlierCount, uncertainCount)]


    results = cellsRDD.mapPartitions(processCells).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    totalOutliers, totalUncertain = results

    cellsBySize = cellsRDD.flatMap(lambda x: [(x[0], x[1])])\
                          .sortBy(lambda x: x[1], ascending=True)\
                          .take(K)

    print(f"Number of sure outliers =: {totalOutliers}")
    print(f"Number of uncertain points = {totalUncertain}")
    for cellId, size in cellsBySize:
        print(f"Cell: {cellId}, Size: {size}")

def main(argv):
    # Parse command-line arguments
    if len(argv) != 6:
        print("Usage: <script> <file_path> D M K L", file=sys.stderr)
        sys.exit(-1)

    file_path = sys.argv[1]
    D = float(sys.argv[2])
    M = int(sys.argv[3])
    K = int(sys.argv[4])
    L = int(sys.argv[5])

    print(f"File={file_path} D={D} M={M} K={K} L={L}")

    # Load and preprocess the dataset
    rawData = sc.textFile(file_path)
    #inputPoints = transformAndPartition(rawData, L)
    inputPoints = rawData.map(lambda line: tuple(map(float, line.split(','))))
    Number_of_points = inputPoints.count()
    print("Number of points =", Number_of_points)

    # Repartition the RDD based on the command-line argument, if specified
    if L > 0:
        inputPoints = inputPoints.repartition(L)

    if inputPoints.count() <= 200000:
        # Collect points for ExactOutliers if dataset is small enough
        listOfPoints = inputPoints.collect()
        start_time = time.time()
        ExactOutliers(listOfPoints, D, M, K)
        end_time = time.time()
        print("Running time of ExactOutliers = {:.2f} seconds".format(end_time - start_time))
    start_time = time.time()
    MRApproxOutliers(inputPoints, D, M, K)
    end_time = time.time()
    print("Running time of MRApproxOutliers = {:.2f} seconds".format(end_time - start_time))

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main(sys.argv)
