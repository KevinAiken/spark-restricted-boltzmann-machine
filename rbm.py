from pyspark import SparkContext
import pyspark
import numpy
from pyspark.mllib.linalg.distributed import IndexedRowMatrix, IndexedRow
from pyspark.sql import SparkSession
import csv


def as_block_matrix(rdd, rowsPerBlock=65000, colsPerBlock=65000):
    return IndexedRowMatrix(
        rdd.zipWithIndex().map(lambda xi: IndexedRow(xi[1], xi[0]))
    ).toBlockMatrix(rowsPerBlock, colsPerBlock)


def train_rbm(data, spark_context, visible_nodes, hidden_nodes, learning_rate=.2, iterations=30):
    SparkSession(spark_context) # without this `'PipelinedRDD' object has no attribute 'toDF'` will be thrown

    weights = numpy.asarray(numpy.random.uniform(
        low=-0.1 * numpy.sqrt(6. / (hidden_nodes + visible_nodes)),
        high=0.1 * numpy.sqrt(6. / (hidden_nodes + visible_nodes)),
        size=(visible_nodes, hidden_nodes)))
    weights = numpy.insert(weights, 0, 0, axis=0)
    weights = numpy.insert(weights, 0, 0, axis=1)

    num_data_entries = numpy.array(data).shape[0]

    # Insert bias units of 1 into the first column.
    data = numpy.insert(numpy.array(data), 0, 1, axis=1)

    print weights
    print data

    # Creating RDD's from vectors
    weightsRDD = spark_context.parallelize(weights)
    dataRDD = spark_context.parallelize(data)

    weightsBlockMatrix = as_block_matrix(weightsRDD)
    dataBlockMatrix = as_block_matrix(dataRDD)


    for iteration in range(iterations):
        #positive phase
        pos_hidden_activations = dataBlockMatrix.multiply(weightsBlockMatrix)

        # pulling data out of the cluster to run the sigmoid function
        pos_hidden_probs = 1.0 / (1 + numpy.exp(-pos_hidden_activations.blocks.collect()[0][1].toArray()))
        # overwrite the bias unit with 1 after the transformations
        pos_hidden_probs[:, 0] = 1
        # creating states from the hidden probabilities
        pos_hidden_states = pos_hidden_probs > numpy.random.rand(num_data_entries, hidden_nodes + 1)
        # putting states and probabilities into the cluster
        pos_hidden_states = as_block_matrix(spark_context.parallelize(pos_hidden_states))
        pos_hidden_probs = as_block_matrix(spark_context.parallelize(pos_hidden_probs))

        pos_associations = dataBlockMatrix.transpose().multiply(pos_hidden_probs)

        # negative phase
        neg_visible_activations = pos_hidden_states.multiply(weightsBlockMatrix.transpose())

        # pulling data out of the cluster to run the sigmoid function
        neg_visible_probs = 1.0 / (1 + numpy.exp(-neg_visible_activations.blocks.collect()[0][1].toArray()))
        # overwrite the bias unit with 1 after the transformations
        neg_visible_probs[:, 0] = 1
        # putting probabilities back into the cluster
        neg_visible_probs = as_block_matrix(spark_context.parallelize(neg_visible_probs))

        neg_hidden_activations = neg_visible_probs.multiply(weightsBlockMatrix)
        # pulling data out of the cluster to run the sigmoid function
        neg_hidden_probs = 1.0 / (1 + numpy.exp(-neg_hidden_activations.blocks.collect()[0][1].toArray()))
        # putting data back in after applying the sigmoid function
        neg_hidden_probs = as_block_matrix(spark_context.parallelize(neg_hidden_probs))

        neg_associations = neg_visible_probs.transpose().multiply(neg_hidden_probs)

        # Updating weights
        weights = numpy.copy(weightsBlockMatrix.blocks.collect()[0][1].toArray())
        weights += learning_rate * ((pos_associations.blocks.collect()[0][1].toArray() - neg_associations.blocks.collect()[0][1].toArray()) / num_data_entries)
        weightsBlockMatrix = as_block_matrix(spark_context.parallelize(weights))

        cost = numpy.sum((data - neg_visible_probs.blocks.collect()[0][1].toArray()) ** 2)
        print("Iteration %s: cost is %s" % (iteration, cost))

    # returns the trained weights of the network
    return weightsBlockMatrix.blocks.collect()[0][1].toArray()
if __name__ == '__main__':

    conf = pyspark.SparkConf().setAll([('spark.executor.memory', '6g'), ('spark.executor.cores', '8'), ('spark.cores.max', '8'), ('spark.driver.memory','6g')])
    sc = SparkContext(appName="RBM", conf=conf)

    # data used consists of many rows of entries with 784 features and no labels
    mnist_data = list(csv.reader(open("mnist_train.csv")))

    assignmentData = [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 1, 0],
                 [1, 0, 1, 0, 0, 0],
                 [1, 0, 0, 1, 0, 0],
                 [1, 1, 0, 1, 0, 0],
                 [1, 0, 1, 1, 0, 0]]

    xorData = [[0, 0, 0],
               [0, 1, 1],
               [1, 0, 1],
               [1, 1, 0]]

    # output is  the trained weights of the network
    print train_rbm(assignmentData, sc, 6, 3)
    print train_rbm(xorData, sc, 3, 2)
    print train_rbm(mnist_data, sc, 784, 200) # not working




