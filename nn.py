"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.

Based on AbaloneTest.java by Hannah Lau
"""
from __future__ import with_statement
import os
import csv
import time
import sys
import math
from itertools import product

sys.path.append("/Users/james/school/CS7641/Assignment2/ABAGAIL/ABAGAIL.jar")
from func.nn.backprop import BackPropagationNetworkFactory
from func.nn.backprop import BatchBackPropagationTrainer
from func.nn.backprop import RPROPUpdateRule
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm

do_fmnist = False
do_chess = True

do_rhc = True
do_sa = False
do_ga = True
do_bp = True

sweep = True

if sweep:
    sa_temp = [1E10, 1E11, 1E12]
    sa_cooling = [0.85, 0.9, 0.95]

    ga_pop = [100, 200, 300]
    ga_prop_mate = [0.5, 0.6]
    ga_prop_mutate = [0.05, 0.1]
else:
    # TBD
    sa_temp = [1E11]
    sa_cooling = [0.95]

    # TBD
    ga_pop = [200]
    ga_prop_mate = [0.5]
    ga_prop_mutate = [0.05]

if do_chess:
    TRAIN_FILE = os.path.join("..",  "..", "chess_train.csv")
    TEST_FILE = os.path.join("..",  "..", "chess_test.csv")

if do_fmnist:
    # TRAIN_FILE = os.path.join("..",  "..", "fmnist_train.csv")
    TRAIN_FILE = os.path.join("..",  "..", "fmnist_debug.csv") # TODO REMOVE THIS, just doing this to cut runtime down
    TEST_FILE = os.path.join("..",  "..", "fmnist_debug.csv")


if do_chess:
    INPUT_LAYER = 73  # Chess
    HIDDEN_LAYER1 = 50    # Chess
    HIDDEN_LAYER2 = 10    # Chess
    OUTPUT_LAYER = 1    # Chess
if do_fmnist:
    INPUT_LAYER = 784   # FMNIST
    OUTPUT_LAYER = 10   # FMNIST
    HIDDEN_LAYER = 5    # TBD

TRAINING_ITERATIONS = 10000
GA_TRAINING_ITERATIONS = 1000


def get_accuracy(network, instances):
    correct = 0
    incorrect = 0
    for instance in instances:
        network.setInputValues(instance.getData())
        network.run()
        if do_chess:
            predicted = instance.getLabel().getContinuous()
            actual = network.getOutputValues().get(0)
            if abs(predicted - actual) < 0.5:
                correct += 1
            else:
                incorrect += 1
        if do_fmnist:
            actual = instance.getLabel()
            predicted = network.getOutputValues()
            temp_max = 0
            predicted_classification = -1
            actual_classification = -1
            for j in range(predicted.size()):
                temp = predicted.get(j)
                if temp > temp_max:
                    temp_max = temp
                    predicted_classification = j
            if abs(actual.getContinuous(j) - 1.0) < 1E-9:
                actual_classification = j
            if abs(predicted_classification - actual_classification) < 0.5:
                correct += 1
            else:
                incorrect += 1
    accuracy = float(correct)/float(correct+incorrect)
    return correct, incorrect, accuracy


def initialize_instances(test=False):
    """Read the abalone.txt CSV data into a list of instances."""
    instances = []

    if test:
        INPUT_FILE = TEST_FILE
    else:
        INPUT_FILE = TRAIN_FILE
    with open(INPUT_FILE, "r") as chess:
        reader = csv.reader(chess)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            if do_chess:
                instance.setLabel(Instance(float(row[-1])))
            if do_fmnist:
                classes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                index = int(row[-1])
                # print "Value is: ", row[-1], " index is: ", index
                classes[index] = 1.0
                temp = Instance(classes)
                # print "Size is: ", temp.size()
                instance.setLabel(temp)
            instances.append(instance)
    return instances


def train(oa, network, oaName, instances, measure, training_iters, test_instances, print_header=False, param1=None, param2=None, param3=None):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """
    start = time.time()
    print "\nError results for %s\n---------------------------" % (oaName,)
    if oaName == "SA" or oaName == "SA_sweep":
        print "Temperature: %d, Cooling %f" % (param1, param2)
    elif oaName == "GA" or oaName == "GA_sweep":
        print "Population: %d, prop_mate %f, prop_mutate %f" % (param1, param2, param3)
    file = 'output_temp/'+oaName+'_nn_results.csv'
    if print_header:
        with open(file, 'w') as f:
            if oaName == "SA" or oaName == "SA_sweep":
                f.write('{},{},{},{},{},{},{}\n'.format("iter_num","temperature", "cooling", "error","runtime","training_accuracy", "testing_accuracy"))
            elif oaName == "GA" or oaName == "GA_sweep":
                f.write('{},{},{},{},{},{},{},{}\n'.format("iter_num","pop", "prop_mate", "prop_mutate", "error","runtime","training_accuracy", "testing_accuracy"))
            else:
                f.write('{},{},{},{},{}\n'.format("iter_num", "error","runtime","training_accuracy", "testing_accuracy"))

    for iteration in xrange(training_iters):
        oa.train()

        error = 0.00
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            if do_fmnist:
                example = Instance(output_values, Instance(output_values))
            if do_chess:
                example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

        if iteration % 10 == 0:
            current = time.time()
            elapsed = current - start
            dummy1, dummy2, train_accuracy = get_accuracy(network, instances)
            dummy1, dummy2, test_accuracy = get_accuracy(network, test_instances)
            with open(file, 'a') as f:
                if oaName == "SA" or oaName == "SA_sweep":
                    f.write('{},{},{},{},{},{},{}\n'.format(iteration, param1, param2, error, elapsed, train_accuracy, test_accuracy))
                elif oaName == "GA" or oaName == "GA_sweep":
                    f.write('{},{},{},{},{},{},{},{}\n'.format(iteration, param1, param2, param3, error, elapsed, train_accuracy, test_accuracy))
                else:
                    f.write('{},{},{},{},{}\n'.format(iteration, error, elapsed, train_accuracy, test_accuracy))
            if error < 1E-9:
                print "Terminating at iteration: ", iteration," as error is small: ", error
                break
            print "Error = %0.03f on iter %d" % (error, iteration)


def main():
    """Run algorithms on the abalone dataset."""
    train_instances = initialize_instances()
    test_instances = initialize_instances(test=True)
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(train_instances)

    networks = []  # BackPropagationNetwork
    oa = []  # OptimizationAlgorithm
    oa_names = []
    if do_rhc:
        oa_names.append("RHC")
    if do_sa:
        oa_names.append("SA")
    if do_ga:
        oa_names.append("GA")
    if do_bp:
        oa_names.append("BP")
    results = ""

    # For each algo, need to see if we are doing sweeps

    # No need to sweep rhc as there are no parameters
    if do_rhc and sweep == False:
        training_iter = TRAINING_ITERATIONS
        if do_fmnist:
            classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
        if do_chess:
            classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER])
        nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
        oa = RandomizedHillClimbing(nnop)
        name = "RHC"
        train(oa, classification_network, name, train_instances, measure, training_iter, test_instances, True)

    if do_sa:
        training_iter = TRAINING_ITERATIONS
        count = 0
        for temp, cooling in product(sa_temp, sa_cooling):
            if do_fmnist:
                classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
            if do_chess:
                classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER])
            nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
            oa = SimulatedAnnealing(temp, cooling, nnop)
            name = "SA_sweep"
            if count == 0:
                print_head = True
            else:
                print_head = False
            train(oa, classification_network, name, train_instances, measure, training_iter, test_instances, print_head, temp, cooling)
            count += 1

    if do_ga:
        training_iter = GA_TRAINING_ITERATIONS
        count = 0
        for pop, prop_mate, prop_mutate in product(ga_pop, ga_prop_mate, ga_prop_mutate):
            if do_fmnist:
                classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
            if do_chess:
                classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER])
            nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
            mate = int(math.floor(pop*prop_mate))
            mutate = int(math.floor(pop*prop_mutate))
            oa = StandardGeneticAlgorithm(pop, mate, mutate, nnop)
            name = "GA_sweep"
            if count == 0:
                print_head = True
            else:
                print_head = False
            train(oa, classification_network, name, train_instances, measure, training_iter, test_instances, print_head, pop, prop_mate, prop_mutate)
            count += 1

    if do_bp and sweep == False:
        training_iter = TRAINING_ITERATIONS
        if do_fmnist:
            classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
        if do_chess:
            classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER])
        oa = BatchBackPropagationTrainer(data_set, classification_network, measure, RPROPUpdateRule())
        name = "BP"
        train(oa, classification_network, name, train_instances, measure, training_iter, test_instances, True)


if __name__ == "__main__":
    main()

