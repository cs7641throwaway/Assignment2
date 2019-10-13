# traveling salesman algorithm implementation in jython
# This also prints the index of the points of the shortest route.
# To make a plot of the route, write the points at these indexes
# to a file and plot them in your favorite tool.
import sys
import os
import time
import collections
import math

sys.path.append("/Users/james/school/CS7641/Assignment2/ABAGAIL/ABAGAIL.jar")
import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays

from array import array


def eval_algo(ef, algo, fixed_iter):
    fit = FixedIterationTrainer(algo, fixed_iter)
    start = time.time()
    fit.train()
    end = time.time()
    score = ef.value(algo.getOptimal())
    runtime = end - start
    call_count = 0
    return score, call_count, runtime


def print_rhc(rhc_results, str):
    # num_points, iter, scores/runtimes, N
    file = "./output/rhc_"+str+"_results.csv"
    with open(file, 'w') as f:
        f.write('{},{},{},{}\n'.format("num_points", "iter", "scores", "runtimes"))
    for num_points, d0 in rhc_results.iteritems():
        for iter, d1 in d0.iteritems():
            scores = []
            runtimes = []
            for metric, d2 in d1.iteritems():
                for N, val in d2.iteritems():
                    if metric == "scores":
                        scores.append(val)
                    elif metric == "runtimes":
                        runtimes.append(val)
                    else:
                        print "ERROR: Unexpected metric of: ", metric
            print "RHC Scores with num_points: ", num_points, " iter: ", iter, " metric: scores are: ",scores
            print "RHC Scores with num_points: ", num_points, " iter: ", iter, " metric: runtimes are: ", runtimes
            with open(file, 'a') as f:
                f.write('{},{},"{}","{}"\n'.format(num_points, iter, scores, runtimes))


def print_sa(sa_results, str):
    # num_points, iter, t, cooling, scores/runtimes, N
    file = "./output/sa_"+str+"_results.csv"
    with open(file, 'w') as f:
        f.write('{},{},{},{},{},{}\n'.format("num_points", "iter", "temp", "cooling", "scores", "runtimes"))
    for num_points, d0 in sa_results.iteritems():
        for iter, d1 in d0.iteritems():
            for t, d2 in d1.iteritems():
                for cooling, d3 in d2.iteritems():
                    scores = []
                    runtimes = []
                    for metric, d4 in d3.iteritems():
                        for N, val in d4.iteritems():
                            if metric == "scores":
                                scores.append(val)
                            elif metric == "runtimes":
                                runtimes.append(val)
                            else:
                                print "ERROR: Unexpected metric of: ", metric
                    print "SA Scores with num_points: ", num_points, " iter: ", iter, " temp: ", t, " cooling: ", cooling, " metric: scores are: ",scores
                    print "SA Scores with num_points: ", num_points, " iter: ", iter, " temp: ", t, " cooling: ", cooling, " metric: runtimes are: ", runtimes
                    with open(file, 'a') as f:
                        f.write('{},{},{},{},"{}","{}"\n'.format(num_points, iter, t, cooling, scores, runtimes))


def print_ga(ga_results, str):
    # num_points, iter, pop_size, prop_mate, prop_mutate, scores/runtimes, N
    file = "./output/ga_"+str+"_results.csv"
    with open(file, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format("num_points", "iter", "pop", "prop_mate", "prop_mutate", "scores", "runtimes"))
    for num_points, d0 in ga_results.iteritems():
        for iter, d1 in d0.iteritems():
            for pop, d2 in d1.iteritems():
                for prop_mate, d3 in d2.iteritems():
                    for prop_mutate, d4 in d3.iteritems():
                        scores = []
                        runtimes = []
                        for metric, d5 in d4.iteritems():
                            for N, val in d5.iteritems():
                                if metric == "scores":
                                    scores.append(val)
                                elif metric == "runtimes":
                                    runtimes.append(val)
                                else:
                                    print "ERROR: Unexpected metric of: ", metric
                        print "GA Scores with num_points: ", num_points, " iter: ", iter, " pop_size: ", pop, " prop_mate: ", prop_mate, "prop_mutate: ", prop_mutate, " metric: scores are: ",scores
                        print "GA Scores with num_points: ", num_points, " iter: ", iter, " pop_size: ", pop, " prop_mate: ", prop_mate, "prop_mutate: ", prop_mutate, " metric: runtimes are: ", runtimes
                        with open(file, 'a') as f:
                            f.write('{},{},{},{},{},"{}","{}"\n'.format(num_points, iter, pop, prop_mate, prop_mutate, scores, runtimes))


def print_mi(mi_results, str):
    # num_points, iter, sample, prop_keep, scores/runtimes, N
    file = "./output/mi_"+str+"_results.csv"
    with open(file, 'w') as f:
        f.write('{},{},{},{},{},{}\n'.format("num_points", "iter", "samples", "prop_keep", "scores", "runtimes"))
    for num_points, d0 in mi_results.iteritems():
        for iter, d1 in d0.iteritems():
            for sample, d2 in d1.iteritems():
                for prop_keep, d3 in d2.iteritems():
                    scores = []
                    runtimes = []
                    for metric, d4 in d3.iteritems():
                        for N, val in d4.iteritems():
                            if metric == "scores":
                                scores.append(val)
                            elif metric == "runtimes":
                                runtimes.append(val)
                            else:
                                print "ERROR: Unexpected metric of: ", metric
                    print "MI Scores with num_points: ", num_points, " iter: ", iter, " sample: ", sample, " prop_keep: ", prop_keep, " metric: scores are: ",scores
                    print "MI Scores with num_points: ", num_points, " iter: ", iter, " sample: ", sample, " prop_keep: ", prop_keep, " metric: runtimes are: ", runtimes
                    with open(file, 'a') as f:
                        f.write('{},{},{},{},"{}","{}"\n'.format(num_points, iter, sample, prop_keep, scores, runtimes))


def print_path(algo, N):
    path = []
    for x in range(0,N):
        path.append(algo.getOptimal().getDiscrete(x))
    print path

