# traveling salesman algorithm implementation in jython
# This also prints the index of the points of the shortest route.
# To make a plot of the route, write the points at these indexes
# to a file and plot them in your favorite tool.
import sys
import os
import time
import collections
import math
import helpers

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



"""
Commandline parameter(s):
    none
"""


num_trials = 5

input_curve = False     # Used to do sweeps of num points
iter_curve = True      # Used to do sweeps of number of iterations
param_curve = False     # Used to do sweeps of parameter settings

do_rhc = True
do_sa = True
do_ga = True
do_mi = True
# Need to write function to capture the key values for each algo (score, time taken, # of iterations)

if input_curve:
    num_points = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
else:
    num_points = [50]

if iter_curve:
    hc_iter = [100, 1000, 10000, 100000, 1000000]
    sa_iter = [100, 1000, 10000, 100000, 1000000]
    ga_iter = [10, 100, 1000, 10000]
    mi_iter = [10, 100, 1000, 10000]
else:
    hc_iter = [100000]
    sa_iter = [100000]
    ga_iter = [1000]
    mi_iter = [1000]    # Could actually be >10000

if param_curve:
    # RHC
    # SA
    sa_temp = [1E10, 1E11, 1E12, 1E13]
    sa_cooling = [1 - 10**-1, 1 - 10**-2, 1 - 10**-3, 1 - 10**-4]
    # GA
    ga_pop = [1000, 1500, 2000, 2500]
    ga_mate = [0.55, 0.65, 0.75, 0.85, 0.95]
    ga_mutate = [0.05, 0.075, 0.1, 0.125, 0.15]
    # MIMIC
    mi_samples = [100, 300, 500, 700, 900]
    mi_keep_prop = [0.1, 0.2, 0.3, 0.4]
else:
    # Determined
    sa_temp = [1E13]
    sa_cooling = [0.999]

    # Determined
    ga_pop = [2500]
    ga_mate = [0.85]
    ga_mutate = [0.125]

    # Determined
    mi_samples = [300]
    mi_keep_prop = [0.2]

# Do we want to use a dataframe to store results?
    # Perhaps do something like one row per set of attr values and 1 col per N so we can slice to easily make means
    #

rhc_scores = []
rhc_times = []
rhc_params = []
sa_scores = []
sa_times = []
sa_params = []
ga_scores = []
ga_times = []
ga_params = []
mi_scores = []
mi_times = []
mi_params = []

nested_dict = lambda: collections.defaultdict(nested_dict)

rhc_results = nested_dict()
sa_results = nested_dict()
ga_results = nested_dict()
mi_results = nested_dict()

# Two tasks: sweep params to figure out best settings, then run sweeps of iters, etc.
for trial_num in range(num_trials):
    print "On trial num: ", trial_num, " of total of ", num_trials
    for N in num_points:
        print "\tUsing num_points: ", N
        # set N value.  This is the number of points
        # N = 50
        # TODO: Set a seed to get consistent results
        random = Random()

        points = [[0 for x in xrange(2)] for x in xrange(N)]
        for i in range(0, len(points)):
            points[i][0] = random.nextDouble()
            points[i][1] = random.nextDouble()

        ef = TravelingSalesmanRouteEvaluationFunction(points)
        odd = DiscretePermutationDistribution(N)
        nf = SwapNeighbor()
        mf = SwapMutation()
        cf = TravelingSalesmanCrossOver(ef)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

        # TODO: Need to find good way to store results when sweeping; maybe store param values along with them

        # Randomized Hill Climbing
        # Default: iter count of 100000 was sufficient
        if do_rhc:
            rhc = RandomizedHillClimbing(hcp)
            for iter in hc_iter:
                score, call_count, runtime = helpers.eval_algo(ef, rhc, iter)
                rhc_scores.append(score)
                rhc_times.append(runtime)
                sa_params.append([iter, N])
                rhc_results[N][iter]['scores'][trial_num] = score
                rhc_results[N][iter]['runtimes'][trial_num] = runtime
                # print("RHC best score: ", score, " in time: ", runtime, " with iters: ", iter)
            print "\tFinished RHC"

        # Simulated Annealing
        # 100k sufficient
        if do_sa:
            for t in sa_temp:
                for cooling in sa_cooling:
                    sa = SimulatedAnnealing(t, cooling, hcp)
                    for iter in sa_iter:
                        score, call_count, runtime = helpers.eval_algo(ef, sa, iter)
                        sa_params.append([t, cooling, iter, N])
                        sa_scores.append(score)
                        sa_times.append(runtime)
                        sa_results[N][iter][t][cooling]['scores'][trial_num] = score
                        sa_results[N][iter][t][cooling]['runtimes'][trial_num] = runtime
                        # print("SA best score: ", score, " in time: ", runtime, " with iters: ", iter)
            print "\tFinished SA"

        # Genetic Algo
        if do_ga:
            # pop size, num to mate, num to mutate
            # Let's make these proportions of pop size
            for pop_size in ga_pop:
                for prop_mate in ga_mate:
                    num_mate = int(math.floor(prop_mate*pop_size))
                    for prop_mutate in ga_mutate:
                        num_mutate = int(math.floor(prop_mutate*pop_size))
                        ga = StandardGeneticAlgorithm(pop_size, num_mate, num_mutate, gap)
                        for iter in ga_iter:
                            score, call_count, runtime = helpers.eval_algo(ef, ga, iter)
                            ga_params.append([pop_size, prop_mate, prop_mutate, iter, N])
                            ga_scores.append(score)
                            ga_times.append(runtime)
                            ga_results[N][iter][pop_size][prop_mate][prop_mutate]['scores'][trial_num] = score
                            ga_results[N][iter][pop_size][prop_mate][prop_mutate]['runtimes'][trial_num] = runtime
                # print("GA best score: ", score, " in time: ", runtime, " with iters: ", iter)
            print "\tFinished GA"

        # for mimic we use a sort encoding
        if do_mi:
            ef = TravelingSalesmanSortEvaluationFunction(points);
            fill = [N] * N
            ranges = array('i', fill)
            odd = DiscreteUniformDistribution(ranges);
            df = DiscreteDependencyTree(.1, ranges);
            pop = GenericProbabilisticOptimizationProblem(ef, odd, df);

            # Note: 10000 best fit but takes a long time
            for sample in mi_samples:
                for keep_prop in mi_keep_prop:
                    keep = int(math.floor(keep_prop*sample))
                    mimic = MIMIC(sample, keep, pop)
                    for iter in mi_iter:
                        score, call_count, runtime = helpers.eval_algo(ef, mimic, iter)
                        mi_scores.append(score)
                        mi_times.append(runtime)
                        mi_params.append([sample, keep_prop, iter, N])
                        mi_scores.append(score)
                        mi_times.append(runtime)
                        mi_results[N][iter][sample][keep_prop]['scores'][trial_num] = score
                        mi_results[N][iter][sample][keep_prop]['runtimes'][trial_num] = runtime
            # print("GA best score: ", score, " in time: ", runtime, " with iters: ", iter)
                # print("MIMIC Inverse of Distance: ", score, " in time: ", runtime, " with iters: ", iter)
            print "\tFinished MIMIC"

print("RHC Scores: ",rhc_scores)
print("RHC Times: ",rhc_times)
print("SA Scores: ",sa_scores)
print("SA Times: ",sa_times)
print("GA Scores: ",ga_scores)
print("GA Times: ",ga_times)
print("MIMIC Scores: ",mi_scores)
print("MIMIC Times: ",mi_times)

if input_curve:
    str = "tsp_size"
if iter_curve:
    str = "tsp_iter"
if param_curve:
    str = "tsp"

if do_rhc:
    helpers.print_rhc(rhc_results, str)
if do_sa:
    helpers.print_sa(sa_results, str)
if do_ga:
    helpers.print_ga(ga_results, str)
if do_mi:
    helpers.print_mi(mi_results, str)




# Results w/ defaults
# ('RHC best score: ', 0.08952243166751593, ' in time: ', 0.039999961853027344, ' with iters: ', 1000)
# ('RHC best score: ', 0.13126834661422446, ' in time: ', 0.02200007438659668, ' with iters: ', 10000)
# ('RHC best score: ', 0.13489903808101927, ' in time: ', 0.05299997329711914, ' with iters: ', 100000)
# ('RHC best score: ', 0.13489903808101927, ' in time: ', 0.43899989128112793, ' with iters: ', 1000000)
# ('SA best score: ', 0.03713654603125923, ' in time: ', 0.0009999275207519531, ' with iters: ', 1000)
# ('SA best score: ', 0.039665438389516416, ' in time: ', 0.007999897003173828, ' with iters: ', 10000)
# ('SA best score: ', 0.11708985738032143, ' in time: ', 0.07699990272521973, ' with iters: ', 100000)
# ('SA best score: ', 0.11708985738032143, ' in time: ', 1.0250000953674316, ' with iters: ', 1000000)
# ('GA best score: ', 0.09727963380903194, ' in time: ', 0.10700011253356934, ' with iters: ', 10)
# ('GA best score: ', 0.13993093756504102, ' in time: ', 0.4609999656677246, ' with iters: ', 100)
# ('GA best score: ', 0.1470453509298889, ' in time: ', 4.0959999561309814, ' with iters: ', 1000)
# ('GA best score: ', 0.1459585802792045, ' in time: ', 40.782999992370605, ' with iters: ', 10000)
# ('MIMIC Inverse of Distance: ', 0.0540017595793275, ' in time: ', 0.29399991035461426, ' with iters: ', 10)
# ('MIMIC Inverse of Distance: ', 0.07583076825671756, ' in time: ', 1.9000000953674316, ' with iters: ', 100)
# ('MIMIC Inverse of Distance: ', 0.09763027807207111, ' in time: ', 17.60099983215332, ' with iters: ', 1000)
# ('MIMIC Inverse of Distance: ', 0.11647279252094686, ' in time: ', 171.42300009727478, ' with iters: ', 10000)
