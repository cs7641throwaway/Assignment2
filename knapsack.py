import sys
import os
import time
import math
import collections
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
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
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
    hc_iter = [100, 1000, 10000, 100000, 200000]
    sa_iter = [100, 1000, 10000, 100000, 200000]
    ga_iter = [10, 100, 1000, 10000]
    mi_iter = [10, 100, 1000, 10000]
else:
    hc_iter = [200000]
    sa_iter = [200000]
    ga_iter = [1000]
    mi_iter = [1000]    # Could actually be >10000

if param_curve:
    # RHC
    # SA
    sa_temp = [10, 100, 1000, 1000]
    sa_cooling = [.85, .9, .95, .99]
    # GA
    ga_pop = [100, 150, 200, 250, 300]
    ga_mate = [0.55, 0.65, 0.75, 0.85, 0.95]
    ga_mutate = [0.05, 0.075, 0.1, 0.125, 0.15]
    # MIMIC
    mi_samples = [100, 150, 200, 250, 300]
    mi_keep_prop = [0.3, 0.4, 0.5, 0.6, 0.7]
else:
    # Determined
    sa_temp = [1000]
    sa_cooling = [0.99]

    # Determined
    ga_pop = [300]
    ga_mate = [0.85]
    ga_mutate = [0.05]

    # Determined
    mi_samples = [300]
    mi_keep_prop = [0.3]

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

for trial_num in range(num_trials):
    print "On trial num: ", trial_num, " of total of ", num_trials
    for N in num_points:
        print "\tUsing num_points: ", N
        # set N value.  This is the number of points
        # Random number generator */
        random = Random()
        # The number of items
        NUM_ITEMS = N
        # The number of copies each
        COPIES_EACH = 4
        # The maximum weight for a single element
        MAX_WEIGHT = 50
        # The maximum volume for a single element
        MAX_VOLUME = 50
        # The volume of the knapsack
        KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

        # create copies
        fill = [COPIES_EACH] * NUM_ITEMS
        copies = array('i', fill)

        # create weights and volumes
        fill = [0] * NUM_ITEMS
        weights = array('d', fill)
        volumes = array('d', fill)
        for i in range(0, NUM_ITEMS):
            weights[i] = random.nextDouble() * MAX_WEIGHT
            volumes[i] = random.nextDouble() * MAX_VOLUME


        # create range
        fill = [COPIES_EACH + 1] * NUM_ITEMS
        ranges = array('i', fill)

        ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = UniformCrossOver()
        df = DiscreteDependencyTree(.1, ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

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

        if do_mi:
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
    str = "knapsack_size"
if iter_curve:
    str = "knapsack_iter"
if param_curve:
    str = "knapsack"

if do_rhc:
    helpers.print_rhc(rhc_results, str)
if do_sa:
    helpers.print_sa(sa_results, str)
if do_ga:
    helpers.print_ga(ga_results, str)
if do_mi:
    helpers.print_mi(mi_results, str)

