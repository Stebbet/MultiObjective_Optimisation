import csv
import json
from os import write

import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv.monte_carlo import alpha
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.config import Config

from pygmo import test

from SAMO_NN import *
from Surrogate import *
from plot_graphs import plot_graphs

Config.warnings['not_compiled'] = False

def write_to_csv(computation_times, classical_computation_times, hypervolumes, classical_hypervolumes, accuracy, ref_point_mlp, ref_point_c):
    with open('computation_times.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['nvar', 'nobj', 'dtlz1', 'dtlz2', 'dtlz5', 'dtlz7'])
        csvwriter.writerows(computation_times)

    with open('accuracy.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['nvar', 'nobj', 'dtlz1', 'dtlz2', 'dtlz5', 'dtlz7'])
        csvwriter.writerows(accuracy)

    with open('hypervolumes.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['nvar', 'nobj', 'dtlz1', 'dtlz2', 'dtlz5', 'dtlz7'])
        csvwriter.writerows(hypervolumes)

    with open('classical_computation_times.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['nvar', 'nobj', 'dtlz1', 'dtlz2', 'dtlz5', 'dtlz7'])
        csvwriter.writerows(classical_computation_times)

    with open('classical_hypervolumes.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['nvar', 'nobj', 'dtlz1', 'dtlz2', 'dtlz5', 'dtlz7'])
        csvwriter.writerows(classical_hypervolumes)

    with open('ref_point_mlp.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['nvar', 'nobj', 'dtlz1', 'dtlz2', 'dtlz5', 'dtlz7'])
        csvwriter.writerows(ref_point_mlp)

    with open('ref_point_c.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['nvar', 'nobj', 'dtlz1', 'dtlz2', 'dtlz5', 'dtlz7'])
        csvwriter.writerows(ref_point_c)


def run():
    computation_times = []
    hypervolumes = []

    var = 5
    obj = 3

    ref_point_c = []
    ref_point_mlp = []

    accuracy = []
    classical_hypervolumes = []
    classical_computation_times = []

    generations = 10000
    pop_size = 100

    dtlz1 = SurrogateWithActiveLearning("dtlz1", var, obj, generations, pop_size, True)
    dtlz2 = SurrogateWithActiveLearning("dtlz2", var, obj, generations, pop_size, True)
    dtlz5 = SurrogateWithActiveLearning("dtlz5", var, obj, generations, pop_size, True)
    dtlz7 = SurrogateWithActiveLearning("dtlz7", var, obj, generations, pop_size, False)

    dtlz1.run()
    dtlz2.run()
    dtlz5.run()
    dtlz7.run()

    nsga1_c, nsga1_h = minimise_problem('dtlz1', var, obj, pop_size, generations, dtlz1.ref_point)
    nsga2_c, nsga2_h = minimise_problem('dtlz2', var, obj, pop_size, generations, dtlz2.ref_point)
    nsga5_c, nsga5_h = minimise_problem('dtlz5', var, obj, pop_size, generations, dtlz5.ref_point)
    nsga7_c, nsga7_h = minimise_problem('dtlz7', var, obj, pop_size, generations, dtlz7.ref_point)

    computation_times.append([var, obj, dtlz1.computation_time, dtlz2.computation_time, dtlz5.computation_time, dtlz7.computation_time])
    hypervolumes.append([var, obj, dtlz1.hypervolume, dtlz2.hypervolume, dtlz5.hypervolume, dtlz7.hypervolume])
    accuracy.append([var, obj, dtlz1.surrogate_accuracy, dtlz2.surrogate_accuracy, dtlz5.surrogate_accuracy,
                     dtlz7.surrogate_accuracy])
    classical_hypervolumes.append([var, obj, nsga1_h, nsga2_h, nsga5_h, nsga7_h])
    classical_computation_times.append([var, obj, nsga1_c, nsga2_c, nsga5_c, nsga7_c])

    write_to_csv(computation_times, classical_computation_times, hypervolumes, classical_hypervolumes, accuracy,
                 ref_point_mlp, ref_point_c)

    """for var in [2, 5, 10, 20]:
        for obj in [3, 5, 10]:
            if var <= obj:
                continue

            # More initial computations before fitting as it is a more complex algorithm and needs more training data
            dtlz1 = SurrogateWithActiveLearning("dtlz1", var, obj, generations, pop_size, True)
            dtlz2 = SurrogateWithActiveLearning("dtlz2", var, obj, generations, pop_size, True)
            dtlz5 = SurrogateWithActiveLearning("dtlz5", var, obj, generations, pop_size, True)
            dtlz7 = SurrogateWithActiveLearning("dtlz7", var, obj, generations, pop_size, False)

            dtlz1.run()
            dtlz2.run()
            dtlz5.run()
            dtlz7.run()

            nsga1_c, nsga1_h = minimise_problem('dtlz1', var, obj, pop_size, generations, dtlz1.ref_point)
            nsga2_c, nsga2_h = minimise_problem('dtlz2', var, obj, pop_size, generations, dtlz2.ref_point)
            nsga5_c, nsga5_h = minimise_problem('dtlz5', var, obj, pop_size, generations, dtlz5.ref_point)
            nsga7_c, nsga7_h = minimise_problem('dtlz7', var, obj, pop_size, generations, dtlz7.ref_point)


            print(f'Done: {var, obj}')

            computation_times.append([var, obj, dtlz1.computation_time, dtlz2.computation_time, dtlz5.computation_time, dtlz7.computation_time])
            hypervolumes.append([var, obj, dtlz1.hypervolume, dtlz2.hypervolume, dtlz5.hypervolume, dtlz7.hypervolume])
            accuracy.append([var, obj, dtlz1.surrogate_accuracy, dtlz2.surrogate_accuracy, dtlz5.surrogate_accuracy, dtlz7.surrogate_accuracy])
            classical_hypervolumes.append([var, obj, nsga1_h, nsga2_h, nsga5_h, nsga7_h])
            classical_computation_times.append([var, obj, nsga1_c, nsga2_c, nsga5_c, nsga7_c])

            write_to_csv(computation_times, classical_computation_times, hypervolumes, classical_hypervolumes, accuracy, ref_point_mlp, ref_point_c)
    """

def plot_dtlz():

    problem = 'dtlz2'
    generations = 10000


    var5obj3 = SurrogateWithActiveLearning(problem, 5, 3, generations, 100)

    var10obj3 = SurrogateWithActiveLearning(problem, 10, 3, generations, 100)
    var20obj3 = SurrogateWithActiveLearning(problem, 20, 3, generations, 100)
    var10obj5 = SurrogateWithActiveLearning(problem, 10, 5, generations, 100)
    var20obj5 = SurrogateWithActiveLearning(problem, 20, 5, generations, 100)
    var20obj10 = SurrogateWithActiveLearning(problem, 20, 10, generations, 100)

    var5obj3.run()
    var10obj3.run()
    var20obj3.run()
    var10obj5.run()
    var20obj5.run()
    var20obj10.run()

    iterations = [i * (var5obj3.n_generations / len(var10obj3.accuracy_plot)) for i in range(len(var5obj3.accuracy_plot))]

    plt.figure(figsize=(7,5))
    plt.plot(var5obj3.accuracy_plot[:,0], var5obj3.accuracy_plot[:,1], label="var5_obj3")
    plt.plot(var10obj3.accuracy_plot[:,0], var10obj3.accuracy_plot[:,1],label="var10_obj3")
    plt.plot(var20obj3.accuracy_plot[:,0], var20obj3.accuracy_plot[:,1], label="var20_obj3")
    plt.plot(var10obj5.accuracy_plot[:,0], var10obj5.accuracy_plot[:,1], label="var10_obj5")
    plt.plot(var20obj5.accuracy_plot[:,0], var20obj5.accuracy_plot[:,1], label="var20_obj5")
    plt.plot(var20obj10.accuracy_plot[:,0], var20obj10.accuracy_plot[:,1], label="var20_obj10")
    plt.xlabel('Generations')
    plt.ylabel('Accuracy')
    plt.title(f"{problem.upper()} Test Accuracy for different number of variables and objectives")
    plt.legend()
    plt.show()





if __name__ == '__main__':
    #plot_dtlz()

    #plot_graphs()
    #run()


    dtlz2 = SurrogateWithActiveLearning("dtlz5", 20, 3, 10000, 100, True)
    result = dtlz2.run(retrain_interval=10)

    #print(f"Surrogate Computation Time: {dtlz2.computation_time}")
    #print(f"Surrogate Hypervolume: {dtlz2.hypervolume}")

