import time

from matplotlib import pyplot as plt
from numpy.f2py.crackfortran import verbose
from pygmo.core import hypervolume, hvwfg, hv3d
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.problems.static import StaticProblem
from pymoo.optimize import minimize
import numpy as np
from pymoo.visualization.pcp import PCP
from pymoo.visualization.scatter import Scatter
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from scipy.stats.qmc import LatinHypercube


def minimise_problem(problem_name, var, obj, pop_size, generations, ref_point):

    temp = SurrogateWithActiveLearning('dtlz1', 5, 3, 0, 0, False)
    temp.ref_point = ref_point

    problem = get_problem(problem_name, n_var=var, n_obj=obj)
    algorithm = NSGA2(pop_size=pop_size)
    start = time.time()
    res = minimize(problem, algorithm=algorithm, termination=('n_gen', generations), verbose=False)
    computation_time = time.time() - start
    hypervolume = temp.get_hypervolume(res.F)

    return computation_time, hypervolume


class SurrogateProblem(StaticProblem):
    def __init__(self, problem, mlp_model, nvar, nobj):
        super().__init__(problem, n_var=nvar, n_obj=nobj, xl=0, xu=1)
        self.surrogate = mlp_model

    def _evaluate(self, x, out, *args, **kwargs):
        # Use the MLP surrogate model to predict objective values
        predictions = self.surrogate.predict(x)
        out["F"] = predictions  # Set the predicted objectives



class SurrogateWithActiveLearning():
    def __init__(self, problem_name:str, n_var:int, n_obj:int, n_generations:int=1000, pop_size:int=100, show_pareto=False):
        self.problem_name = problem_name
        self.n_obj = n_obj
        self.n_var = n_var
        self.n_generations = n_generations

        self.algorithm = NSGA2(pop_size=pop_size)

        self.real_problem = get_problem(problem_name, n_var=n_var, n_obj=n_obj)
        self.show_pareto = show_pareto
        surrogate = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(100, 100), max_iter=1000, random_state=1, warm_start=True)
        #self.X_train = np.random.rand(200, self.n_var)
        self.X_train = LatinHypercube(self.n_var, seed=round(time.time())).random(n=200)
        self.y_train = self.real_problem.evaluate(self.X_train)

        surrogate.fit(self.X_train, self.y_train)

        self.surrogate_problem = SurrogateProblem(self.real_problem, mlp_model=surrogate, nvar=self.n_var, nobj=self.n_obj)
        self.algorithm.setup(self.surrogate_problem, termination=('n_gen', self.n_generations), seed=1, verbose=False)

        self.ref_point = None

        self.hypervolume = 0.0
        self.training_time = []

        self.computation_time = 0.0
        self.surrogate_accuracy = 0.0
        self.accuracy_plot = []

        self.my_path = f"/Users/samtebbet/Library/CloudStorage/OneDrive-UniversityofExeter/Documents/Advanced Computer Science/Mult-Objective Optimisation and Decision Learning/Coursework/CondaProject/figures/{self.problem_name}/var_{self.n_var}/obj_{self.n_obj}"
        self.mkdir_p()

        self.plot_objective_vectors(self.y_train, f"{self.problem_name.upper()}: n_var={self.n_var}, n_obj={self.n_obj}: Initial Data", "InitialData")



    def run(self, retrain_interval=10, error_threshold=0.1):
    # Storage for high-error samples
        new_samples = []
        start_time = time.time()

        for gen in range(1, self.n_generations + 1):

            population = self.algorithm.ask()

            #Check error at every retrain interval
            if gen % retrain_interval == 0:
                # Perform one iteration with the actual FE evaluations
                self.algorithm.evaluator.eval(self.real_problem, population)
                self.algorithm.tell(infills=population)
                res = self.algorithm.result()

                for x in res.X:
                    # Evaluate the actual DTLZ2 objective to compare with surrogate prediction
                    true_objective = self.real_problem.evaluate(np.array([x]))[0]
                    predicted_objective = self.surrogate_problem.surrogate.predict([x])[0]

                    # Calculate the mean absolute error between true and predicted objectives
                    error = np.mean(np.abs(true_objective - predicted_objective))

                    # If error exceeds threshold, add the point to retraining data
                    if error > error_threshold:
                        new_samples.append((x, true_objective))


                if new_samples:
                    X_new, Y_new = zip(*new_samples)

                    self.X_train = np.vstack((self.X_train, np.array(X_new)))
                    self.y_train = np.vstack((self.y_train, np.array(Y_new)))

                    # Limit the training dataset to 1000 inputs to avoid very long training times when the error threshold is constantly exceeded
                    # and also get rid of redundant training data
                    if self.X_train.shape[0] > 1000:
                        self.X_train = self.X_train[-1000:]
                        self.y_train = self.y_train[-1000:]

                    x_tr, x_te, y_tr, y_te = train_test_split(self.X_train, self.y_train, random_state=1)

                    # Retrain the model with the expanded dataset
                    training_start_time = time.time()
                    self.surrogate_problem.surrogate.fit(x_tr, y_tr)
                    self.surrogate_accuracy = self.surrogate_problem.surrogate.score(x_te, y_te)
                    self.accuracy_plot.append([gen, self.surrogate_accuracy])
                    self.training_time.append(time.time() - training_start_time)
                new_samples.clear()

            else:
                # MLP iterations
                self.algorithm.evaluator.eval(self.surrogate_problem, population)
                self.algorithm.tell(infills=population)
                res = self.algorithm.result()

            self.computation_time = time.time() - start_time

        self.get_hypervolume(res.pop.get('F'))
        if len(self.accuracy_plot) > 1:
            self.plot_surrogate_accuracy(f"{self.problem_name.upper()}: n_var={self.n_var}, n_obj={self.n_obj}: Surrogate Accuracy Over Time")
            self.plot_training_time()

        self.plot_objective_vectors(res.F, f"{self.problem_name.upper()}: n_var={self.n_var}, n_obj={self.n_obj}: Objective Vectors - {self.n_generations}", "ObjectiveVectors")

        print(f"DONE {self.problem_name}: {self.n_var}, {self.n_obj} - {self.computation_time}s")

        return res

    def get_hypervolume(self, O):
        if self.n_obj > 3:
            hv = hypervolume(O)
            if self.ref_point is None:
                temp_ref = hv.refpoint(10)
                self.ref_point = [i * (i / 5) for i in temp_ref]
            self.hypervolume = hv.compute(self.ref_point, hv_algo=hvwfg())
        else:
            hv = hypervolume(O)
            if self.ref_point is None:
                self.ref_point = hv.refpoint(2)
            self.hypervolume = hv.compute(self.ref_point, hv_algo=hv3d())
        return self.hypervolume

    def plot_training_time(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.accuracy_plot[:, 0], self.training_time)
        plt.xlabel('Generations')
        plt.ylabel(r'Training Time (s)')
        plt.title(f"{self.problem_name.upper()}: Training Time over generations")
        plt.savefig(f"{self.my_path}/TrainingTime.png")
        plt.close('all')

    def plot_surrogate_accuracy(self, title):

        #iterations = [i * (self.n_generations / len(self.accuracy_plot)) for i in range(len(self.accuracy_plot))]
        self.accuracy_plot = np.array(self.accuracy_plot)
        plt.figure(figsize=(10, 8))
        plt.plot(self.accuracy_plot[:,0], self.accuracy_plot[:,1])
        plt.xlabel('Generations')
        plt.ylabel(r'Test Accuracy')
        plt.title(title)
        plt.savefig(f"{self.my_path}/AccuracyVsGeneration.png")
        plt.close('all')


    def plot_objective_vectors(self, O, title, path_title):
        if len(O[1]) > 3:
            # Plot a PCP if the number of objectives is greater than 3
            plot = PCP(title=(title, {'pad': 30}), figsize=(6, 5))
            plot.set_axis_style(color="grey", alpha=0.5)
            plot.add(O, color="grey", alpha=0.3)
        else:
            plot = Scatter(angle=(20, 20),
                           title=title,
                           tight_layout=True, figsize=(10, 10))
            if self.show_pareto:
                plot.add(self.real_problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
            plot.add(O)

        plot.save(f"{self.my_path}/{path_title}.png")
        plot.__del__()


    def mkdir_p(self):
        '''Creates a directory. equivalent to using mkdir -p on the command line'''

        from errno import EEXIST
        from os import makedirs, path

        try:
            makedirs(self.my_path)
        except OSError as exc:  # Python >2.5
            if exc.errno == EEXIST and path.isdir(self.my_path):
                pass
            else:
                raise
