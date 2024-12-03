import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import bar_label

path = 'figs'


def plot_graphs():

    accuracy = pd.read_csv(f"{path}/accuracy.csv")
    classical_computation_times = pd.read_csv(f"{path}/classical_computation_times.csv")
    classical_hypervolumes = pd.read_csv(f"{path}/classical_hypervolumes.csv")
    computation_times = pd.read_csv(f"{path}/computation_times.csv")
    hypervolumes = pd.read_csv(f"{path}/hypervolumes.csv")

    print(computation_times.head())
    print('-----------------------------')

    print(classical_computation_times.head())
    print('-----------------------------')

    print(hypervolumes.head())
    print('-----------------------------')
    print(classical_hypervolumes.head())
    plot_data(hypervolumes, classical_hypervolumes, "Hypervolume Size", "Hypervolume Sizes for SAMO and NSGA-II")
    #plot_data(computation_times, classical_computation_times, "Computation Time (s)", "Computation Times for SAMO and NSGA-II")
    #plot_accuracy(accuracy)
    #plot_accuracy_problem(accuracy)

def plot_accuracy_problem(accuracy):
    problems = ["dtlz1", "dtlz2", "dtlz5", "dtlz7"]

    objectives = ("nvar-5:nobj-3", "nvar-10:nobj-3", "nvar-10:nobj-5", "nvar-20:nobj-3", "nvar-20:nobj-5", "nvar-20:nobj-10")
    for i in problems:
        data = accuracy[i]
        print(data)
        fig, ax = plt.subplots(layout='constrained', figsize=(8, 6))
        rects = ax.bar(objectives, data, width=0.8 ,label=[round(i, 4) for i in data])
        ax.bar_label(rects, padding=1)
        plt.ylim((0.68, 1.01))
        plt.ylabel('Accuracy')
        plt.title(f"Model Accuracy for {i.upper()} problem")

        plt.show()

def plot_accuracy(accuracy):
    for i in range(len(accuracy)):
        nvar = accuracy['nvar'][i]
        nobj = accuracy['nobj'][i]
        problem = ("DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7")
        data = (accuracy['dtlz1'][i], accuracy['dtlz2'][i], accuracy['dtlz5'][i], accuracy['dtlz7'][i])

        fig, ax = plt.subplots(layout='constrained', figsize=(6, 5))
        rects = ax.bar(problem, data, label=[round(i, 4) for i in data])
        ax.bar_label(rects, padding=1)
        plt.ylim((0.68, 1.01))
        plt.ylabel('Accuracy')
        plt.title(f"Model Accuracy for different problems: n_var={nvar}, n_obj={nobj}")
        plt.show()

def plot_data(mlp_data, stand_data, label, title):
    for i in range(len(mlp_data['nvar'])):
        nvar = mlp_data['nvar'][i]
        nobj = mlp_data['nobj'][i]
        problem = ("DTLZ2", "DTLZ5", "DTLZ7")

        fig, ax = plt.subplots(layout='constrained', figsize=(6, 5))

        x = np.arange(3)
        y1 = [mlp_data['dtlz2'][i], mlp_data['dtlz5'][i], mlp_data['dtlz7'][i]]
        y2 = [stand_data['dtlz2'][i], stand_data['dtlz5'][i], stand_data['dtlz7'][i]]
        width = 0.4

        ax.bar(x - width/2, y1, width, label="SAEA")
        ax.bar(x + width/2, y2, width, label="NSGA-II")

        ax.set_xticks(x, problem)
        ax.set_title(title)
        ax.set_ylabel(label)
        ax.legend()

        rects = ax.patches
        y3 = [round(i, 3) for i in y1 + y2]


        for rect, label in zip(rects, y3):
            height = rect.get_height()

            ax.text(
                rect.get_x() + rect.get_width() / 2, height + 1, label, ha="center", va="bottom"
            )

        plt.show()

        """data = {
            'SAMO': { mlp_data['dtlz2'][i], mlp_data['dtlz5'][i], mlp_data['dtlz7'][i]},
            'NSGA-II': { stand_data['dtlz2'][i], stand_data['dtlz5'][i], stand_data['dtlz7'][i]}
        }

        x = np.arange(len(problem))  # the label locations
        width = 0.4  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained', figsize=(8, 6))


        for attribute, measurement in data.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=1)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(label)
        ax.set_title(f"{title}: nvar={nvar}, nobj={nobj}")
        ax.set_xticks(x + (width /2), problem)
        ax.legend(loc='upper left', ncols=2)
"""
        plt.show()