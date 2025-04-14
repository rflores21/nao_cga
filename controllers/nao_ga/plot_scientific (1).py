import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv

import scienceplots
from scipy.optimize import curve_fit
import json

def plot(FILE_NAME, Y_LABEL="Total Fitness", Y_LIM=(10, 85), X_LIM=(0, 56), LINE_LABEL="Total Fitness", Y_TICKS = np.arange(0, 201, 20), plot_sigmoid=True, baseline=None):
    DPI = 800
    RESULT_FOLDER = "RESULT_PLOTS"
    PATH = "data"
    CSV_PATH = os.path.join(PATH, FILE_NAME)


    def create_result_folder(folder_name=RESULT_FOLDER):
        """Ensure the result folder exists."""
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        return folder_name


    def save_plot_to_folder(file_name, folder_name=RESULT_FOLDER):
        """Save the plot to a folder."""
        full_path = os.path.join(folder_name, file_name + ".png")
        plt.savefig(full_path, dpi=DPI, bbox_inches='tight')
        plt.close()

    # ------------------- LOAD DATA -------------------
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV data file {CSV_PATH} not found.")
        exit()

    all_generations = []
    fitness_values = []

    with open(CSV_PATH, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            all_generations.append(int(row[1]))
            fitness_values.append(float(row[3]))

    # Compute percentiles
    fitness_values = np.array(fitness_values)
    percentile_25 = np.percentile(fitness_values, 25)
    percentile_50 = np.percentile(fitness_values, 50)
    percentile_75 = np.percentile(fitness_values, 75)

    # ------------------- PLOT -------------------
    # plt.style.use(['science', 'ieee'])
    # plt.style.use(['science', 'ieee', 'no-latex'])
    plt.style.use(['science', 'no-latex'])

    plt.figure(dpi=DPI)

    plt.plot(all_generations, fitness_values, label=LINE_LABEL)

    # Plot sigmoid function
    if plot_sigmoid:

        def sigmoid(x, L ,x0, k, b):
            y = L / (1 + np.exp(-k*(x-x0))) + b
            return (y)

        # Initial guess for the parameters [L, x0, k, b]
        # L: maximum value of the sigmoid curve
        # x0: the value of the sigmoid's midpoint
        # k: the steepness of the curve
        # b: the minimum value of the sigmoid curve
        p0 = [max(fitness_values), np.median(all_generations), 1, min(fitness_values)]
        # p0 = [115, 0, 0.2, -40]

        popt, pcov = curve_fit(sigmoid, all_generations, fitness_values, p0, method='lm')

        x = np.linspace(0, 56, 57)
        y = sigmoid(x, *popt)
        plt.plot(x, y, label="Best Fit Curve", color='black', linestyle='dotted')

    if baseline:
        # add a horizontal line equal to the value of baseline
        plt.axhline(y=baseline, color='orange', linestyle='-', label="Baseline")

    # x = np.linspace(0, 56, 2*57)
    # # print(x)
    # L = 115
    # x0 = 0
    # k = 0.2
    # b = -40


    # y = L / (1 + np.exp(-k*(x-x0))) + b
    # # print(y)
    # plt.plot(x, y, label="Sigmoid Function", color='green', linestyle='dashed')

    

    # plt.plot(all_generations,
    #          [percentile_50] * len(all_generations),
    #          color='black',
    #          label="Median Fitness")

    # plt.fill_between(all_generations,
    #                  [percentile_25] * len(all_generations),
    #                  [percentile_75] * len(all_generations),
    #                  color='gray',
    #                  alpha=0.4,
    #                  label="25th-75th Percentile")

    # Enhancing plot aesthetics
    plt.xlabel("Generation")
    plt.ylabel(Y_LABEL)
    # plt.yticks(Y_TICKS)
    plt.ylim(Y_LIM)

    major_ticks = [1, 10, 20, 30, 40, 50, 60]
    minor_ticks = [1] + list(range(2, 56, 2))

    ax = plt.gca()  # Get current axis
    # ax.set_xticks(major_ticks)  # Set major ticks
    plt.xlim(X_LIM)
    # ax.set_xticks(minor_ticks, minor=True)  # Set minor ticks explicitly

    plt.legend(loc='center right')

    # Save and close
    create_result_folder()
    save_plot_to_folder(FILE_NAME.replace(".csv", "_graph"))

    print(f"✅ Plot saved in RESULT_PLOTS. Run open RESULT_PLOTS/{FILE_NAME} to view.")


def average_every_n_values(input_file, output_file, n=10):
        """Average every n values in the input CSV file and write to the output CSV file."""
        with open(input_file, 'r') as f:
            data = list(csv.reader(f))

        averaged_data = []
        for i in range(0, len(data), n):
            chunk = data[i:i + n]
            avg_iteration = sum(int(row[0]) for row in chunk) // len(chunk)
            avg_fitness = sum(float(row[1]) for row in chunk) / len(chunk)
            averaged_data.append([i//n, avg_fitness])

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "fitness"])
            writer.writerows(averaged_data)


def convert_json_to_csv(json_file, csv_file):
        """Convert JSON file to CSV."""
        with open(json_file, 'r') as f:
            data = json.load(f)

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "fitness"])
            for entry in data:
                writer.writerow([entry["iteration"], entry["fitness"]])

def get_average_random_fitness():
    csv_file = "ds fitness functions/results_random_iterations.csv"

    # Calculate the average fitness value
    with open(csv_file, 'r') as f:
        data = list(csv.reader(f))
        fitness_values = [float(row[1]) for row in data[1:]]  # Skip header
        average_fitness = sum(fitness_values) / len(fitness_values)

    print(f"Average fitness value: {average_fitness}")

if __name__ == "__main__":
    # 590 generations
    # y value is from 0 to 3
    plot("evolution_data.csv", Y_LIM=(0, 100), X_LIM=(0, 590), LINE_LABEL="Fitness", plot_sigmoid=False)
    # plot("evolution_data.csv", plot_sigmoid=False)
    exit()
    
    # plot("fitness.csv", LINE_LABEL="Mean Fitness")
    # plot("fitness.csv", Y_LIM=(10, 85), LINE_LABEL="Mean Fitness", baseline=18.009225949783293)
    # plot("win_percentage.csv", "Percentage", (0, 37), (0, 55), "Win Percentage", Y_TICKS=np.arange(0, 37, 10))

    
    # json_file = "ds fitness functions/results_random_iterations.json"
    # csv_file = "ds fitness functions/results_random_iterations.csv"
    # convert_json_to_csv(json_file, csv_file)
    # plot("results_random_iterations.csv", LINE_LABEL="Random Iterations Fitness", Y_LIM=(-100, 200))

    
    # Convert JSON to CSV
    # json_file = "ds fitness functions/results_random_iterations.json"
    csv_file = "ds fitness functions/results_random_iterations.csv"
    # convert_json_to_csv(json_file, csv_file)

    # Average every 10 values
    averaged_csv_file = "ds fitness functions/results_random_iterations_averaged.csv"
    # average_every_n_values(csv_file, averaged_csv_file, n=30)

    # Plot the averaged data
    plot("results_random_iterations_averaged.csv", LINE_LABEL="Random Iterations Fitness (Averaged)", Y_LIM=(10, 85), plot_sigmoid=False)