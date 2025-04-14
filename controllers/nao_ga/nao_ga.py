from controller import Robot, GPS, Supervisor
import math
import random
import pickle
import os
import csv
import cma
import numpy as np

# Constants
NUM_GENERATIONS = 15
POPULATION_SIZE = 50
MUTATION_RATE = 0.03
PARAMS = 10
TIME_STEP = 20
HEIGHT_WEIGHT = 0.2

JOINT_LIMITS = {
    "RHipRoll": (-0.738274, 0.449597),
    "RHipPitch": (-1.77378, 0.48398),
    "RKneePitch": (-0.0923279, 2.11255),
    "RAnklePitch": (-1.1863, 0.932006),
    "RAnkleRoll": (-0.768992, 0.397935),
    "LHipRoll": (-0.379435, 0.79046),
    "LHipPitch": (-1.77378, 0.48398),
    "LKneePitch": (-0.0923279, 2.11255),
    "LAnklePitch": (-1.18944, 0.922581),
    "LAnkleRoll": (-0.39788, 0.769001)
}

robot = Supervisor()
gps = robot.getDevice("gps")
gps.enable(TIME_STEP)

motor_names = ["RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
               "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll"]
motors = [robot.getDevice(name) for name in motor_names]
for motor in motors:
    motor.setPosition(0.0)

robot.getDevice("RShoulderPitch").setPosition(1.1)
robot.getDevice("LShoulderPitch").setPosition(1.1)

root = robot.getRoot()
children_field = root.getField("children")
robot_node = next((children_field.getMFNode(i) for i in range(children_field.getCount())
                   if children_field.getMFNode(i).getTypeName() == "Nao"), None)
translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")
initial_position = translation_field.getSFVec3f()
initial_rotation = rotation_field.getSFRotation()

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def reset_robot():
    for motor in motors:
        motor.setPosition(0.0)
    translation_field.setSFVec3f(initial_position)
    rotation_field.setSFRotation(initial_rotation)
    for _ in range(3):
        robot.step(TIME_STEP)

def evaluate(individual):
    reset_robot()
    start_time = robot.getTime()
    max_distance, height_sum, height_samples = 0.0, 0.0, 0
    initial_pos = gps.getValues()
    f = 1.0

    while robot.getTime() - start_time < 20.0:
        time = robot.getTime()
        for i, motor in enumerate(motors):
            position = (individual["amplitude"][i] * math.sin(2.0 * math.pi * f * time + individual["phase"][i])
                        + individual["offset"][i])
            motor_name = motor.getName()
            if motor_name in JOINT_LIMITS:
                min_limit, max_limit = JOINT_LIMITS[motor_name]
                position = clamp(position, min_limit, max_limit)
            motor.setPosition(position)

        robot.step(TIME_STEP)
        current_pos = gps.getValues()
        distance = math.sqrt((current_pos[0] - initial_pos[0]) ** 2 + (current_pos[2] - initial_pos[2]) ** 2)
        max_distance = max(max_distance, distance)
        height_sum += current_pos[1]
        height_samples += 1

    avg_height = height_sum / height_samples if height_samples > 0 else 0.0
    individual["fitness"] = max_distance + avg_height * HEIGHT_WEIGHT
    return individual["fitness"]

def objective_function(x):
    individual = {
        "amplitude": x[:PARAMS],
        "phase": x[PARAMS:2*PARAMS],
        "offset": x[2*PARAMS:],
        "fitness": 0.0
    }
    fitness = evaluate(individual)

    # Save to CSV
    data_directory = "data"
    csv_file_name = os.path.join(data_directory, "evolution_data.csv")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    if not os.path.exists(csv_file_name):
        with open(csv_file_name, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Generation', 'Fitness', 'Amplitude', 'Phase', 'Offset'])

    with open(csv_file_name, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            objective_function.generation,
            fitness,
            individual['amplitude'],
            individual['phase'],
            individual['offset']
        ])

    return -fitness

objective_function.generation = 0

if __name__ == "__main__":
    initial_guess = [0.25] * PARAMS * 3
    sigma0 = 0.1
    bounds_lower = [0.0] * PARAMS + [0.0] * PARAMS + [-0.5] * PARAMS
    bounds_upper = [0.5] * PARAMS + [2 * np.pi] * PARAMS + [0.5] * PARAMS

    opts = {
        'popsize': 20,
        'bounds': [bounds_lower, bounds_upper],
        'verb_disp': 1,
        'maxiter': 1000,
        'tolx': 1e-8,
        'tolfun': 1e-12
    }

    es = cma.CMAEvolutionStrategy(initial_guess, sigma0, opts)
    while not es.stop():
        solutions = es.ask()
        objective_function.generation += 1
        fitnesses = [objective_function(x) for x in solutions]
        es.tell(solutions, fitnesses)
        es.disp()

    best_solution = es.result.xbest
    best_fitness = -es.result.fbest

    print("Best Fitness:", best_fitness)
    print("Best Solution:", best_solution)
