from controller import Robot, GPS, Supervisor
import math
import random
import pickle
import os
import csv
import cma


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

# Initialize Supervisor and Devices
robot = Supervisor()
gps = robot.getDevice("gps")
gps.enable(TIME_STEP)

# Motor Initialization
motor_names = ["RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
               "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll"]
motors = [robot.getDevice(name) for name in motor_names]
for motor in motors:
    motor.setPosition(0.0)
    
robot.getDevice("RShoulderPitch").setPosition(1.1)
robot.getDevice("LShoulderPitch").setPosition(1.1)
    
# Get the initial position and rotation of the robot
root = robot.getRoot()
children_field = root.getField("children")
robot_node = next((children_field.getMFNode(i) for i in range(children_field.getCount())
                   if children_field.getMFNode(i).getTypeName() == "Nao"), None)
translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")
initial_position = translation_field.getSFVec3f()
initial_rotation = rotation_field.getSFRotation()   
 
def get_joint_limits():
    for name in motor_names:
        motor = robot.getDevice(name)
        min_position = motor.getMinPosition()
        max_position = motor.getMaxPosition()
        print(f"{name}: min_position={min_position}, max_position={max_position}")
    robot.step(TIME_STEP)  # Run a single simulation step to initialize devices


# Generate an individual with bounded knee constraints
def create_individual():
    return {
        "amplitude": [random.uniform(0, 0.5) for _ in range(PARAMS)],
        "phase": [random.uniform(0, 2 * math.pi) for _ in range(PARAMS)],
        "offset": [random.uniform(-0.5, 0.5) for _ in range(PARAMS)],
        "fitness": 0.0
    }


# Clamp values within a specific range
def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

#Loads the state of the GA from a file
def load_state(filename):
    with open(filename, 'rb') as file:
        state = pickle.load(file)
    print(f"State loaded from {filename}")
    return state['populations'], state['best_individuals'], state['best_overall'], state['generation']

#Saves the state of the GA to a file
def save_state(filename, populations, best_individuals, best_overall, generation):
    state = {
        'populations': populations,
        'best_individuals': best_individuals,
        'best_overall': best_overall,
        'generation': generation
    }
    with open(filename, 'wb') as file:
        pickle.dump(state, file)
    print(f"State saved to {filename}")
    
#gets the latest checkpoint file
def get_latest_checkpoint():
    saves_directory = "saves"  # Define the directory where saves are stored
    base_name = "run_"
    gen_base = "_generation"
    ext = ""  # No extension since your checkpoints don't have one
    latest_n = 0
    latest_m = 0
    latest_checkpoint = None

    # Ensure the directory exists to avoid FileNotFoundError
    if not os.path.exists(saves_directory):
        print("No saves directory found.")
        return None, 0

    # Iterate through files in the saves directory
    for filename in os.listdir(saves_directory):
        if filename.startswith(base_name) and gen_base in filename:
            try:
                # Extract n and m from the filename
                n_part = filename[len(base_name):filename.index(gen_base)]
                m_part = filename[filename.index(gen_base) + len(gen_base):]
                n = int(n_part)
                m = int(m_part)

                # Update if this file is more recent
                if n > latest_n or (n == latest_n and m > latest_m):
                    latest_n = n
                    latest_m = m
                    latest_checkpoint = os.path.join(saves_directory, filename)  # Include the path to the file

            except ValueError:
                # Ignore files that don't match the expected pattern
                continue

    print(f"Latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint, latest_n

# Function to get the next best fitnesses file
def get_next_best_fitnesses_file():
    base_directory = "best_fitnesses"
    base_name = "best_fitnesses_run_"
    ext = ".txt"
    n1 = 1
    
    # Ensure the directory exists
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    
    # Generate file name with incremental number
    while os.path.exists(os.path.join(base_directory, f"{base_name}{n1}{ext}")):
        n1 += 1
    return (os.path.join(base_directory, f"{base_name}{n1}{ext}")), n1
    
# Function to reset the robot to the initial state
def reset_robot():
    for motor in motors: # Reset motor positions
        motor.setPosition(0.0)
    # Reset the robot's position and orientation
    translation_field.setSFVec3f(initial_position)
    rotation_field.setSFRotation(initial_rotation)
    for _ in range(3): # Step the simulation a few times to stabilize the reset
        robot.step(TIME_STEP)


# Evaluate fitness of an individual
def evaluate(individual):
    reset_robot() # Reset robot to the initial state before evaluating each individual
    start_time = robot.getTime()
    max_distance, height_sum, height_samples = 0.0, 0.0, 0
    initial_pos = gps.getValues()
    f = 1.0  # Gait frequency
    
    while robot.getTime() - start_time < 20.0:
        
        time = robot.getTime()
        for i, motor in enumerate(motors):
            position = (individual["amplitude"][i] * math.sin(2.0 * math.pi * f * time + individual["phase"][i])
                        + individual["offset"][i])
            motor_name = motor.getName()

            # Apply clamping based on joint-specific limits
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
    #print("average height:", avg_height)
    #print("max distance:", max_distance)
    individual["fitness"] = max_distance + avg_height * HEIGHT_WEIGHT
    return individual["fitness"]


# Mutation
def mutate(individual):
    for i in range(PARAMS):
        if random.random() < MUTATION_RATE:
            individual["amplitude"][i] += random.uniform(-0.05, 0.05)
            individual["phase"][i] += random.uniform(-0.05, 0.05)
            individual["offset"][i] += random.uniform(-0.05, 0.05)


# Crossover between two parents to create a child
def crossover(parent1, parent2):
    child = create_individual()
    for i in range(PARAMS):
        if random.choice([True, False]):
            child["amplitude"][i] = parent1["amplitude"][i]
            child["phase"][i] = parent1["phase"][i]
            child["offset"][i] = parent1["offset"][i]
        else:
            child["amplitude"][i] = parent2["amplitude"][i]
            child["phase"][i] = parent2["phase"][i]
            child["offset"][i] = parent2["offset"][i]
    mutate(child)
    return child


# Roulette wheel selection based on fitness
def select_parent(population):
    total_fitness = sum(ind["fitness"] for ind in population)
    selection_probs = [ind["fitness"] / total_fitness for ind in population] if total_fitness > 0 else None
    return random.choices(population, weights=selection_probs, k=1)[0] if selection_probs else random.choice(population)


# Evolutionary process to create a new generation
def evolve_population(population):
    population.sort(key=lambda ind: ind["fitness"], reverse=True)
    new_population = population[:POPULATION_SIZE // 2]
    while len(new_population) < POPULATION_SIZE:
        parent1, parent2 = select_parent(population), select_parent(population)
        child = crossover(parent1, parent2)
        new_population.append(child)
    return new_population


# Main Evolution Loop
def main():
    best_fitnesses_file, n1 = get_next_best_fitnesses_file()
    gens_per_run = 1  # 1 generation per run to avoid deterioration

    load_checkpoint, n2 = get_latest_checkpoint()
    # load_checkpoint = None # Set to None if starting fresh

    # Initialize state
    if load_checkpoint:
        try:
            populations, best_individuals, best_overall, generation = load_state(load_checkpoint)
            generation += 1
            print(f"Loaded state from {load_checkpoint}: Resuming from generation {generation}")
        except FileNotFoundError:
            print(f"Error: File '{load_checkpoint}' not found.")
            exit(1)
        except Exception as e:
            print(f"Error loading file '{load_checkpoint}': {e}")
            exit(1)
    else:
        populations = [create_individual() for _ in range(POPULATION_SIZE)]
        best_individuals = create_individual()  # Initial random best individual (So robot can walk)
        best_overall = best_individuals
        generation = 0  # First gen

    saves_directory = "saves"
    checkpoint_file = os.path.join(saves_directory, f"run_{n2+1}_generation{generation}")  # Checkpoint organized by run and generation

    # Ensure the saves directory exists
    if not os.path.exists(saves_directory):
        os.makedirs(saves_directory)

    # Define the directory for the data
    data_directory = "data"
    csv_file_name = os.path.join(data_directory, "evolution_data.csv")

    print(f"Progress will be saved to: {checkpoint_file}, data will be saved to: {csv_file_name}")

    # Check if the directory exists, and create it if it does not
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Ensure CSV file has a header if it doesn't exist
    if not os.path.exists(csv_file_name):
        with open(csv_file_name, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write the header row
            csv_writer.writerow(['Run', 'Generation', 'Individual Index', 'Fitness', 'Amplitude', 'Phase', 'Offset'])

    with open(csv_file_name, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        while gens_per_run > 0:
            print(f"Generation {generation}")

            for individual_index in range(POPULATION_SIZE):
                individual = populations[individual_index]

                # Evaluate individuals
                print(f"Evaluating individual {individual_index} in generation {generation}")
                individual["fitness"] = evaluate(individual)
                reset_robot()

                # Log individual fitness/info to file
                csv_writer.writerow([
                    n2 + 1,  # Run number
                    generation,  # Current generation
                    individual_index,  # Individual index
                    individual['fitness'],  # Fitness
                    individual['amplitude'],  # Amplitude list
                    individual['phase'],  # Phase list
                    individual['offset']  # Offset list
                ])

                # Update the best individual
                if individual['fitness'] > best_individuals['fitness']:
                    best_individuals = individual
                    if best_individuals['fitness'] > best_overall['fitness']:
                        best_overall = best_individuals

            # Evolve the population
            populations = evolve_population(populations)

            with open(best_fitnesses_file, "w") as file:
                print(f"\n--- Best Individual ---")
                print(f"Fitness: {best_individuals['fitness']:.3f}")
                print(f"Amplitude: {best_individuals['amplitude']}")
                print(f"Phase: {best_individuals['phase']}")
                print(f"Offset: {best_individuals['offset']}")
                print("------------------------------------------")

                file.write(f"Generation {generation}, Best Fitness: {best_individuals['fitness']:.3f}, "
                           f"Amplitude: {best_individuals['amplitude']}, "
                           f"Phase: {best_individuals['phase']}, "
                           f"Offset: {best_individuals['offset']}\n"
                           f"-------------------------------------\n"
                           f"Generation {generation}, Best Overall: {best_overall['fitness']:.3f}, "
                           f"Amplitude: {best_overall['amplitude']}, "
                           f"Phase: {best_overall['phase']}, "
                           f"Offset: {best_overall['offset']}\n"
                           f"-------------------------------------\n")
                file.flush()

            save_state(checkpoint_file, populations, best_individuals, best_overall, generation)  # Make a checkpoint

            generation += 1
            gens_per_run -= 1
            print("Resetting Webots environment...")
            robot.worldReload()


if __name__ == "__main__":
    main()
