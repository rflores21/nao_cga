import matplotlib.pyplot as plt  # Importing the matplotlib library for plotting

# Time step for numerical integration
dt = 0.0001

# # Initial conditions for the variables x and y
# x = 1  # Initial value of x
# y = 0  # Initial value of y

# # Lists to store the values of x and y over time
# activate = []  # List to store x values
# deactivate = []  # List to store y values

# # Iterating over a large number of steps to simulate the system
# for ii in range(1000000):
#     # Update x and y using simple coupled differential equations
#     x = x + -y * dt  # Update x based on y
#     y = y + x * dt   # Update y based on x

#     # Append the updated values of x and y to their respective lists
#     activate.append(x)
#     deactivate.append(y)

# # Plot the trajectory of the system in the x-y plane
# plt.plot(activate, deactivate)  # Plot x (activate) vs y (deactivate)
# plt.show()  # Display the plot

# Parameters for the Matsuoka CPG
beta = 2.5  # Mutual inhibition strength
u1, u2 = 0.5, 0.5  # Initial neural activations
v1, v2 = 0.0, 0.0  # Initial outputs of the neurons
w12, w21 = 2.0, 2.0  # Weights of inhibition between neurons

# Lists to store the outputs of the neurons over time
neuron1_output = []
neuron2_output = []

# Simulate the Matsuoka CPG for a fixed number of steps
for t in range(1000000): # Number of time steps
    # Update the activations of the neurons
    du1 = (-u1 - beta * v2 + w12 * v1) * dt
    du2 = (-u2 - beta * v1 + w21 * v2) * dt
    u1 += du1
    u2 += du2

    # Update the outputs of the neurons
    v1 = max(0, u1)  # Rectified output of neuron 1
    v2 = max(0, u2)  # Rectified output of neuron 2

    # Store the outputs for plotting
    neuron1_output.append(v1)
    neuron2_output.append(v2)

# Plot the outputs of the Matsuoka CPG in a single figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 10))  # Create a figure with 2 rows and 1 column of subplots

# Plot the output of Neuron 1 in the first subplot
axs[0].plot(neuron1_output, label="Neuron 1 Output", color="blue")
axs[0].legend()
axs[0].set_title("Neuron 1 Output")
axs[0].set_xlabel("Time Steps")
axs[0].set_ylabel("Neuron Output")

# Plot the output of Neuron 2 in the second subplot
axs[1].plot(neuron2_output, label="Neuron 2 Output", color="red")
axs[1].legend()
axs[1].set_title("Neuron 2 Output")
axs[1].set_xlabel("Time Steps")
axs[1].set_ylabel("Neuron Output")

# Adjust layout to prevent overlap and display the plots
plt.tight_layout()
plt.show()



