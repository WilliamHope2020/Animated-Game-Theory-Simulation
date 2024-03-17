import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

# Define parameters
num_dots = 4
box_size = 3
speed = 0.1
interaction_distance = 0.2
dot_sizes = np.random.randint(10, 25, size=num_dots)  # Random initial dot sizes
dot_colors = np.random.rand(num_dots, 3)  # Random colors for each dot
dot_strategies = np.random.choice([0, 1], size=num_dots)  # 0 for cooperate, 1 for defect
dot_memory = np.zeros((num_dots, num_dots))  # Memory of past interactions
interaction_counter = 0
iteration_counter = 0
new_dot_probability = 0.05  # Probability of a new dot spawning

# Initialize dot positions
dot_positions = np.random.rand(num_dots, 2) * box_size

# Initialize the plot
fig, ax = plt.subplots()
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
dots = ax.scatter(dot_positions[:, 0], dot_positions[:, 1], c=dot_colors)

# Initialize legend elements and labels
legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Dot {i+1}: Size {dot_sizes[i]}' if dot_sizes[i] > 0 else 'Defeated', markerfacecolor=dot_colors[i], markersize=np.sqrt(dot_sizes[i])) for i in range(num_dots)]
legend_elements.append(Line2D([0], [0], color='w', lw=1, label=f'Interactions: {interaction_counter}'))
legend_elements.append(Line2D([0], [0], color='w', lw=1, label=f'Iterations: {iteration_counter}'))

# Initialize legend
legend = ax.legend(handles=legend_elements, loc='upper left', title='Dot Legend')

# Function to update dot positions and play the game
def update(frame):
    global dot_sizes, dot_positions, dot_strategies, dot_memory, legend_elements, interaction_counter, iteration_counter, num_dots, dot_colors
    
    iteration_counter += 1
    
    # Check if a new dot spawns (up to 10 dots)
    if num_dots < 10 and np.random.rand() < new_dot_probability:
        num_dots += 1
        dot_sizes = np.append(dot_sizes, np.random.randint(10, 25))
        dot_colors = np.vstack([dot_colors, np.random.rand(1, 3)])
        dot_strategies = np.append(dot_strategies, np.random.choice([0, 1]))
        dot_memory = np.pad(dot_memory, ((0, 1), (0, 1)), mode='constant')  # Expand dot_memory
        dot_positions = np.vstack([dot_positions, np.random.rand(1, 2) * box_size])
        new_dot_label = f'Dot {num_dots}: Size {dot_sizes[-1]}'
        new_dot_handle = Line2D([0], [0], marker='o', color='w', label=new_dot_label, markerfacecolor=dot_colors[-1], markersize=5)
        legend_elements.insert(-2, new_dot_handle)  # Insert new dot legend before interactions counter
    
    for i in range(num_dots):
        # Move each dot randomly
        dot_positions[i] += np.random.uniform(-speed, speed, size=2)
        
        # Keep dots inside the box
        dot_positions[i] = np.clip(dot_positions[i], 0, box_size)
    
    # Check for interactions between dots
    for i in range(num_dots):
        for j in range(i+1, num_dots):
            distance = np.linalg.norm(dot_positions[i] - dot_positions[j])
            if distance < interaction_distance and dot_sizes[i] > 0 and dot_sizes[j] > 0:
                interaction_counter += 1
                
                # Play the game if both dots have sizes greater than 0
                cooperate_i = (dot_strategies[i] == 0)
                cooperate_j = (dot_strategies[j] == 0)
                
                # Update dot memory
                dot_memory[i, j] += 1
                dot_memory[j, i] += 1
                
                # Adjust strategy based on past interactions
                if dot_memory[i, j] > dot_memory[j, i]:
                    dot_strategies[i] = dot_strategies[j]
                elif dot_memory[j, i] > dot_memory[i, j]:
                    dot_strategies[j] = dot_strategies[i]
                
                if cooperate_i and cooperate_j:
                    dot_sizes[i] += 3
                    dot_sizes[j] += 3
                elif cooperate_i and not cooperate_j:
                    dot_sizes[i] -= 2
                    dot_sizes[j] += 5
                elif not cooperate_i and cooperate_j:
                    dot_sizes[i] += 5
                    dot_sizes[j] -= 2
                else:  # Both defect
                    dot_sizes[i] -= 5
                    dot_sizes[j] -= 5
                    
                # Ensure dot sizes don't go below 0
                dot_sizes = np.maximum(dot_sizes, 0)
                
                # Check if a dot's size reaches 0
                if dot_sizes[i] == 0:
                    print(f"Dot {i+1} has disappeared!")
                if dot_sizes[j] == 0:
                    print(f"Dot {j+1} has disappeared!")
                
    # Update dot positions and sizes on the plot
    dots.set_offsets(dot_positions)
    dots.set_sizes(dot_sizes)
    
    # Update legend labels for dots
    for handle, size, idx in zip(legend_elements[:num_dots], dot_sizes, range(1, num_dots + 1)):
        if size > 0:
            handle.set_label(f'Dot {idx}: Size {size}')
        else:
            handle.set_label('Defeated')
        handle.set_markersize(np.sqrt(size)) if size > 0 else handle.set_markersize(0)

    # Update legend elements for interactions
    legend_elements[-2].set_label(f'Interactions: {interaction_counter}')
    legend_elements[-1].set_label(f'Iterations: {iteration_counter}')
    
    # Update legend
    legend = ax.legend(handles=legend_elements, loc='upper left', title='Dot Legend')

    return dots, legend

# Create animation
ani = FuncAnimation(fig, update, frames=range(100), blit=True, interval=50)

# Show the plot
plt.show()
