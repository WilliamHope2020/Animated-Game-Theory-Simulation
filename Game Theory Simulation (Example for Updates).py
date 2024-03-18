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
interaction_counter = 0 # interactions (based on number of times dots interact/play games)
iteration_counter = 0 # iterations (based on number of loops executed)
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
legend_elements.append(Line2D([0], [0], color='w', lw=1, label=f'Interactions: {interaction_counter}')) # add legend element for interaction counter
legend_elements.append(Line2D([0], [0], color='w', lw=1, label=f'Iterations: {iteration_counter}')) # add legend element for iteration counter
legend_elements.append(Line2D([0], [0], color='w', lw=1, label=f'Total Value: {sum(dot_sizes)}'))  # Add legend element for total value

# Initialize legend
legend = ax.legend(handles=legend_elements, loc='upper left', title='Dot Legend')

# Function to remove defeated dots from the legend
def remove_defeated_dots():
    global legend_elements
    defeated_indices = np.where(dot_sizes == 0)[0]
    if len(defeated_indices) > 0:
        for idx in defeated_indices:
            dot_label = f'Dot {idx + 1}: Size {dot_sizes[idx]}' if dot_sizes[idx] > 0 else 'Defeated'
            dot_index = next((i for i, handle in enumerate(legend_elements) if handle.get_label() == dot_label), None)
            if dot_index is not None:
                legend_elements.pop(dot_index)

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
        new_dot_number = num_dots
        new_dot_label = f'Dot {new_dot_number}: Size {dot_sizes[-1]}'
        new_dot_handle = Line2D([0], [0], marker='o', color='w', label=new_dot_label, markerfacecolor=dot_colors[-1], markersize=5)
        # Find indices of interaction and iteration counters in legend elements
        interaction_index = [i for i, handle in enumerate(legend_elements) if handle.get_label().startswith('Interactions')][0]
        # Insert new dot legend just above interaction and iteration counters
        legend_elements.insert(interaction_index, new_dot_handle)

    remove_defeated_dots()  # Remove defeated dots from legend

    for i in range(num_dots):
        # Move each dot randomly
        dot_positions[i] += np.random.uniform(-speed, speed, size=2)
        
        # Keep dots inside the box
        dot_positions[i] = np.clip(dot_positions[i], 0, box_size)
    
    i = 0  # Initialize loop index
    while i < num_dots:
        j = i + 1
        while j < num_dots:
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
                    print(f"Dot {i+1} and Dot {j+1} cooperated!")
                elif cooperate_i and not cooperate_j:
                    dot_sizes[i] -= 2
                    dot_sizes[j] += 5
                    print(f"Dot {i+1} cooperated but Dot {j+1} defected")
                elif not cooperate_i and cooperate_j:
                    dot_sizes[i] += 5
                    dot_sizes[j] -= 2
                    print(f"Dot {i+1} defected when Dot {j+1} cooperated")
                else:  # Both defect
                    dot_sizes[i] -= 5
                    dot_sizes[j] -= 5
                    print(f"Dot {i+1} and Dot {j+1} defected!")
                        
                # Ensure dot sizes don't go below 0
                dot_sizes = np.maximum(dot_sizes, 0)

                # Check for high market share, implements Antitrust Scrutiny policy if true
                if dot_sizes[i] >= 0.25 * sum(dot_sizes):
                    # Reduce the dot's size by half
                    dot_sizes[i] //= 2
                    # Redistribute the reduced value to all other dots equally
                    redistributed_value = dot_sizes[i] // (num_dots - 1)
                    dot_sizes += redistributed_value
                    print(f"Dot {i+1} got its size redistributed!")

                if dot_sizes[j] >= 0.25 * sum(dot_sizes):
                    # Reduce the dot's size by half
                    dot_sizes[j] //= 2
                    # Redistribute the reduced value to all other dots equally
                    redistributed_value = dot_sizes[j] // (num_dots - 1)
                    dot_sizes += redistributed_value
                    print(f"Dot {j+1} got its size redistributed!")

                if dot_sizes[i] >= 2 * dot_sizes[j]:
                    # Dot i's size is at least twice as big as dot j's size
                    dot_sizes[i] += dot_sizes[j]
                    print(f"Dot {i+1} gained {dot_sizes[j]} from Dot {j+1}!")
                    dot_sizes[j] = 0  # Set dot j's size to 0 after adding its value to dot i
                elif dot_sizes[j] >= 2 * dot_sizes[i]:
                    # Dot j's size is at least twice as big as dot i's size
                    dot_sizes[j] += dot_sizes[i]
                    print(f"Dot {j+1} gained {dot_sizes[i]} from Dot {i+1}!")
                    dot_sizes[i] = 0  # Set dot i's size to 0 after adding its value to dot j
                else:
                    # Neither dot's size is twice as big as the other's size
                    pass  # No action needed
                    
                # Check if a dot's size reaches 0
                if dot_sizes[i] == 0:
                    # Remove dot from legend if it reaches 0 size
                    legend_elements.pop(i)
                    # Update dot positions, sizes, strategies, and memory to remove dot
                    dot_positions = np.delete(dot_positions, i, axis=0)
                    dot_sizes = np.delete(dot_sizes, i)
                    dot_strategies = np.delete(dot_strategies, i)
                    dot_memory = np.delete(dot_memory, i, axis=0)
                    dot_memory = np.delete(dot_memory, i, axis=1)
                    num_dots -= 1  # Update the number of dots
                    print(f"Dot {i+1} has disappeared!")
                    continue  # Skip to the next iteration since dot i has been removed
                    
                if dot_sizes[j] == 0:
                    # Remove dot from legend if it reaches 0 size
                    legend_elements.pop(j)
                    # Update dot positions, sizes, strategies, and memory to remove dot
                    dot_positions = np.delete(dot_positions, j, axis=0)
                    dot_sizes = np.delete(dot_sizes, j)
                    dot_strategies = np.delete(dot_strategies, j)
                    dot_memory = np.delete(dot_memory, j, axis=0)
                    dot_memory = np.delete(dot_memory, j, axis=1)
                    num_dots -= 1  # Update the number of dots
                    print(f"Dot {j+1} has disappeared!")
                    continue  # Skip to the next iteration since dot j has been removed
                    
            j += 1
        i += 1
                
    # Update dot positions and sizes on the plot
    dots.set_offsets(dot_positions)
    dots.set_sizes(dot_sizes)
    
    # Update legend labels for dots
    for handle, size, idx in zip(legend_elements[:num_dots], dot_sizes, range(1, num_dots + 1)):
        if size > 0:
            handle.set_label(f'Dot {idx}: Size {size}')
        else:
            handle.set_label('Defeated')
        
    # Update legend elements for interactions
    legend_elements[-3].set_label(f'Interactions: {interaction_counter}')
    legend_elements[-2].set_label(f'Iterations: {iteration_counter}')
    legend_elements[-1].set_label(f'Total Value: {sum(dot_sizes)}')

    # Update legend
    legend = ax.legend(handles=legend_elements, loc='upper left', title='Dot Legend')

    return dots, legend

# Create animation
ani = FuncAnimation(fig, update, frames=range(100), blit=True, interval=50)

# Show the plot
plt.show()
