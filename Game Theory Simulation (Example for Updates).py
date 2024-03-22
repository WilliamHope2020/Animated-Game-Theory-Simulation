import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define learning algorithms as classes
class TitForTat:
    def __init__(self):
        self.cooperate = True

    def play(self, opponent_cooperate):
        return self.cooperate

    def update(self, opponent_cooperate):
        self.cooperate = opponent_cooperate

class RandomPlay:
    def __init__(self):
        pass

    def play(self, opponent_cooperate=None):  
        return random.choice([True, False])

    def update(self, opponent_cooperate):
        pass

class Pavlov:
    def __init__(self):
        self.cooperate = True

    def play(self, opponent_cooperate=None):
        return self.cooperate

    def update(self, opponent_cooperate):
        if opponent_cooperate:
            self.cooperate = not self.cooperate

class FictitiousPlay:
    def __init__(self):
        self.cooperate_count = 0
        self.defect_count = 0

    def play(self, opponent_cooperate=None):
        return self.cooperate_count > self.defect_count

    def update(self, opponent_cooperate):
        if opponent_cooperate:
            self.cooperate_count += 1
        else:
            self.defect_count += 1

class QLearning:
    def __init__(self, learning_rate=0.1, discount_factor=0.9):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}

    def play(self, state):
        if state not in self.q_table:
            self.q_table[state] = 0.5  # Initial value for unknown states
        return random.random() < self.q_table[state]

    def update(self, state, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = 0.5  # Initial value for unknown states
        if next_state not in self.q_table:
            self.q_table[next_state] = 0.5  # Initial value for unknown states
        max_next_q = max(self.q_table[next_state], 0.5)  # Assume unknown states have value of 0.5
        self.q_table[state] += self.learning_rate * (reward + self.discount_factor * max_next_q - self.q_table[state])

class DotSimulation:
    def __init__(self, num_dots=4, box_size=3, speed=0.1, interaction_distance=0.3, new_dot_probability=0.05):
        self.num_dots = num_dots
        self.box_size = box_size
        self.speed = speed
        self.interaction_distance = interaction_distance
        self.new_dot_probability = new_dot_probability
        
        # Initialize dot attributes
        self.dot_sizes = np.random.randint(10, 25, size=num_dots)
        self.dot_positions = np.random.rand(num_dots, 2) * box_size
        self.dot_strategies = [random.choice([TitForTat(), RandomPlay(), Pavlov(), FictitiousPlay(), QLearning()]) for _ in range(num_dots)]
        self.dot_memory = np.zeros((num_dots, num_dots))
        self.interaction_counter = 0
        self.iteration_counter = 0
        self.dot_colors = np.random.rand(num_dots, 3)

        # Initialize variables for tracking dot numbers and defeated dots
        self.next_dot_number = num_dots + 1
        self.defeated_dot_indices = set()

        # Initialize the plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, box_size)
        self.ax.set_ylim(0, box_size)
        self.dots = self.ax.scatter(self.dot_positions[:, 0], self.dot_positions[:, 1], c=self.dot_colors)
        
        # Initialize legend
        self.legend_elements = []
        self.update_legend()
        self.legend = self.ax.legend(handles=self.legend_elements, loc='upper left', title='Dot Legend')
        
        # Initialize animation
        self.ani = FuncAnimation(self.fig, self.update, frames=range(100), blit=True, interval=50)

    def remove_defeated_dots(self):
        for idx in list(self.defeated_dot_indices):  # Use list() to avoid changing set size during iteration
            dot_label = f'Dot {idx + 1}: Size {self.dot_sizes[idx]}' if self.dot_sizes[idx] > 0 else 'Defeated'
            dot_index = next((i for i, handle in enumerate(self.legend_elements) if handle.get_label() == dot_label), None)
            if dot_index is not None:
                self.legend_elements.pop(dot_index)
                logging.info(f"Dot {idx+1} remains defeated.")
                self.defeated_dot_indices.remove(idx)  # Remove dot from defeated indices set
            else:
                logging.warning(f"Dot {idx+1} not found in legend elements.")
                # Remove the dot from defeated indices if it's not found in legend elements
                self.defeated_dot_indices.remove(idx)

    def update_legend(self):
        legend_elements = []
        for i in range(self.num_dots):
            size = self.dot_sizes[i]
            if size > 0:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Dot {i+1}: Size {size}', markerfacecolor=self.dot_colors[i]))
            else:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Dot {i+1}: Defeated', markerfacecolor='black', markersize=5))
        
        # Add legend elements for interactions, iterations, and total value
        legend_elements.append(Line2D([0], [0], color='w', lw=1, label=f'Interactions: {self.interaction_counter}'))
        legend_elements.append(Line2D([0], [0], color='w', lw=1, label=f'Iterations: {self.iteration_counter}'))
        legend_elements.append(Line2D([0], [0], color='w', lw=1, label=f'Total Value: {sum(self.dot_sizes)}'))

        self.legend_elements = legend_elements
        self.legend = self.ax.legend(handles=legend_elements, loc='upper left', title='Dot Legend')

    def close_event(self, event):
        plt.close(self.fig)
        strategies = self.gather_strategies()
        print("Dot Strategies:")
        for dot, strategy in strategies.items():
            print(f"{dot}: {strategy}")

    def update(self, frame):
        self.iteration_counter += 1
        
        # Check if a new dot spawns
        if np.random.rand() < self.new_dot_probability:
            if self.num_dots < 10:
                self.num_dots += 1
                new_dot_size = np.random.randint(10, 25)
                self.dot_sizes = np.append(self.dot_sizes, new_dot_size)
                self.dot_colors = np.vstack([self.dot_colors, np.random.rand(1, 3)])
                self.dot_strategies.append(random.choice([TitForTat(), RandomPlay(), Pavlov(), FictitiousPlay(), QLearning()]))
                self.dot_memory = np.pad(self.dot_memory, ((0, 1), (0, 1)), mode='constant')  # Expand dot_memory
                new_dot_position = np.random.rand(1, 2) * self.box_size
                self.dot_positions = np.vstack([self.dot_positions, new_dot_position])
            logging.info(f"Dot {self.next_dot_number} has been added.")
            self.next_dot_number += 1

        self.remove_defeated_dots()

        # Check if a dot has been defeated and add a new dot in the next iteration
        if self.defeated_dot_indices:
            self.num_dots += 1
            new_dot_size = np.random.randint(10, 25)
            self.dot_sizes = np.append(self.dot_sizes, new_dot_size)
            self.dot_colors = np.vstack([self.dot_colors, np.random.rand(1, 3)])
            self.dot_strategies.append(random.choice([TitForTat(), RandomPlay(), Pavlov(), FictitiousPlay(), QLearning()]))
            self.dot_memory = np.pad(self.dot_memory, ((0, 1), (0, 1)), mode='constant')  # Expand dot_memory
            new_dot_position = np.random.rand(1, 2) * self.box_size
            self.dot_positions = np.vstack([self.dot_positions, new_dot_position])
            logging.info(f"A new dot has been added after another dot was defeated.")
            
            # Clear defeated dot indices after adding a new dot
            self.defeated_dot_indices = set()

        for i in range(self.num_dots):
            self.dot_positions[i] += np.random.uniform(-self.speed, self.speed, size=2)
            self.dot_positions[i] = np.clip(self.dot_positions[i], 0, self.box_size)

        for i in range(self.num_dots):
            for j in range(i + 1, self.num_dots):
                distance = np.linalg.norm(self.dot_positions[i] - self.dot_positions[j])
                if distance < self.interaction_distance and self.dot_sizes[i] > 0 and self.dot_sizes[j] > 0:
                    self.interaction_counter += 1
                    cooperate_i = self.dot_strategies[i].play(self.dot_memory[i, j])
                    cooperate_j = self.dot_strategies[j].play(self.dot_memory[j, i])

                    self.dot_memory[i, j] = cooperate_i
                    self.dot_memory[j, i] = cooperate_j
                    
                    if cooperate_i and cooperate_j:
                        self.dot_sizes[i] += 3
                        self.dot_sizes[j] += 3
                        logger.info(f"Dot {i+1} and Dot {j+1} cooperated!")
                    elif cooperate_i and not cooperate_j:
                        self.dot_sizes[i] -= 2
                        self.dot_sizes[j] += 5
                        logger.info(f"Dot {i+1} cooperated but Dot {j+1} defected.")
                    elif not cooperate_i and cooperate_j:
                        self.dot_sizes[i] += 5
                        self.dot_sizes[j] -= 2
                        logger.info(f"Dot {i+1} defected when Dot {j+1} cooperated.")
                    else:
                        self.dot_sizes[i] -= 5
                        self.dot_sizes[j] -= 5
                        logger.info(f"Dot {i+1} and Dot {j+1} defected!")

                    self.dot_sizes = np.maximum(self.dot_sizes, 0)
                    
                    if self.dot_sizes[i] >= 0.25 * sum(self.dot_sizes):
                        self.dot_sizes[j] //= 2
                        redistributed_value = self.dot_sizes[j] // (self.num_dots - 1)
                        self.dot_sizes += redistributed_value
                        logger.info(f"Antitrust scrutiny policy applied. Dot {j+1} got its size redistributed.")

                    if self.dot_sizes[i] >= 2 * self.dot_sizes[j]:
                        logger.info(f"Dot {i+1} consumed Dot {j+1} as it was twice as big.")
                        self.dot_sizes[j] = 0

                        if np.random.rand() < 0.5 and self.dot_sizes[j] <= 0:
                            recovered_amount = np.random.randint(10, 25)  # Random amount of recovery
                            self.dot_sizes[j] = recovered_amount
                            logging.info(f"The Government has bailed Dot {j+1} with size {recovered_amount}.") # Skip spawning a new dot if this dot recovers
                        else:
                            self.skip_new_dot_spawn = True
                            self.defeated_dot_indices.add(j)

                    elif self.dot_sizes[j] >= 2 * self.dot_sizes[i]:
                        logger.info(f"Dot {j+1} consumed Dot {i+1} as it was twice as big.")
                        self.dot_sizes[i] = 0
                        self.defeated_dot_indices.add(i)

                        if np.random.rand() < 0.5 and self.dot_sizes[i] <= 0:
                            recovered_amount = np.random.randint(10, 25)  # Random amount of recovery
                            self.dot_sizes[i] = recovered_amount
                            logging.info(f"The Government has bailed Dot {i+1} with size {recovered_amount}.") # Skip spawning a new dot if this dot recovers
                        else:
                            self.skip_new_dot_spawn = True

        self.dots.set_offsets(self.dot_positions)
        self.dots.set_sizes(self.dot_sizes)

        self.update_legend()
        self.legend = self.ax.legend(handles=self.legend_elements, loc='upper left', title='Dot Legend')

        return self.dots, self.legend

class YourClass:
    def __init__(self):
        self.num_dots = 0
        self.strategy_names = {
            0: "TitForTat",
            1: "RandomPlay",
            2: "Pavlov",
            3: "FictitiousPlay",
            4: "QLearning"
        }
        self.dot_strategies = {}
    
    def assign_strategies(self):
        available_strategies = list(self.strategy_names.keys())
        self.dot_strategies = {f"Dot {i+1}": random.choice(available_strategies) for i in range(self.num_dots)}
    
    def gather_strategies(self):
        return {dot: self.strategy_names.get(strategy, 'Unknown Strategy') for dot, strategy in self.dot_strategies.items()}
    
    def display_strategies(self):
        strategies = self.gather_strategies()
        print(strategies)
        print("Dot Strategies:")
        for dot, strategy in strategies.items():
            print(f"{dot}: {strategy}")

# Instantiate YourClass
your_instance = YourClass()

# Set the number of dots
your_instance.num_dots = 1000

# Assign strategies to dots
your_instance.assign_strategies()

# Display the assigned strategies
your_instance.display_strategies()

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

# Create DotSimulation instance
dot_simulation = DotSimulation()

# Show the plot
plt.show()
