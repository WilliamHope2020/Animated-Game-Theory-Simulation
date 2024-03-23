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

class PlayerSimulation:
    def __init__(self, num_players=4, box_size=3, speed=0.1, interaction_distance=0.3, new_players_probability=0.05, market_crash_probability=0.03, recession_depression_occurrence_probability=0.01):
        self.num_players = num_players
        self.box_size = box_size
        self.speed = speed
        self.interaction_distance = interaction_distance
        self.new_players_probability = new_players_probability
        self.market_crash_probability = market_crash_probability
        self.recession_depression_occurrence_probability = recession_depression_occurrence_probability
        
        # Initialize players attributes
        self.players_sizes = np.random.randint(10, 25, size=num_players)
        self.players_positions = np.random.rand(num_players, 2) * box_size
        self.players_strategies = [random.choice([TitForTat(), RandomPlay(), Pavlov(), FictitiousPlay(), QLearning()]) for _ in range(num_players)]
        self.players_memory = np.zeros((num_players, num_players))
        self.interaction_counter = 0
        self.iteration_counter = 0
        self.event_duration = 0
        self.rare_event_counter = 0
        self.players_colors = np.random.rand(num_players, 3)

        # Initialize variables for tracking player numbers and defeated players
        self.next_players_number = num_players + 1
        self.defeated_players_indices = set()

        # Initialize the plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, box_size)
        self.ax.set_ylim(0, box_size)
        self.players = self.ax.scatter(self.players_positions[:, 0], self.players_positions[:, 1], c=self.players_colors)
        
        # Initialize legend
        self.legend_elements = []
        self.update_legend()
        self.legend = self.ax.legend(handles=self.legend_elements, loc='upper left', title='Player Legend')
        
        # Initialize animation
        self.ani = FuncAnimation(self.fig, self.update, frames=range(100), blit=True, interval=50)

    def rare_event(self):
            # Check for stock market occurence 
            if np.random.rand() < self.market_crash_probability and self.event_duration == 0:
                self.rare_event_counter +=1
                logging.info("A stock market crash has occurred!")  # Stock market crash occurrence
                for i in range(len(self.players_sizes)):
                    percentage = np.random.uniform(0.25, 0.5)
                    self.players_sizes[i] -= self.players_sizes[i] * percentage
                self.event_duration = 1
                    
            # Check for recession or depression occurrence
            if np.random.rand() < self.recession_depression_occurrence_probability and self.event_duration == 0:
                event_type = np.random.choice(["recession", "depression"])  # Randomly choose between recession and depression
                if event_type == "recession":
                    self.rare_event_counter +=1
                    logging.info("A recession has occurred!")  # Recession occurrence
                    for _ in range(2):
                        for i in range(len(self.players_sizes)):
                            percentage = np.random.uniform(0.05, 0.15)
                            self.players_sizes[i] -= self.players_sizes[i] * percentage
                    logging.info("The recession has ended!")  # Recession end occurrence
                else:
                    self.rare_event_counter +=1
                    logging.info("A depression has occurred!")  # Depression occurrence
                    for _ in range(6):
                        for i in range(len(self.players_sizes)):
                            percentage = np.random.uniform(0.15, 0.3)
                            self.players_sizes[i] -= self.players_sizes[i] * percentage
                    logging.info("The depression has ended!")  # Depression end occurrence

    def remove_defeated_players(self):
        for idx in list(self.defeated_players_indices):
            players_label = f'Player {idx + 1}: Size {self.players_sizes[idx]}' if self.players_sizes[idx] > 0 else 'Defeated'
            players_index = next((i for i, handle in enumerate(self.legend_elements) if handle.get_label() == players_label), None)
            if players_index is not None:
                self.legend_elements.pop(players_index)
                logging.info(f"Player {idx+1} remains defeated.")
                self.defeated_players_indices.remove(idx)  # Remove players from defeated indices set
            else:
                logging.warning(f"Player {idx+1} not found in legend elements.")
                # Remove the players from defeated indices if it's not found in legend elements
                self.defeated_players_indices.remove(idx)

    def update_legend(self):
        legend_elements = []
        for i in range(self.num_players):
            size = self.players_sizes[i]
            if size > 0:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Player {i+1}: Size {size}', markerfacecolor=self.players_colors[i]))
            else:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Player {i+1}: Defeated', markerfacecolor='black', markersize=5))
        
        # Add legend elements for interactions, iterations, and total value
        legend_elements.append(Line2D([0], [0], color='w', lw=1, label=f'Interactions: {self.interaction_counter}'))
        legend_elements.append(Line2D([0], [0], color='w', lw=1, label=f'Iterations: {self.iteration_counter}'))
        legend_elements.append(Line2D([0], [0], color='w', lw=1, label=f'Total Value: {sum(self.players_sizes)}'))
        legend_elements.append(Line2D([0], [0], color='w', lw=1, label=f'Rare Events: {self.rare_event_counter}'))

        self.legend_elements = legend_elements
        self.legend = self.ax.legend(handles=legend_elements, loc='upper left', title='Player Legend')

    def update(self, frame):
        self.iteration_counter += 1
        
        self.rare_event()

        # Check if a new player spawns
        if np.random.rand() < self.new_players_probability:
            if self.num_players < 10:
                self.num_players += 1
                new_players_size = np.random.randint(10, 25)
                self.players_sizes = np.append(self.players_sizes, new_players_size)
                self.players_colors = np.vstack([self.players_colors, np.random.rand(1, 3)])
                self.players_strategies.append(random.choice([TitForTat(), RandomPlay(), Pavlov(), FictitiousPlay(), QLearning()]))
                self.players_memory = np.pad(self.players_memory, ((0, 1), (0, 1)), mode='constant')  # Expand player memory
                new_players_position = np.random.rand(1, 2) * self.box_size
                self.players_positions = np.vstack([self.players_positions, new_players_position])
            logging.info(f"Player {self.next_players_number} has been added.")
            self.next_players_number += 1

        self.remove_defeated_players()

        # Check if a player has been defeated and add a new player in the next iteration
        if self.defeated_players_indices:
            self.num_players += 1
            new_players_size = np.random.randint(10, 25)
            self.players_sizes = np.append(self.players_sizes, new_players_size)
            self.players_colors = np.vstack([self.players_colors, np.random.rand(1, 3)])
            self.players_strategies.append(random.choice([TitForTat(), RandomPlay(), Pavlov(), FictitiousPlay(), QLearning()]))
            self.players_memory = np.pad(self.players_memory, ((0, 1), (0, 1)), mode='constant')  # Expand player memory
            new_players_position = np.random.rand(1, 2) * self.box_size
            self.players_positions = np.vstack([self.players_positions, new_players_position])
            logging.info(f"A new player has been added after another player was defeated.")
            
            # Clear defeated player indices after adding a new player
            self.defeated_players_indices = set()

        for i in range(self.num_players):
            self.players_positions[i] += np.random.uniform(-self.speed, self.speed, size=2)
            self.players_positions[i] = np.clip(self.players_positions[i], 0, self.box_size)

        for i in range(self.num_players):
            for j in range(i + 1, self.num_players):
                distance = np.linalg.norm(self.players_positions[i] - self.players_positions[j])
                if distance < self.interaction_distance and self.players_sizes[i] > 0 and self.players_sizes[j] > 0:
                    self.interaction_counter += 1
                    cooperate_i = self.players_strategies[i].play(self.players_memory[i, j])
                    cooperate_j = self.players_strategies[j].play(self.players_memory[j, i])

                    self.players_memory[i, j] = cooperate_i
                    self.players_memory[j, i] = cooperate_j
                    
                    if cooperate_i and cooperate_j:
                        self.players_sizes[i] += 3
                        self.players_sizes[j] += 3
                        logger.info(f"Player {i+1} and Player {j+1} cooperated!")
                    elif cooperate_i and not cooperate_j:
                        self.players_sizes[i] -= 2
                        self.players_sizes[j] += 5
                        logger.info(f"Player {i+1} cooperated but Player {j+1} defected.")
                    elif not cooperate_i and cooperate_j:
                        self.players_sizes[i] += 5
                        self.players_sizes[j] -= 2
                        logger.info(f"Player {i+1} defected when Player {j+1} cooperated.")
                    else:
                        self.players_sizes[i] -= 5
                        self.players_sizes[j] -= 5
                        logger.info(f"Player {i+1} and Player {j+1} defected!")

                    self.players_sizes = np.maximum(self.players_sizes, 0)
                    
                    if self.players_sizes[i] >= 0.25 * sum(self.players_sizes):
                        self.players_sizes[j] //= 2
                        redistributed_value = self.players_sizes[j] // (self.num_players - 1)
                        self.players_sizes += redistributed_value
                        logger.info(f"Antitrust scrutiny policy applied. Player {j+1} got its size redistributed.")

                    if self.players_sizes[i] >= 2 * self.players_sizes[j]:
                        logger.info(f"Player {i+1} consumed Player {j+1} as it was twice as big.")
                        self.players_sizes[j] = 0

                        if np.random.rand() < 0.5 and self.players_sizes[j] <= 0:
                            recovered_amount = np.random.randint(10, 25)  # Random amount of recovery
                            self.players_sizes[j] = recovered_amount
                            logging.info(f"The Government has bailed Player {j+1} with size {recovered_amount}.") # Skip spawning a new Player if this Player recovers
                        else:
                            self.skip_new_players_spawn = True
                            self.defeated_players_indices.add(j)

                    elif self.players_sizes[j] >= 2 * self.players_sizes[i]:
                        logger.info(f"Player {j+1} consumed Player {i+1} as it was twice as big.")
                        self.players_sizes[i] = 0
                        self.defeated_players_indices.add(i)

                        if np.random.rand() < 0.5 and self.players_sizes[i] <= 0:
                            recovered_amount = np.random.randint(10, 25)  # Random amount of recovery
                            self.players_sizes[i] = recovered_amount
                            logging.info(f"The Government has bailed Player {i+1} with size {recovered_amount}.") # Skip spawning a new Player if this Player recovers
                        else:
                            self.skip_new_players_spawn = True

    
        self.players.set_offsets(self.players_positions)
        self.players.set_sizes(self.players_sizes)

        self.update_legend()
        self.legend = self.ax.legend(handles=self.legend_elements, loc='center right', title='Player Legend', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)

        return self.players, self.legend

class YourClass:
    def __init__(self):
        self.num_players = 0
        self.strategy_names = {
            0: "TitForTat",
            1: "RandomPlay",
            2: "Pavlov",
            3: "FictitiousPlay",
            4: "QLearning"
        }
        self.players_strategies = {}
    
    def assign_strategies(self):
        available_strategies = list(self.strategy_names.keys())
        self.players_strategies = {f"Player {i+1}": random.choice(available_strategies) for i in range(self.num_players)}
    
    def gather_strategies(self):
        return {player: self.strategy_names.get(strategy, 'Unknown Strategy') for player, strategy in self.players_strategies.items()}
    
    def display_strategies(self):
        strategies = self.gather_strategies()
        print(strategies)

# Instantiate YourClass
your_instance = YourClass()

# Set the number of players
your_instance.num_players = 20

# Assign strategies to players
your_instance.assign_strategies()

# Display the assigned strategies
your_instance.display_strategies()

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

# Create PlayerSimulation instance
player_simulation = PlayerSimulation()

# Show the plot
plt.show()
