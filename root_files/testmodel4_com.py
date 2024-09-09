'''
    Import packages for other libraries
    
'''
import numpy as np
import os
import csv
import random
from collections import deque, defaultdict, Counter

'''
    Tensorflow imports
    - For model creation, loading and tensor conversion
    
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

'''
    Catanatron model imports for specific functions, classes and enums 
    - Imports are from majority of the files in models due to their importance rooted in the game engine.
        
'''
from catanatron.models.player import Player, Color
from catanatron.models.enums import Action, ActionType
from catanatron.models.decks import CITY_COST_FREQDECK, SETTLEMENT_COST_FREQDECK, ROAD_COST_FREQDECK, DEVELOPMENT_CARD_COST_FREQDECK, freqdeck_contains
from catanatron.models.enums import FastResource
from catanatron.models.map import get_node_counter_production

'''
    Catanatron state_function import for specific functions, and player key
    - Many of these functions are utilised for checking resources for other actions and decision making
    
'''
from catanatron.state_functions import player_key, get_player_freqdeck, get_player_buildings, get_actual_victory_points, get_dev_cards_in_hand, player_has_rolled, player_num_resource_cards

'''
    Catanatron gym import for specific functions, and action space
    - Utilising the provided functions we can convert the game state data for model prediction and training
    - Additionally ACTIONS_ARRAY and ACTION_SPACE_SIZE will be needed for prediction translation and model
      variables
    
'''
from catanatron_gym.features import create_sample_vector
from catanatron_gym.envs.catanatron_env import ACTIONS_ARRAY, ACTION_SPACE_SIZE

'''
    Catanatron experimental import for Player and SimulationAccumulator classes and game registering
    - Without importing then base classes and registration to build upon, trying to connect from-scratch 
      player classes and accumulators would take a significant amount of time and effort to integrate, 
      with the addition of missing out on the useful built in game functions
    
'''
from catanatron_experimental.cli.cli_players import register_player
from catanatron_experimental import SimulationAccumulator, register_accumulator

@register_accumulator
class GameActionCounter(SimulationAccumulator):
    robber_counter = 0
    trade_counter = 0
    dev_counter = 0
    knight_counter = 0
    monopoly_counter = 0
    roadbuild_counter = 0
    yearofplenty_counter = 0
    
    game_counter = 0
    turn_count = 0
    
    cities = 0
    settlements = 0
    roads = 0
    
    vpcards = 0
    vp = 0
    
    color = Color("WHITE")
    csv = 'test1_com.csv'
    
    def before(self, game):
        self.robber_counter = 0
        self.trade_counter = 0
        self.dev_counter = 0
        self.knight_counter = 0
        self.monopoly_counter = 0
        self.roadbuild_counter = 0
        self.yearofplenty_counter = 0
        
        self.cities = 0
        self.settlements = 0
        self.roads = 0
        
        self.vpcards = 0
        self.vp = 0
        
        if not os.path.exists(self.csv):
            header = [
                'game','trades', 'robbers', 'dev_cards_bought', 'knights',
                'monopolys', 'roadbuilds', 'yearofplentys', 'vpcards', 'victorypoints',
                'cities', 'settlements', 'roads', 'model_actions', 'helper_actions'
            ]
            with open(self.csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
        
        # Start a new row for the new game
        with open(self.csv, mode='r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            max_game_counter_row = max(rows, key=lambda x: int(x['game']), default=None)
            if max_game_counter_row:
                self.game_counter = int(max_game_counter_row['game'])
        
        with open(self.csv, mode='a', newline='') as file:
            self.game_counter += 1
            writer = csv.writer(file)
            row = [self.game_counter] + [0] * 14  # Initialize row with zeroes
            writer.writerow(row)
        
    def step(self, game_after_action, action):
        if action.color == self.color:
            if action.action_type == ActionType.MOVE_ROBBER:
                self.robber_counter += 1
            if action.action_type == ActionType.MARITIME_TRADE:
                self.trade_counter += 1
            if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                self.dev_counter += 1
            if action.action_type == ActionType.PLAY_KNIGHT_CARD:
                self.knight_counter += 1
            if action.action_type == ActionType.PLAY_MONOPOLY:
                self.monopoly_counter += 1
            if action.action_type == ActionType.PLAY_ROAD_BUILDING:
                self.roadbuild_counter += 1
            if action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
                self.yearofplenty_counter += 1
            
        self.cities = len(get_player_buildings(game_after_action.state, self.color, 'CITY')) or 0
        self.settlements = len(get_player_buildings(game_after_action.state, self.color, 'SETTLEMENT')) or 0
        self.roads = len(get_player_buildings(game_after_action.state, self.color, 'ROAD')) or 0
        
        self.vpcards = get_dev_cards_in_hand(game_after_action.state, self.color, dev_card="VICTORY_POINT")
        self.vp = get_actual_victory_points(game_after_action.state, self.color)

    def after(self, game):
        victorypoints = 10 if game.winning_color() == self.color else self.vp
        
        # Read the CSV file and find the row with the largest game_counter value
        with open(self.csv, mode='r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        # Find the row with the largest game_counter
        max_game_counter_row = max(rows, key=lambda x: int(x['game']), default=None)
        if max_game_counter_row:
            max_game_counter_row.update({
                'trades': self.trade_counter,
                'robbers': self.robber_counter,
                'dev_cards_bought': self.dev_counter,
                'knights': self.knight_counter,
                'monopolys': self.monopoly_counter,
                'roadbuilds': self.roadbuild_counter,
                'yearofplentys': self.yearofplenty_counter,
                'vpcards': self.vpcards,
                'victorypoints': victorypoints,
                'cities': self.cities,
                'settlements': self.settlements,
                'roads': self.roads
            })

        # Write the updated data back to the CSV file
        with open(self.csv, mode='w', newline='') as file:
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

def create_q_network(state_shape, action_space):
    model = Sequential()
    
    # Input layer of 256 Neurons and an imput shape of the game state
    model.add(Dense(256, input_dim=state_shape, activation='relu'))
    model.add(Dropout(0.2)) # Dropout layer
    
    # First Hidden Layer of 128 Neurons
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2)) # Dropout layer
    
    # Second Hidden Layer of 64 Neurons
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2)) # Dropout layer
    
    # Third Hidden Layer of 32 Neurons
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2)) # Dropout layer
    
    # Fourth Hidden Layer of 16 Neurons
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2)) # Dropout layer
    
    # Fifth Hidden Layer of 8 Neurons
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2)) # Dropout layer
    
    # Sixth Hidden Layer of 4 Neurons
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.2)) # Dropout layer
    
    # Output layer
    model.add(Dense(action_space, activation='linear'))
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')
    
    model.summary()
    return model

@register_player("AI")
class RLModel(Player):
    state_shape = None # For detecting change of state
    action_space = None
    model = None

    # Variable configuration for Replay Buffer and learning
    replay_buffer = deque(maxlen=2000)
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    discount_factor = 0.99
    batch_size = 32
    
    # Local Variables for tracking and condition checking
    vpcards = 0
    points = 0
    
    reward_points = 0
    last_state = None
    turn_count = 0
    model_action_count = 0
    other_action_count = 0
    
    csv = 'test1_com.csv'
    game_seed = 0
    training_phase = False

    def __init___(self, color): # Initialise the class
        super(RLModel, self).__init__(color)
        self.tester = FooPlayer()

    def decide(self, game, playable_actions):
        if self.game_seed != game.seed: # Checks if the next game has started
            self.reset() # Resets local variables and trackers
            self.game_seed = game.seed # Resets game check
            self.game_phase = True  # Changes to indicate that a game has ended so the model understands
        elif self.game_seed == game.seed:
            self.game_phase = False # To tell the model that the game has ended and a new one has begun
            
        done = self.game_phase
        if self.turn_count != game.state.num_turns:
            self.turn_count = game.state.num_turns
            self.reward_points = 0
        
        # Get the current state vector for the game
        current_state_vector = create_sample_vector(game, self.color)
        
        if self.model is None:
            # For loading a prexisting model to train
            self.state_shape = len(create_sample_vector(game, self.color))
            self.action_space = ACTION_SPACE_SIZE
            self.load_model()
            
            # For creating a fresh model for testing and training
            # shape_vector = len(create_sample_vector(game, self.color))
            # self.initialize(state_shape=shape_vector, action_space=ACTION_SPACE_SIZE)

        if len(playable_actions) == 1:
            return playable_actions[0]  # Handles Roll, Discard
        
        # self.print_stats(game)

        # Translate action into proper tuple format for passing and checking
        action = self.act(game, playable_actions)
        action_tuple = self.translate_action(action)
        action = Action(color=self.color, action_type=action_tuple[0], value=action_tuple[1])
        
        # If playable action it adds the count for the model, if not then for the helper
        if action in playable_actions:
            # print("PLAYABLE", action_tuple)
            self.model_action_count += 1
        else:
            action = self.decider(game, playable_actions)
            self.other_action_count += 1
 
        # Reward function call
        reward = self.calculate_reward(game, action, len(playable_actions))

        # Remember if last state is not None
        if self.last_state is not None:
            self.remember(self.last_state, action, self.reward_points, current_state_vector, done)

        # Runs replay buffer every time the turn count mods 4 with no remainder, to give the game time to run
        if len(self.replay_buffer) >= self.batch_size and self.turn_count % 4 == 0:
            self.replay()
            
        # Update counters
        self.turn_count += 1
        self.vpcards = get_dev_cards_in_hand(game.state, self.color, dev_card="VICTORY_POINT")
        self.points = get_actual_victory_points(game.state, self.color)
        self.last_state = current_state_vector
        self.save_counts()
        return action

    # Initalize function for model creation
    def initialize(self, state_shape, action_space):
        self.state_shape = state_shape
        self.action_space = action_space
        self.model = create_q_network(state_shape, action_space)

    # Act function for translating and predicting
    def act(self, game, playable_actions):
        state_vector = create_sample_vector(game, self.color)
        state_vector = np.array(state_vector)  # Ensure it's a NumPy array
        state_vector = state_vector.reshape(1, -1)  # Reshape to 2D array with shape (1, input_dim)

        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space) # Return random choice if epsilon is too high still

        q_values = self.model.predict(state_vector) # Model predicts based on the state vector

        best_action = None
        highest_q_value = float('-inf')
                
        for action in playable_actions: # Calculates the approximate aciton based on the value
            action_tuple = self.translate_action(action) # Translate the number into an action using the actions array provided by the library
            action_idx = ACTIONS_ARRAY.index(action_tuple)
            if q_values[0][action_idx] > highest_q_value:
                best_action = action
                highest_q_value = q_values[0][action_idx]
        
        # Returns action once done
        return best_action

    # Remember function to save data to the replay buffer
    def remember(self, state, action, reward, next_state, done):
        if self.state_shape is None:
            # Error handling
            print("Warning: state_shape is None, skipping remember.")
            return  # Skip the rest of the method
        
        state = np.reshape(state, [1, self.state_shape])
        next_state = np.reshape(next_state, [1, self.state_shape])
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.save_model() # Saves model

    # Replay function that replayes experiences stored in replay buffer
    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return # Error handling
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward # Calculated reward
            if not done:
                target = reward + self.discount_factor * np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(state) # Predicts on past experience
            
            action_tuple = self.translate_action(action) # Translates the action
            action_idx = ACTIONS_ARRAY.index(action_tuple)

            target_f[0][action_idx] = target
            self.model.fit(state, target_f, epochs=10, verbose=0) 
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay # Decays epsilon as progress is made
            
    def calculate_reward(self, game, action, check):
        reward = self.reward_points # Compiles and adds on the current reward value for the turn
        if action.action_type == ActionType.BUILD_CITY: reward += 0.6 
        if action.action_type == ActionType.BUILD_SETTLEMENT: reward += 0.5 
        if action.action_type == ActionType.BUILD_ROAD: reward += 0.3 
        if action.action_type == ActionType.MOVE_ROBBER: reward += 0.2 
        if action.action_type == ActionType.MARITIME_TRADE: reward += 0.2 
        if action.action_type == ActionType.PLAY_KNIGHT_CARD: reward += 0.3 
        if action.action_type == ActionType.PLAY_MONOPOLY: reward += 0.3 
        if action.action_type == ActionType.PLAY_ROAD_BUILDING: reward += 0.3 
        if action.action_type == ActionType.PLAY_YEAR_OF_PLENTY: reward += 0.3 
        if action.action_type == ActionType.END_TURN and check > 1: reward += 0
        self.reward_points = reward # Returns reward value after the end of calculations

    # Function for printing stats to check training in realtime
    def print_stats(self, game):
        player_freqdeck = get_player_freqdeck(game.state, self.color)
        cities = get_player_buildings(game.state, self.color, 'CITY')
        settlements = get_player_buildings(game.state, self.color, 'SETTLEMENT')
        roads = get_player_buildings(game.state, self.color, 'ROAD')
            
        print("[=================================================================]")
        print("Cities: ", cities)
        print("Settlements: ", settlements)
        print("Roads: ", roads)
        print("Resources: ", player_freqdeck)
        print("Model Actions: ", self.model_action_count)
        print("Other Actions: ", self.other_action_count)
        
    # Function for translating the action into the proper tuple
    def translate_action(self, action):
        if isinstance(action, int):
            action = ACTIONS_ARRAY[action]
            return action 
        
        # Certain actions required specific attention due to how they need to be passed
        if action.action_type == ActionType.MOVE_ROBBER:
            action_tuple = (action.action_type, action.value[0])
        elif action.action_type == ActionType.END_TURN:
            action_tuple = (action.action_type, action.value)
        else:
            action_tuple = (action.action_type, action.value)
        return action_tuple
        
    # Function for saving the model
    def save_model(self):
        # Define the path for saving the model
        model_path = os.path.join(os.path.dirname(__file__), "rl_test_com.h5")
        
        # Save the entire model for future loading
        self.model.save(model_path)
        print(f"Model saved at {model_path}")
        
    # Function for loadin the model
    def load_model(self):
        # Define the path where the model is saved
        model_path = os.path.join(os.path.dirname(__file__), "rl_test_com.h5")

        # Check if the model file exists before loading
        if os.path.exists(model_path):
            try:
                # Load the saved model from the specified path
                self.model = tf.keras.models.load_model(model_path)
                print(f"Model loaded successfully from {model_path}")
            except Exception as e: # Error handling
                print(f"Error loading model from {model_path}: {str(e)}")
                self.model = None
        else: # Error handling
            print(f"No saved model found at {model_path}. Starting with a new model.")
            self.model = None
            
    # Function to save count statistics
    def save_counts(self):
        with open(self.csv, mode='r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        # Find the row with the largest game_counter
        max_game_counter_row = max(rows, key=lambda x: int(x['game']), default=None)
        if max_game_counter_row:
            max_game_counter_row.update({
                'model_actions': self.model_action_count,
                'helper_actions': self.other_action_count
            })

        # Write the updated data back to the CSV file
        with open(self.csv, mode='w', newline='') as file:
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    # Function to reset counts
    def reset(self):
        self.turn_count = 0
        self.last_state = None

        self.model_action_count = 0
        self.other_action_count = 0
            
#------------------------------------------------------------------------------------------------#
'''
    This is the helper function that provides reliable performance.
    It is a non-commented version of Kane, to keep the line count low and the code compact.

'''
    def decider(self, game, playable_actions):
        player_freqdeck = get_player_freqdeck(game.state, self.color) # Player's hand
        cities = get_player_buildings(game.state, self.color, 'CITY') # Player's cities 
        settlements = get_player_buildings(game.state, self.color, 'SETTLEMENT') # Player's settlements
        roads = get_player_buildings(game.state, self.color, 'ROAD') # Player's roads
        
        knights = get_dev_cards_in_hand(game.state, self.color, dev_card="KNIGHT") # Player's dev cards
        monopolys = get_dev_cards_in_hand(game.state, self.color, dev_card="MONOPOLY")
        roadbuilds = get_dev_cards_in_hand(game.state, self.color, dev_card="ROAD_BUILDING")
        yearofplentys = get_dev_cards_in_hand(game.state, self.color, dev_card="YEAR_OF_PLENTY")
        victorys = get_dev_cards_in_hand(game.state, self.color, dev_card="VICTORY_POINT")
        
        #---------------------------------------------------------------------------------------#
        player_nodes = set(settlements + cities)
        port_nodes = game.state.board.map.port_nodes
        game_map = game.state.board.map
        
        if len(playable_actions) == 1:
            if playable_actions[0].action_type == ActionType.ROLL:
                return playable_actions[0]
            elif playable_actions[0].action_type == ActionType.END_TURN: 
                return playable_actions[0]
            elif playable_actions[0].action_type == ActionType.DISCARD:
                return playable_actions[0]
        
        for action in playable_actions:
            if action.action_type == ActionType.MOVE_ROBBER:
                return action
            elif freqdeck_contains(player_freqdeck, SETTLEMENT_COST_FREQDECK) == False and action.action_type == ActionType.BUILD_ROAD:
                return action
            elif freqdeck_contains(player_freqdeck, SETTLEMENT_COST_FREQDECK) == False and action.action_type == ActionType.BUILD_SETTLEMENT:
                return action
        
        special_card_action = self.play_special_card(playable_actions)
        if special_card_action:
            return special_card_action
        
        best_action = self.main_checks(player_freqdeck, settlements, player_nodes, port_nodes, playable_actions)
        if best_action:
            return best_action
        else:
            return playable_actions[0]
        
    def main_checks(self, player_freqdeck, settlements, player_nodes, port_nodes, playable_actions):
        if freqdeck_contains(player_freqdeck, CITY_COST_FREQDECK):
            for action in playable_actions:
                if action.action_type == ActionType.BUILD_CITY and action.value in settlements:
                    return action
        else:
            needed_resources = self.check_difference(player_freqdeck, CITY_COST_FREQDECK)
            if needed_resources:
                trade_action = self.trade_resource(player_freqdeck, needed_resources, player_nodes, port_nodes, playable_actions)
                if trade_action:
                    return trade_action

        if freqdeck_contains(player_freqdeck, SETTLEMENT_COST_FREQDECK):
            for action in playable_actions:
                if action.action_type == ActionType.BUILD_SETTLEMENT:
                    return action
        else:
            needed_resources = self.check_difference(player_freqdeck, SETTLEMENT_COST_FREQDECK)
            if needed_resources:
                trade_action = self.trade_resource(player_freqdeck, needed_resources, player_nodes, port_nodes, playable_actions)
                if trade_action:
                    return trade_action

        if freqdeck_contains(player_freqdeck, ROAD_COST_FREQDECK):
            for action in playable_actions:
                if action.action_type == ActionType.BUILD_ROAD:
                    return action
        else:
            needed_resources = self.check_difference(player_freqdeck, ROAD_COST_FREQDECK)
            if needed_resources:
                trade_action = self.trade_resource(player_freqdeck, needed_resources, player_nodes, port_nodes, playable_actions)
                if trade_action:
                    return trade_action

        if freqdeck_contains(player_freqdeck, DEVELOPMENT_CARD_COST_FREQDECK):
            for action in playable_actions:
                if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                    return action
        else:
            needed_resources = self.check_difference(player_freqdeck, DEVELOPMENT_CARD_COST_FREQDECK)
            if needed_resources:
                trade_action = self.trade_resource(player_freqdeck, needed_resources, player_nodes, port_nodes, playable_actions)
                if trade_action:
                    return trade_action

        return None

    def trade_resource(self, player_freqdeck, required_resources, player_nodes, port_nodes, playable_actions):
        port_access = self.check_ports(player_nodes, port_nodes)
        resource_tag = ['WOOD','BRICK','SHEEP','WHEAT','ORE']
        
        for i in range(len(required_resources)):
            if required_resources[i] > 0:
                ask = resource_tag[i]
                if port_access is not None:
                    for j in range(len(resource_tag)):
                        if resource_tag[j] in port_access:
                            give = resource_tag[j]
                            surplus = self.check_surplus(player_freqdeck, 2)
                            if surplus != False and surplus[j] > 1 and resource_tag[j] != ask:
                                trade_action = Action(color=self.color, action_type=ActionType.MARITIME_TRADE, value=(give,give,None,None,ask))
                                return trade_action

                    if None in port_access:
                        surplus = self.check_surplus(player_freqdeck, 3)
                        if surplus != False:
                            for k in range(len(surplus)):
                                if surplus[k] > 0:
                                    give = resource_tag[k]
                                    trade_action = Action(color=self.color, action_type=ActionType.MARITIME_TRADE, value=(give,give,give,None,ask))
                                    return trade_action
                    
                surplus = self.check_surplus(player_freqdeck, 4)
                if surplus != False:
                    for l in range(len(surplus)):
                        if surplus[l] > 0:
                            give = resource_tag[l]
                            trade_action = Action(color=self.color, action_type=ActionType.MARITIME_TRADE, value=(give,give,give,give,ask))
                            return trade_action
        return None
            

    def check_surplus(self, player_freqdeck, ratio):
        surplus = [0,0,0,0,0]

        for i in range(len(player_freqdeck)):
            if player_freqdeck[i] >= ratio:
                surplus[i] = player_freqdeck[i] 

        if any(surplus):
            return surplus
        else:
            return False
    
    def check_difference(self, player_freqdeck, target_building):
        needed = [0,0,0,0,0]
        check = 0
        for r in range(len(player_freqdeck)):
            if player_freqdeck[r] < target_building[r]:
                needed[r] = target_building[r] - player_freqdeck[r]
                check = 1
        if check == 1: 
            return needed 
        else: 
            return False
    
    def get_adjacent_tiles(self, state):
        adjacent_tiles = {}
        board = state.board
        for node_id in board.nodes:
            adjacent_tiles[node_id] = get_adjacent_tiles(board, node_id)
        return adjacent_tiles
    
    def get_node_resources(self, game_map, node_id):
        adjacent_tiles = game_map.adjacent_tiles[node_id]
        resource_tiles = []

        for tile in adjacent_tiles:
            if tile.resource is not None:
                resource_info = {'resource': tile.resource, 'number': tile.number}
                resource_tiles.append(resource_info)

        return resource_tiles
    
    def play_special_card(self, playable_actions):
        for action in playable_actions:
            if action.action_type == ActionType.PLAY_KNIGHT_CARD:
                    return action
            elif action.action_type == ActionType.PLAY_MONOPOLY:
                    return action
            elif action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
                    return action
            elif action.action_type == ActionType.PLAY_ROAD_BUILDING:
                    return action

        return None
    
    def check_ports(self, player_nodes, port_nodes):
        port_access = set()
        for resource, nodes in port_nodes.items():
            for node in nodes:
                if node in player_nodes:
                    port_access.add(resource)

        return port_access if port_access else None
            