'''
    Import packages for other libraries
    
'''
import os
import numpy
import random
import numpy as np


'''
    Tensorflow imports
    - For model loading
    
'''
import tensorflow as tf

'''
    Catanatron imports for specific enums and tuple data shapes 
    - Required for creating and customising actions for passing
        
'''
from collections import defaultdict, Counter
from catanatron import Player, Action, ActionType


'''
    Catanatron model imports for specific functions, classes and enums 
    - Imports are from majority of the files in models due to their importance rooted in the game engine.
        
'''
from catanatron.models.enums import FastResource
from catanatron.models.map import get_node_counter_production
from catanatron.models.decks import CITY_COST_FREQDECK, SETTLEMENT_COST_FREQDECK, ROAD_COST_FREQDECK, DEVELOPMENT_CARD_COST_FREQDECK, freqdeck_contains

'''
    Catanatron state_function import for specific functions, and player key
    - Many of these functions are utilised for checking resources for other actions and decision making
    
'''
from catanatron.state_functions import player_key, get_player_freqdeck, get_player_buildings, get_dev_cards_in_hand, player_has_rolled, get_longest_road_length, get_visible_victory_points, get_actual_victory_points

'''
    Catanatron experimental import for Player and SimulationAccumulator classes and game registering
    - Without importing then base classes and registration to build upon, trying to connect from-scratch 
      player classes and accumulators would take a significant amount of time and effort to integrate, 
      with the addition of missing out on the useful built in game functions
    
'''
from catanatron_experimental.cli.cli_players import register_player
from catanatron_experimental import SimulationAccumulator, register_accumulator

'''
    Catanatron gym import for specific functions, and action space
    - Utilising the provided functions we can convert the game state data for model prediction and training
    - Additionally ACTIONS_ARRAY and ACTION_SPACE_SIZE will be needed for prediction translation and model
      variables
    
'''
from catanatron_gym.features import create_sample_vector
from catanatron_gym.envs.catanatron_env import ACTIONS_ARRAY, ACTION_SPACE_SIZE

#----------------------------------------------------------------------------------------------------------------------------#

'''
    Pre-programmed AI Agent Kane
'''
@register_player("KANE")
class Kane(Player):
    dev_track = 0
    city_track = 0
    settlement_track = 0
    road_track = 0
    devbuy_track = 0
    trade_track = 0
    
    # The main decision function that handles overall logic
    def decide(self, game, playable_actions):
        
        # Retrieves all the necessary numbers and resource decks needed
        player_freqdeck = get_player_freqdeck(game.state, self.color) # Player's hand
        cities = get_player_buildings(game.state, self.color, 'CITY') # Player's cities 
        settlements = get_player_buildings(game.state, self.color, 'SETTLEMENT') # Player's settlements
        roads = get_player_buildings(game.state, self.color, 'ROAD') # Player's roads
        
        knights = get_dev_cards_in_hand(game.state, self.color, dev_card="KNIGHT") # Player's dev cards
        monopolys = get_dev_cards_in_hand(game.state, self.color, dev_card="MONOPOLY") # Player's monopoly card
        roadbuilds = get_dev_cards_in_hand(game.state, self.color, dev_card="ROAD_BUILDING") # Player's year of plenty cards
        yearofplentys = get_dev_cards_in_hand(game.state, self.color, dev_card="YEAR_OF_PLENTY") # Player's year of plenty cards
        victorys = get_dev_cards_in_hand(game.state, self.color, dev_card="VICTORY_POINT") # Player's victory point cards
        
        #---------------------------------------------------------------------------------------#
        player_nodes = set(settlements + cities) # Gets player's nodes (settlements and cities)
        port_nodes = game.state.board.map.port_nodes # Gets game's port nodes (That offers cheaper trades
        game_map = game.state.board.map # Gets game map dict
        
        # Printing statistics on the player for tracking and logging
        # self.print_stats(player_freqdeck, cities, settlements, roads, game_map)
        
        # print("[---------------------------------------------------------------------------]")
        # print("Turn: ", game.state.num_turns)
        # print("[---------------------------------------------------------------------------]")
        
        # This handling manages for when only a single action can be played
        if len(playable_actions) == 1:
            if playable_actions[0].action_type == ActionType.ROLL:
                # print("[-------------------------------------------------------------------Play Roll]")
                return playable_actions[0]
            elif playable_actions[0].action_type == ActionType.END_TURN: 
                # print("[-------------------------------------------------------------------End Turn]")
                return playable_actions[0]
            elif playable_actions[0].action_type == ActionType.DISCARD:
                # print("[------------------------------------------------------------------Play Discard]")
                return playable_actions[0]
        
        # This handling manages robber play actions and the start of turn free builds
        for action in playable_actions:
            if action.action_type == ActionType.MOVE_ROBBER:
                # print("[-------------------------------------------------------------------Play Robber]")
                return action
            elif freqdeck_contains(player_freqdeck, SETTLEMENT_COST_FREQDECK) == False and action.action_type == ActionType.BUILD_ROAD:
                # print("[------------------------------------------------------------------Build Free Road]")
                return action
            elif freqdeck_contains(player_freqdeck, SETTLEMENT_COST_FREQDECK) == False and action.action_type == ActionType.BUILD_SETTLEMENT:
                # print("[------------------------------------------------------------------Build Free Settlement]")
                return action
            
        # print(self.check_ports(player_nodes, port_nodes))
        
        # Handle special cards before main checks
        # self.print_dev(knights, monopolys, roadbuilds, yearofplentys, victorys)
        special_card_action = self.play_special_card(playable_actions)
        if special_card_action:
            return special_card_action
        
        # Performs the main bulk of decision making
        best_action = self.main_checks(game, player_freqdeck, settlements, player_nodes, port_nodes, playable_actions)
        # self.print_overall_stats()
        if best_action:
            return best_action
        else:
            return playable_actions[0]

    # Function for printing stats to check training in realtime
    def print_stats(self, player_freqdeck, cities, settlements, roads, game_map):
        # Printing data on the player to check what it has and does
        print("[=================================================================]")
        print("Cities: ", cities)
        print("Settlements: ", settlements)
        print("Roads: ", roads)
        print("Resources: ", player_freqdeck)
        
        # # Retrieving resources at each settlement
        resources_at_settlements = {}        
        for node_id in settlements:
            resources_at_settlements[node_id] = self.get_node_resources(game_map, node_id)
        print("Resource points:", resources_at_settlements)
    
    # Function for printing development card stats to check training in realtime
    def print_dev(self, knights, monopolys, roadbuilds, yearofplentys, victorys):
        print("[=========================+")
        print("Dev Cards In Hand")
        print("Knights: ", knights)
        print("Monopoly: ",monopolys)
        print("Road-Build: ",roadbuilds)
        print("Year of Plenty: ",yearofplentys)
        print("Victory Points: ",victorys)
        print("[=========================+")
        
    # Function for printing overall stats of all games to check training in realtime
    # Deppreciated to an extent due the introduction of the statistic accumulator in kane_test.py
    def print_overall_stats(self):
        print("[==============================)")
        print("Cities Built: " ,self.city_track)
        print("Settlements Built: " ,self.settlement_track)
        print("Roads Built: " ,self.road_track)
        print("Trades Performed: " ,self.trade_track)
        print("Dev Cards Bought:" ,self.devbuy_track)
        print("Dev Cards Played: " ,self.dev_track)
        print("[==============================)")
        
    # Function that handles the main priority action decisions
    def main_checks(self, game, player_freqdeck, settlements, player_nodes, port_nodes, playable_actions):
        """
        This function makes decisions based on priority, resources at hand and port node access for trades
        """
        # Check for the ability to build a city
        if freqdeck_contains(player_freqdeck, CITY_COST_FREQDECK):
            for action in playable_actions: # Double checks to see if it adheres to the game engine
                if action.action_type == ActionType.BUILD_CITY and action.value in settlements:
                    # print(f"[------------------------Building a city at {action.value}")
                    self.city_track += 1
                    return action
        else:
            needed_resources = self.check_difference(player_freqdeck, CITY_COST_FREQDECK)
            if needed_resources:
                trade_action = self.trade_resource(game, player_freqdeck, needed_resources, player_nodes, port_nodes, playable_actions)
                if trade_action:
                    # print(f"[------------------------Trading to build a city")
                    self.trade_track += 1
                    return trade_action

        # Check for the ability to build a settlement
        if freqdeck_contains(player_freqdeck, SETTLEMENT_COST_FREQDECK):
            for action in playable_actions: # Double checks to see if it adheres to the game engine
                if action.action_type == ActionType.BUILD_SETTLEMENT:
                    # print(f"[------------------------Building a settlement at {action.value}")
                    self.settlement_track += 1
                    return action
        else:
            needed_resources = self.check_difference(player_freqdeck, SETTLEMENT_COST_FREQDECK)
            if needed_resources:
                trade_action = self.trade_resource(game, player_freqdeck, needed_resources, player_nodes, port_nodes, playable_actions)
                if trade_action:
                    # print(f"[------------------------Trading to build a settlement")
                    self.trade_track += 1
                    return trade_action

        # Check for the ability to build a road
        if freqdeck_contains(player_freqdeck, ROAD_COST_FREQDECK):
            for action in playable_actions: # Double checks to see if it adheres to the game engine
                if action.action_type == ActionType.BUILD_ROAD:
                    # print(f"[------------------------Building a road on {action.value}")
                    self.road_track += 1
                    return action
        else:
            needed_resources = self.check_difference(player_freqdeck, ROAD_COST_FREQDECK)
            if needed_resources:
                trade_action = self.trade_resource(game, player_freqdeck, needed_resources, player_nodes, port_nodes, playable_actions)
                if trade_action:
                    # print(f"[------------------------Trading to build a road")
                    self.trade_track += 1
                    return trade_action

        # Check for the ability to buy a development card
        if freqdeck_contains(player_freqdeck, DEVELOPMENT_CARD_COST_FREQDECK):
            for action in playable_actions: # Double checks to see if it adheres to the game engine
                if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                    # print(f"[------------------------Buying a development card")
                    self.devbuy_track += 1
                    return action
        else:
            needed_resources = self.check_difference(player_freqdeck, DEVELOPMENT_CARD_COST_FREQDECK)
            if needed_resources:
                trade_action = self.trade_resource(game, player_freqdeck, needed_resources, player_nodes, port_nodes, playable_actions)
                if trade_action:
                    # print(f"[------------------------Trading to buy a development card")
                    self.trade_track += 1
                    return trade_action

        # If no action can be taken, return None or consider another fallback strategy
        # print("[--------------------------------No valid actions available with current resources.")
        return None

    def trade_resource(self, game, player_freqdeck, required_resources, player_nodes, port_nodes, playable_actions):
        """
        Perform a maritime trade if the player has a surplus of resources.
        
        The logic is as follows, if the bank has available resources of the type that we need we proceed.
        We check the port access and work downwards, if there is port access and a resource is provided it means there
        are 2 trades available. If port access is providing None then its a general 3 resource trade port. Otherwise it 
        falls back on the default 4 resource trade.

        Parameters:
        - player_freqdeck (list): An array representing the player's resources [wood, brick, sheep, wheat, ore].
        - required_resources (list): An array representing the desired resources [wood, brick, sheep, wheat, ore].

        Returns:
        - action (Action): The maritime trade action if a valid trade is possible, otherwise None.
        """
        
        # Local Variables needed for trade conditions and functionality
        port_access = self.check_ports(player_nodes, port_nodes)
        resource_tag = ['WOOD','BRICK','SHEEP','WHEAT','ORE']
        availability = game.state.resource_freqdeck # Gets the back's resources for checking
        
        for i in range(len(required_resources)):
            if required_resources[i] > 0 and availability[i] > 0: # Checks the bank availability
                ask = resource_tag[i]
                if port_access is not None: # If there is a port_access available
                    for j in range(len(resource_tag)):
                        if resource_tag[j] in port_access: # If resource defines it checks for surplus of 2
                            give = resource_tag[j]
                            surplus = self.check_surplus(player_freqdeck, 2)
                            if surplus != False and surplus[j] > 1 and resource_tag[j] != ask:
                                trade_action = Action(color=self.color, action_type=ActionType.MARITIME_TRADE, value=(give,give,None,None,ask))
                                return trade_action

                    if None in port_access: # If resource not defined it checks for surplus of 3
                        surplus = self.check_surplus(player_freqdeck, 3) 
                        if surplus != False:
                            for k in range(len(surplus)):
                                if surplus[k] > 0:
                                    give = resource_tag[k]
                                    trade_action = Action(color=self.color, action_type=ActionType.MARITIME_TRADE, value=(give,give,give,None,ask))
                                    return trade_action
                    
                surplus = self.check_surplus(player_freqdeck, 4) # If no port acess it checks for surplus of 4
                if surplus != False:
                    for l in range(len(surplus)):
                        if surplus[l] > 0:
                            # print(resource_tag[l])
                            give = resource_tag[l]
                            trade_action = Action(color=self.color, action_type=ActionType.MARITIME_TRADE, value=(give,give,give,give,ask))
                            return trade_action
                        
            # check if other players have required resource
                # check if we have surplus of any kind in loop
                    # create trade with surplus and target and return action
        return None
            

    # Function for checking surplus dynamically based on input
    def check_surplus(self, player_freqdeck, ratio):
        """
        Check if the player has a surplus of any resources (4 or more in quantity).

        Parameters:
        - player_freqdeck (list): An array representing the player's resources [wood, brick, sheep, wheat, ore].
        - ratio (int): An int that determines how much surplus we are looking for.

        Returns:
        - surplus (list): An array representing the surplus of each resource [wood, brick, sheep, wheat, ore].
                           Values will be 4 or more if there is a surplus, otherwise 0.
        """
        surplus = [0,0,0,0,0]

        for i in range(len(player_freqdeck)):
            if player_freqdeck[i] >= ratio:
                surplus[i] = player_freqdeck[i] 

        if any(surplus):
            # print("Surplus resources detected:", surplus)
            return surplus
        else:
            return False
    
    # Function checks difference to see how much of what is needed for a target action/building
    def check_difference(self, player_freqdeck, target_building):
        """
        Check how many resources are needed by the player for an action

        Parameters:
        - player_freqdeck (list): An array representing the player's resources [wood, brick, sheep, wheat, ore].
        - target_building (list): An array representing what the player needs [wood, brick, sheep, wheat, ore].

        Returns:
        - needed (list): An array representing the resource and how much is needed [wood, brick, sheep, wheat, ore]
          otherwise 0.
        """
        needed = [0,0,0,0,0]
        check = 0
        for r in range(len(player_freqdeck)):
            if player_freqdeck[r] < target_building[r]:
                needed[r] = target_building[r] - player_freqdeck[r]
                check = 1
        if check == 1: 
            # print(needed)
            return needed 
        else: 
            return False
    
    # Function for checking adjacent tiles to get what resources the player has access to
    def get_adjacent_tiles(self, state):
        """
        Extract adjacent tiles for each node from the game state.
        """
        # Use `state['board'].nodes` to get nodes and their adjacency information
        adjacent_tiles = {}
        board = state.board
        for node_id in board.nodes:
            adjacent_tiles[node_id] = get_adjacent_tiles(board, node_id)
        return adjacent_tiles
    
    # Function for getting node resources, using get_adjacent_tiles to help
    def get_node_resources(self, game_map, node_id):
        """
        Retrieve the resources produced by the tiles adjacent to the node.

        Returns a list of dictionaries, where each dictionary represents a resource tile 
        with its resource type and roll number.
        """
        # Access the adjacent tiles for the given node_id directly from the dictionary
        adjacent_tiles = game_map.adjacent_tiles[node_id]

        # List to hold the details of each resource tile separately
        resource_tiles = []

        for tile in adjacent_tiles:
            if tile.resource is not None:
                # Create a dictionary for each tile to store its resource type and roll number
                resource_info = {'resource': tile.resource, 'number': tile.number}
                resource_tiles.append(resource_info)

        return resource_tiles
    
    # Function that handles playing special cards
    def play_special_card(self, playable_actions):
        """
        This function checks to see if the player has any development cards, if so then it will play if able.

        Input Parameters:
        - playable_actions: List of actions the player can take that the game provides.

        Output/Returns:
        - action (Action): Development Card action
        """
        # Iterate over playable actions and decide if a special card should be played
        for action in playable_actions:
            if action.action_type == ActionType.PLAY_KNIGHT_CARD:
                    # print(f"Playing a Knight card")
                    self.dev_track += 1
                    return action
            elif action.action_type == ActionType.PLAY_MONOPOLY:
                    # print(f"Playing a Monopoly card")
                    self.dev_track += 1
                    return action
            elif action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
                    # print(f"Playing a Year of Plenty card")
                    self.dev_track += 1
                    return action
            elif action.action_type == ActionType.PLAY_ROAD_BUILDING:
                    # print(f"Playing a Road Building card")
                    self.dev_track += 1
                    return action

        # If no special card is played, return None
        return None
    
    # Function to check ports, works in conjuction with other handling like trades etc.
    def check_ports(self, player_nodes, port_nodes):
        """
        Check if the player's settlements or cities have access to any ports.

        Parameters:
        - player_nodes (set): A set of node IDs where the player has settlements or cities.
        - port_nodes (defaultdict): A defaultdict where keys are resources (or None for 3:1 ports) and values are sets of node IDs.

        Returns:
        - port_access (set): A set of unique resource strings indicating port access.
        """
        port_access = set()

        for resource, nodes in port_nodes.items():
            for node in nodes:
                if node in player_nodes:
                    port_access.add(resource)

        return port_access if port_access else None

#----------------------------------------------------------------------------------------------------------------------------# 
    
@register_player("HUMAN")
class HumanPlayer(Player):
    """Human player that selects which action to take using standard input"""

    def decide(self, game, playable_actions):
        
        port_nodes = game.state.board.map.port_nodes
        game_map = game.state.board.map
        
        index = len(game.state.colors)
        for i in range(index):
            if game.state.colors[i] != self.color:
                opp_color = game.state.colors[i]
                player_freqdeck = get_player_freqdeck(game.state, opp_color)  # Player's hand
                cities = get_player_buildings(game.state, opp_color, 'CITY')  # Player's cities 
                settlements = get_player_buildings(game.state, opp_color, 'SETTLEMENT')  # Player's settlements
                roads = get_player_buildings(game.state, opp_color, 'ROAD')  # Player's roads
                dev = get_dev_cards_in_hand(game.state, opp_color)
                
                vp = get_visible_victory_points(game.state, opp_color)
                player_nodes = set(settlements + cities)

                print("[=================================================================]")
                print("\033[1m[======================================================= Turn: ", game.state.num_turns, "]\033[0m")
                if opp_color.name == 'WHITE': print("\033[1m[======================================= WHITE's Stats ===========]\033[0m")
                if opp_color.name == 'BLUE': print("\033[1m\033[34m[======================================== BLUE's Stats ===========]\033[0m")
                if opp_color.name == 'RED': print("\033[1m\033[31m[======================================== RED's Stats ============]\033[0m")
                if opp_color.name == 'ORANGE': print("\033[1m\033[38;5;214m[======================================== ORANGE's Stats =========]\033[0m")
                self.print_stats(player_freqdeck, cities, settlements, roads, game_map, vp)
                print("[=================================================================]")
                print("[=================================================================]")
                print("Development Cards: ",dev)
                print("Resource Ports: ", self.check_ports(player_nodes, port_nodes))
        
        print("[=================================================================]")
        print("\033[1m[===================== It's Your Turn! ================= Turn: ", game.state.num_turns, "]\033[0m")
        print("\033[1m[===================== Here are your actions: ====================]\033[0m")
        for i, action in enumerate(playable_actions):
            print(f"{i}: {action.action_type} {action.value}")
        i = None
        
        player_freqdeck = get_player_freqdeck(game.state, self.color)  # Player's hand
        cities = get_player_buildings(game.state, self.color, 'CITY')  # Player's cities 
        settlements = get_player_buildings(game.state, self.color, 'SETTLEMENT')  # Player's settlements
        roads = get_player_buildings(game.state, self.color, 'ROAD')  # Player's roads

        knights = get_dev_cards_in_hand(game.state, self.color, dev_card="KNIGHT")  # Player's dev cards
        monopolys = get_dev_cards_in_hand(game.state, self.color, dev_card="MONOPOLY")
        roadbuilds = get_dev_cards_in_hand(game.state, self.color, dev_card="ROAD_BUILDING")
        yearofplentys = get_dev_cards_in_hand(game.state, self.color, dev_card="YEAR_OF_PLENTY")
        victorys = get_dev_cards_in_hand(game.state, self.color, dev_card="VICTORY_POINT")

        vp = get_actual_victory_points(game.state, self.color)
        player_nodes = set(settlements + cities)
        
        print("[=================================================================]")
        bank_resources = game.state.resource_freqdeck
        bank_dev = len(game.state.development_listdeck)
        print("[==================Bank Resources: ", bank_resources,"=========]")
        print("[==================Bank Dev Cards: ", bank_dev,"===========================]")
        self.print_stats(player_freqdeck, cities, settlements, roads, game_map, vp)
        self.print_dev(knights, monopolys, roadbuilds, yearofplentys, victorys)
        print("Resource Ports: ", self.check_ports(player_nodes, port_nodes))
        
        while i is None or (i < 0 or i >= len(playable_actions)):
            print("Please enter a valid index:")
            try:
                x = input(">>> ")
                i = int(x)
            except ValueError:
                pass

        return playable_actions[i]    

    def print_stats(self, player_freqdeck, cities, settlements, roads, game_map, vp):
        print("[=================================================================]")
        print("Cities: ", cities)
        print("Settlements: ", settlements)
        print("Roads: ", roads)
        print("Resources: ", player_freqdeck)
        print("Victory Points: ", vp)
        print("[=================================================================]")
        print("[=================================================================]")
        # Retrieving resources at each settlement
        resources_at_settlements = {}
        resources_at_cities = {}
        for node_id in settlements: resources_at_settlements[node_id] = self.get_node_resources(game_map, node_id)
        for node_id in cities: resources_at_cities[node_id] = self.get_node_resources(game_map, node_id)
        print("City Resource Points: ", resources_at_cities)
        print("Settlement Resource Points: ", resources_at_settlements)
        
    def print_dev(self, knights, monopolys, roadbuilds, yearofplentys, victorys):
        print("[=================================================================]")
        print("[=================================================================]")
        print("Dev Cards In Hand")
        print("Knights: ", knights)
        print("Monopoly: ", monopolys)
        print("Road-Build: ", roadbuilds)
        print("Year of Plenty: ", yearofplentys)
        print("Victory Points: ", victorys)
        print("[=================================================================]")
        print("[=================================================================]")

    def check_ports(self, player_nodes, port_nodes):
        # Checking player's access to ports
        port_access = set()
        for resource, nodes in port_nodes.items():
            for node in nodes:
                if node in player_nodes:
                    port_access.add(resource)
        return port_access if port_access else None
    
    def get_node_resources(self, game_map, node_id):
        # Retrieving the resources produced by tiles adjacent to the node
        adjacent_tiles = game_map.adjacent_tiles[node_id]
        resource_tiles = []
        for tile in adjacent_tiles:
            if tile.resource is not None:
                resource_info = {'resource': tile.resource, 'number': tile.number}
                resource_tiles.append(resource_info)
        return resource_tiles

#----------------------------------------------------------------------------------------------------------------------------#     

@register_player("AI")
class RLModel(Player):
    model = None
    # path = "rl_test_lite.h5"
    path = "rl_test_com.h5"
    action_space = None
    
    def decide(self, game, playable_actions):
        if self.model == None:
            self.state_shape = len(create_sample_vector(game, self.color))
            self.action_space = ACTION_SPACE_SIZE
            self.load_model()
            
        if len(playable_actions) == 1:
            return playable_actions[0]  # Handles Roll, Discard
        
        # Translate action into proper tuple format for passing and checking
        action = self.act(game, playable_actions)
        action_tuple = self.translate_action(action)
        action = Action(color=self.color, action_type=action_tuple[0], value=action_tuple[1])
        
        if action not in playable_actions:
            action = random.choice(playable_actions)
        return action
        
    def act(self, game, playable_actions):
        state_vector = create_sample_vector(game, self.color)
        state_vector = np.array(state_vector)  # Ensure it's a NumPy array
        state_vector = state_vector.reshape(1, -1)  # Reshape to 2D array with shape (1, input_dim)

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
        
    def translate_action(self, action):
        if isinstance(action, int):
            action = ACTIONS_ARRAY[action]
            return action 
        
        if action.action_type == ActionType.MOVE_ROBBER:
            action_tuple = (action.action_type, action.value[0])
        elif action.action_type == ActionType.END_TURN:
            action_tuple = (action.action_type, action.value)
        else:
            action_tuple = (action.action_type, action.value)
        return action_tuple
    
    def load_model(self):
        # Define the path where the model is saved
        model_path = os.path.join(os.path.dirname(__file__), self.path)

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