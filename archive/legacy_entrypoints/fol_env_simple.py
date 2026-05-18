import gym
from gym import spaces
import numpy as np
import fol_attributes as fol
import random
from fol_attributes_simple import *
import pprint

class FoLEnvironmentSimple(gym.Env):
    def __init__(self, debug=False):
        super(FoLEnvironmentSimple, self).__init__()

        # Define the action and observation spaces
        self.personality_names = ["discipline"]
        self.satisfaction_name = "satisfaction"
        self.goal_names = ["discipline"]
        self.agents = ["player_1", "player_2"]
        self.num_agents = 2
        self.debug = debug
        self.action_space = {"player_1": spaces.Discrete(3), "player_2":spaces.Discrete(3)}  # 3 possible actions: 0, 1, 2
        self.observation_space = {"player_1": spaces.Dict({
            "player_1_discipline": spaces.Box(low=-50, high=50, shape=(), dtype=np.int32),
            "player_1_satisfaction": spaces.Discrete(50),
            "player_2_discipline": spaces.Box(low=-50, high=50, shape=(), dtype=np.int32),
            "player_2_satisfaction": spaces.Discrete(50),
            "option_1_discipline": spaces.Box(low=-1, high=1, shape=(), dtype=np.int32),
            "option_1_satisfaction": spaces.Box(low=-1, high=1, shape=(), dtype=np.int32),
            "option_2_discipline": spaces.Box(low=-1, high=1, shape=(), dtype=np.int32),
            "option_2_satisfaction": spaces.Box(low=-1, high=1, shape=(), dtype=np.int32),
            "option_3_discipline": spaces.Box(low=-1, high=1, shape=(), dtype=np.int32),
            "option_3_satisfaction": spaces.Box(low=-1, high=1, shape=(), dtype=np.int32),
            "match": spaces.Box(low=-1, high=1, shape=(), dtype=np.int32),
            "no_match": spaces.Box(low=-1, high=1, shape=(), dtype=np.int32),
            "goal_discipline": spaces.Box(low=-50, high=50, shape=(), dtype=np.int32),
            "goal_satisfaction": spaces.Discrete(50)
            }), 
            "player_2": spaces.Dict({
                "player_1_discipline": spaces.Box(low=-50, high=50, shape=(), dtype=np.int32),
                "player_1_satisfaction": spaces.Discrete(50),
                "player_2_discipline": spaces.Box(low=-50, high=50, shape=(), dtype=np.int32),
                "player_2_satisfaction": spaces.Discrete(50),
                "option_1_discipline": spaces.Box(low=-1, high=1, shape=(), dtype=np.int32),
                "option_1_satisfaction": spaces.Box(low=-1, high=1, shape=(), dtype=np.int32),
                "option_2_discipline": spaces.Box(low=-1, high=1, shape=(), dtype=np.int32),
                "option_2_satisfaction": spaces.Box(low=-1, high=1, shape=(), dtype=np.int32),
                "option_3_discipline": spaces.Box(low=-1, high=1, shape=(), dtype=np.int32),
                "option_3_satisfaction": spaces.Box(low=-1, high=1, shape=(), dtype=np.int32),
                "match": spaces.Box(low=-1, high=1, shape=(), dtype=np.int32),
                "no_match": spaces.Box(low=-1, high=1, shape=(), dtype=np.int32),
                "goal_discipline": spaces.Box(low=-50, high=50, shape=(), dtype=np.int32),
                "goal_satisfaction": spaces.Discrete(50)
        })}

    def reset(self):
        # Reset the environment to initial state
        self.state = {}
        #initialize personality
        personalities = [Virtue("discipline")]

        if self.debug:
            print("DEBUGGING MODE: ON")

        #initialize traits
        num_traits = 38
        max_trait_val = 5
        min_trait_val = -5
        traits = []
        for i in range(num_traits):
            personality = personalities[random.randint(0, 5)]
            value = 0
            while value == 0:
                value = random.randint(min_trait_val, max_trait_val)
            traits.append(Trait(personality, value))
            
        player1_traits_sample = random.sample(traits, k=3)

        if self.debug:
            print("The traits of Player 1 are: ")

            for i, trait in enumerate(player1_traits_sample):
                print(str(i) + ". " + str(trait.get_personality().get_name()) + " " +  str(trait.get_value()))

        player2_traits_sample = random.sample(traits, k=3)

        if self.debug:
            print("The traits of Player 2 are: ")
    
            for i, trait in enumerate(player2_traits_sample):
                print(str(i) + ". " + str(trait.get_personality().get_name()) + " " +  str(trait.get_value()))
        
        # initialize occupation
        num_occupations = 20
        occupations = []
        for i in range(num_occupations):
            personality = personalities[random.randint(0, 5)]
            value = 0
            while value == 0:
                value = random.randint(-1, 1)
            occupations.append(Occupation(personality, value))

            
        player1_occupations_sample = random.sample(occupations, k=1)
        if self.debug:
            print("The occupation of Player 1 is: ")
            print(str(i) + ". " + str(player1_occupations_sample[0].get_personality().get_name()) + " " +  str(player1_occupations_sample[0].get_value()))

        player2_occupations_sample = random.sample(occupations, k=1)
        if self.debug:
            print("The occupation of Player 2 is: ")
            print(str(i) + ". " + str(player2_occupations_sample[0].get_personality().get_name()) + " " +  str(player2_occupations_sample[0].get_value()))

        
        # initialize features
        num_features = 20
        features = []
        for i in range(num_features):
            personality = personalities[random.randint(0, 5)]
            value = 0
            while value == 0:
                value = random.randint(-1, 1)
            features.append(Feature(personality, value))

        player1_features_sample = random.sample(features, k=3)

        if self.debug:
            print("The features Player 2 liked about Player 1 are: ")
            for i, feature in enumerate(player1_features_sample):
                print(str(i) + ". " + str(feature.get_personality().get_name()) + " " +  str(feature.get_value()))


        player2_features_sample = random.sample(features, k=3)
        
        if self.debug:
            print("The features Player 1 liked about Player 2 are: ")
            for i, feature in enumerate(player2_features_sample):
                print(str(i) + ". " + str(feature.get_personality().get_name()) + " " +  str(feature.get_value()))


        player_1 = Player(player1_traits_sample, player1_occupations_sample[0], player1_features_sample)

        if self.debug:
            print("Personality profile", player_1.get_personality_tokens(), player_1.get_satisfaction().get_score())

        player_2 = Player(player2_traits_sample, player2_occupations_sample[0], player2_features_sample)
        
        if self.debug:
            print("Personality profile", player_2.get_personality_tokens(), player_2.get_satisfaction().get_score())

            print("Scene #" + str(1))
        options, match_condition = scene_generator()
        scene = Scene(options, match_condition)

        if self.debug:
            print(scene.get_scene())

        #print(scene.get_scene())
        self.state = {  
            "player_1": {
                "player_1_discipline": player_1.get_personality_tokens_by_key("discipline"),
                "player_1_satisfaction": player_1.get_satisfaction().get_score(),
                "player_2_discipline": player_2.get_personality_tokens_by_key("discipline"),
                "player_2_satisfaction": player_2.get_satisfaction().get_score(),
                "player_1_goal_discipline": player_1.get_trait_by_personality("discipline"),
                "player_1_goal_satisfaction": 30
            },
            "player_2": {
                "player_2_discipline": player_2.get_personality_tokens_by_key("discipline"),
                "player_2_satisfaction": player_2.get_satisfaction().get_score(),
                "player_1_discipline": player_1.get_personality_tokens_by_key("discipline"),
                "player_1_satisfaction": player_1.get_satisfaction().get_score(),
                "player_2_goal_discipline": player_2.get_trait_by_personality("discipline"),
                "player_2_goal_satisfaction": 30
            }
        }
        for i, option in enumerate(options):
            option_dict = {
                "discipline": 0,
                "satisfaction": 0
            }
            for trait in option.keys():
                option_dict[trait] = option[trait]
            for trait in option_dict.keys():    
                self.state["player_1"]["option_" + str(i+1) + "_" + trait] = option_dict[trait]
                self.state["player_2"]["option_" + str(i+1) + "_" + trait] = option_dict[trait]

        self.state["player_1"]["match"] = match_condition["match"]
        self.state["player_2"]["match"] = match_condition["match"]
        self.state["player_1"]["no_match"] = match_condition["no_match"]
        self.state["player_2"]["no_match"] = match_condition["no_match"]
        #print(self.state)
        self.steps_taken = 0
        if self.debug:
            print("THE OLD STATE")
            pprint.pprint(self.state)
        self.truncation = {"player_1": False, "player_2": False}
        return self.state, {"player_1": "TURN ENDED", "player_2": "TURN ENDED"}

    def step(self, actions):
        # Update the environment state based on the chosen action
        rewards = self._get_reward(actions)
        if self.debug:
            print("rewards", rewards)
        self._update_state(actions)
        done = self._is_done()

        self.steps_taken += 1
        if self.debug:
            print("STEP #", self.steps_taken)
            print("THE NEW STATE")
            pprint.pprint(self.state)
        if self._is_done():
            self.truncation = {"player_1": True, "player_2": True}

        # Return the updated state, reward, done flag, and any additional info
        return self.state, rewards,  {"player_1": False, "player_2": False}, self.truncation, {"player_1": "TURN ENDED", "player_2": "TURN ENDED"}

    def render(self):
        # Display or visualize the environment state (optional)
        print(f"Current state: {self.state}")

    def get_items_starting_with(self, agent, starting_string):
        result = {}
        for key, value in self.state[agent].items():
            if key.startswith(starting_string):
                result[key] = value
        return result
    
    def _get_reward(self, actions):
        # always assume player 1 is the player and player 2 is the opponent. we want to train player 1.
        # if player 1 and player 2 choose the same thing, reward = satisfaction
        # else reward = negative satisfaction
        # players rewarded for making choices that improve their traits
        # Calculate the reward based on the chosen action and the current state
        rewards = {"player_1": 0, "player_2": 0}
        options = self.get_items_starting_with("player_1", "option_")
        self.options = {key.replace("option_", ""): value for key, value in options.items() if key.startswith("option_")}
        if self.debug:
            print("actions: ", actions)
        self.player_1_action = actions["player_1"]
        self.player_2_action = actions["player_2"]
        self.player_1_choice = {key.replace(str(self.player_1_action[0] + 1) + "_", ""): value for key, value in self.options.items() if key.startswith(str(self.player_1_action[0] + 1) + "_")}
        self.player_2_choice = {key.replace(str(self.player_2_action[0] + 1) + "_", ""): value for key, value in self.options.items() if key.startswith(str(self.player_2_action[0] + 1) + "_")}
        player_1_goals = {key.replace("player_1_goal_", ""): value for key, value in self.get_items_starting_with("player_1", "player_1_goal_").items() if key.startswith("player_1_goal_")}
        player_2_goals = {key.replace("player_2_goal_", ""): value for key, value in self.get_items_starting_with("player_2", "player_2_goal_").items() if key.startswith("player_2_goal_")}
        
        if self.debug:
            print("player 1 choice: ", self.player_1_choice)
            print("player 2 choice: ", self.player_2_choice)
        '''if self.debug:
            print("player_1_goals, player_2_goals", player_1_goals, player_2_goals)
            print("self.player_1_action, self.player_2_action", self.player_1_action, self.player_2_action)
            print("self.player_1_choice, self.player_2_choice", self.player_1_choice, self.player_2_choice)'''
        if self.debug:
            print("PROCESSING PLAYER 1")

        for player_1_goal in player_1_goals:
            #print("player 1 goal: ", player_1_goal, player_1_goals[player_1_goal])
            #print(self.player_1_choice[player_1_goal])
            #print("SATISFACTION", player_1_goal != "satisfaction")
            self.state["player_1"]["player_1_" + player_1_goal] += self.player_1_choice[player_1_goal]
            self.state["player_2"]["player_1_" + player_1_goal] += self.player_1_choice[player_1_goal]
            if player_1_goal in self.player_1_choice:# and player_1_goal != "satisfaction":
                if player_1_goals[player_1_goal] > 0 and self.player_1_choice[player_1_goal] > 0 and self.state["player_1"]["player_1_" + player_1_goal] < player_1_goals[player_1_goal]:
                    if self.debug:
                        print("both greater than 0")
                    #self.state["player_1"]["player_1_" + player_1_goal] += self.player_1_choice[player_1_goal]
                    #self.state["player_2"]["player_1_" + player_1_goal] += self.player_1_choice[player_1_goal]
                    rewards["player_1"] += abs(self.player_1_choice[player_1_goal])
                elif player_1_goals[player_1_goal] < 0 and self.player_1_choice[player_1_goal] < 0 and self.state["player_1"]["player_1_" + player_1_goal] > player_1_goals[player_1_goal]:
                    if self.debug:
                        print("both less than zero")
                    #self.state["player_1"]["player_1_" + player_1_goal] += self.player_1_choice[player_1_goal]
                    #self.state["player_2"]["player_1_" + player_1_goal] += self.player_1_choice[player_1_goal]
                    rewards["player_1"] += abs(self.player_1_choice[player_1_goal])
                elif player_1_goals[player_1_goal] < 0 and self.player_1_choice[player_1_goal] > 0:
                    if self.debug:
                        print("goal negative, option positive")
                    #self.state["player_1"]["player_1_" + player_1_goal] += self.player_1_choice[player_1_goal]
                    #self.state["player_2"]["player_1_" + player_1_goal] += self.player_1_choice[player_1_goal]
                    rewards["player_1"] -= abs(self.player_1_choice[player_1_goal])
                elif player_1_goals[player_1_goal] > 0 and self.player_1_choice[player_1_goal] < 0:
                    if self.debug:
                        print("goal positive option negative")
                    #self.state["player_1"]["player_1_" + player_1_goal] += self.player_1_choice[player_1_goal]
                    #self.state["player_2"]["player_1_" + player_1_goal] += self.player_1_choice[player_1_goal]
                    rewards["player_1"] -= abs(self.player_1_choice[player_1_goal])
                else:
                    if self.debug:
                        print("does not meet any of the conditions")
                    rewards["player_1"] += 0
            else:
                if self.debug:
                    print("goal not in choice")
                rewards["player_1"] += 0

        
        if self.debug:
            print("PROCESSING PLAYER 2")

        for player_2_goal in player_2_goals:
            #print("player 2 goal: ", player_2_goal, player_2_goals[player_2_goal])
            #print(self.player_2_choice[player_2_goal])
            #print("SATISFACTION", player_2_goal != "satisfaction")
            self.state["player_2"]["player_2_" + player_2_goal] += self.player_2_choice[player_2_goal]
            self.state["player_1"]["player_2_" + player_2_goal] += self.player_2_choice[player_2_goal]
            if player_2_goal in self.player_2_choice:# and player_2_goal != "satisfaction":
                if player_2_goals[player_2_goal] > 0 and self.player_2_choice[player_2_goal] > 0 and self.state["player_2"]["player_2_" + player_2_goal] < player_2_goals[player_2_goal]:
                    if self.debug:
                        print("both greater than 0")
                    self.state["player_2"]["player_2_" + player_2_goal] += self.player_2_choice[player_2_goal]
                    self.state["player_1"]["player_2_" + player_2_goal] += self.player_2_choice[player_2_goal]
                    rewards["player_2"] += abs(self.player_2_choice[player_2_goal])
                elif player_2_goals[player_2_goal] < 0 and self.player_2_choice[player_2_goal] < 0 and self.state["player_2"]["player_2_" + player_2_goal] > player_2_goals[player_2_goal]:
                    if self.debug:
                        print("both less than zero")
                    self.state["player_2"]["player_2_" + player_2_goal] += self.player_2_choice[player_2_goal]
                    self.state["player_1"]["player_2_" + player_2_goal] += self.player_2_choice[player_2_goal]
                    rewards["player_2"] += abs(self.player_2_choice[player_2_goal])
                elif player_2_goals[player_2_goal] < 0 and self.player_2_choice[player_2_goal] > 0:
                    if self.debug:
                        print("goal negative, option positive")
                    self.state["player_2"]["player_2_" + player_2_goal] += self.player_2_choice[player_2_goal]
                    self.state["player_1"]["player_2_" + player_2_goal] += self.player_2_choice[player_2_goal]
                    rewards["player_2"] -= abs(self.player_2_choice[player_2_goal])
                elif player_2_goals[player_2_goal] > 0 and self.player_2_choice[player_2_goal] < 0:
                    if self.debug:
                        print("goal positive option negative")
                    self.state["player_2"]["player_2_" + player_2_goal] += self.player_2_choice[player_2_goal]
                    self.state["player_1"]["player_2_" + player_2_goal] += self.player_2_choice[player_2_goal]
                    rewards["player_2"] -= abs(self.player_2_choice[player_2_goal])
                else:
                    if self.debug:
                        print("does not meet any of the conditions")
                    rewards["player_2"] += 0
            else:
                if self.debug:
                    print("goal not in choice")
                rewards["player_2"] += 0
        
        #pprint.pprint(self.state["player_1"]["player_1_satisfaction"])
        #pprint.pprint(self.state["player_1"]["player_2_satisfaction"])
        #pprint.pprint(self.player_1_choice)
        '''if "satisfaction" in self.player_1_choice:
            #print("update option satisfaction", self.player_1_choice["satisfaction"])
            if self.player_1_choice["satisfaction"] != 0:
                rewards["player_1"] += 1#(self.player_1_choice["satisfaction"]/self.player_1_choice["satisfaction"])
            self.state["player_1"]["player_1_satisfaction"] = self.state["player_1"]["player_1_satisfaction"] + self.player_1_choice["satisfaction"]
            self.state["player_2"]["player_1_satisfaction"] = self.state["player_1"]["player_1_satisfaction"] + self.player_1_choice["satisfaction"]

        if "satisfaction" in self.player_2_choice:
            #print("update option satisfaction", self.player_2_choice["satisfaction"])
            if self.player_2_choice["satisfaction"] != 0:
                rewards["player_2"] += 1#(self.player_2_choice["satisfaction"]/self.player_2_choice["satisfaction"])
            self.state["player_2"]["player_2_satisfaction"] = self.state["player_2"]["player_2_satisfaction"] + self.player_2_choice["satisfaction"]
            self.state["player_1"]["player_2_satisfaction"] = self.state["player_2"]["player_2_satisfaction"] + self.player_2_choice["satisfaction"]
        '''
        if actions["player_1"] == actions["player_2"]:
            #print("actions match", self.state["player_1"]["match"])
            if self.state["player_1"]["match"] != 0:
                #rewards["player_1"] += self.state["player_1"]["match"]#(self.state["player_1"]["match"]/self.state["player_1"]["match"])
                #rewards["player_2"] += self.state["player_2"]["match"]
                self.state["player_1"]["player_1_satisfaction"] += self.state["player_1"]["match"]
                self.state["player_2"]["player_1_satisfaction"] += self.state["player_1"]["match"]
                self.state["player_1"]["player_2_satisfaction"] += self.state["player_2"]["match"]
                self.state["player_2"]["player_2_satisfaction"] += self.state["player_2"]["match"]
        else:
            #print("actions do not match", self.state["player_1"]["no_match"])
            if self.state["player_1"]["no_match"] != 0:
                #rewards["player_1"] += self.state["player_1"]["no_match"]
                #rewards["player_2"] += self.state["player_2"]["no_match"]
                self.state["player_1"]["player_1_satisfaction"] += self.state["player_1"]["no_match"]
                self.state["player_2"]["player_1_satisfaction"] += self.state["player_1"]["no_match"]
                self.state["player_1"]["player_2_satisfaction"] += self.state["player_2"]["no_match"]
                self.state["player_2"]["player_2_satisfaction"] += self.state["player_2"]["no_match"]

        #pprint.pprint(self.state["player_1"]["option_2_satisfaction"])
        #pprint.pprint(self.state["player_1"]["player_1_satisfaction"])
        #pprint.pprint(self.state["player_1"]["player_2_satisfaction"])
        return rewards

    def _update_state(self, actions):
        # Update the environment state based on the chosen action
        #for action in actions:
        
        #print("Scene #" + str(1))
        options, match_condition = scene_generator()
        scene = Scene(options, match_condition)
        
        if self.debug:
            print(scene.get_scene())
        for i, option in enumerate(options):
            option_dict = {
                "discipline": 0,
                "satisfaction": 0
            }
            for trait in option.keys():
                option_dict[trait] = option[trait]
            for trait in option_dict.keys():    
                self.state["player_1"]["option_" + str(i+1) + "_" + trait] = option_dict[trait]
                self.state["player_2"]["option_" + str(i+1) + "_" + trait] = option_dict[trait]

        self.state["player_1"]["match"] = match_condition["match"]
        self.state["player_2"]["match"] = match_condition["match"]
        self.state["player_1"]["no_match"] = match_condition["no_match"]
        self.state["player_2"]["no_match"] = match_condition["no_match"]
            

    def _is_done(self):
        # Determine if the episode is finished
        if self.debug:
            return self.steps_taken >= 5
        return self.steps_taken >= 100

