from typing import Any
import random

class Virtue:
    def __init__(self, name) -> None:
        self.name = name

    def get_name(self):
        return self.name

# To be selected by the player
class Trait:
    def __init__(self, personality, value) -> None:
        self.personality = personality
        self.value = value

    def get_personality(self):
        return self.personality
    
    def get_value(self):
        return self.value

# To be selected by the player
class Occupation:
    def __init__(self, personality, value) -> None:
        self.personality = personality
        self.value = value

    def get_personality(self):
        return self.personality
    
    def get_value(self):
        return self.value

# To be selected randomly since this is arbitrary
class Feature:
    def __init__(self, personality, value) -> None:
        self.personality = personality
        self.value = value

    def get_personality(self):
        return self.personality
    
    def get_value(self):
        return self.value

class Satisfaction:
    def __init__(self) -> None:
        self.score = 0

    def get_score(self):
        return self.score

    def update_score(self, value):
        self.score += value
        if self.score < 0:
            self.score = 0

class Player:
    def __init__(self, traits, occupation, features) -> None:
        self.traits = traits
        self.occupation = occupation
        self.features = features
        self.personality_dict = {
            "discipline" : 0, 
            "curiosity" : 0, 
            "extraversion" : 0, 
            "sensitivity" : 0, 
            "gentleness" : 0, 
            "sincerity" : 0
        }
        self.update_personality_tokens(occupation.get_personality().get_name(), occupation.get_value())
        for feature in features:
            self.update_personality_tokens(feature.get_personality().get_name(), feature.get_value())
        
        self.satisfaction = Satisfaction()

    def get_traits(self):
        return self.traits
    
    def get_trait_by_personality(self, personality):
        for trait in self.traits:
            if personality in trait.get_personality().get_name():
                return trait.value
        return 0
            

    def get_occupation(self):
        return self.occupation
    
    def get_features(self):
        return self.features
    
    def get_personality_tokens(self):
        return self.personality_dict
    
    def get_personality_tokens_by_key(self, personality_name):
        return self.personality_dict[personality_name]
    
    def get_satisfaction(self):
        return self.satisfaction
    
    def get_choice(self):
        return self.choice

    def set_choice(self, choice):
        self.choice = choice
    
    def update_personality_tokens(self, personality, value):
        self.personality_dict[personality] += value
    
    def update_satisfaction(self, value):
        self.satisfaction.update_score(value)

class Scene:
    def __init__(self, options, match_condition) -> None:
        self.options = options
        self.match_condition = match_condition

    def get_scene(self):
        return self.options, self.match_condition

    def play_turn(self, player_1, player_2):
        player1_choice = player_1.get_choice()
        player2_choice = player_2.get_choice()
        for i, option in enumerate(self.options):
            if int(player1_choice) == i:
                for personality, value in option.items():
                    if personality != "satisfaction":
                        player_1.update_personality_tokens(personality, value)
                    else:
                        player_1.update_satisfaction(value)
            
            if int(player2_choice) == i:
                for personality, value in option.items():
                    if personality != "satisfaction":
                        player_2.update_personality_tokens(personality, value)
                    else:
                        player_2.update_satisfaction(value)
        
        if player1_choice == player2_choice:
            player_1.update_satisfaction(self.match_condition["match"])
            player_2.update_satisfaction(self.match_condition["match"])
        else:
            player_1.update_satisfaction(self.match_condition["no_match"])
            player_2.update_satisfaction(self.match_condition["no_match"])

        return self.options
    
 
class Destiny:
    def __init__(self) -> None:
        pass

def scene_generator():
    
    """ test_options = [{
            "sensitivity": 1, 
            "discipline": -1
        }, {
            "sensitivity": -1,
            "satisfaction": 1
        }, {
            "discipline": 1,
            "satisfaction": 1
        }
    ]
    test_match_condition = {
        "match": 2,
        "no_match": -1,
    } """

    #initialize personality
    variable_list = ["discipline", "curiosity", "extraversion", "sensitivity", "gentleness", "sincerity", "satisfaction"]

    num_choices = 3

    options = []
    for choice in range(num_choices):
        num_vars = random.randint(1, 3)
        selected_vars = random.sample(variable_list, k=num_vars)
        option_dict = {}
        for var in selected_vars:
            option_dict[var] = random.choice([-1, 1])
        options.append(option_dict)
    
    match_condition = {
        "match": random.randint(0, 5),
        "no_match": random.randint(-4, 0)
    }

    return options, match_condition



if __name__ == "__main__":
    print("Welcome to Fog of Love!")

    #initialize personality
    personalities = [Virtue("discipline"), Virtue("curiosity"), Virtue("extraversion"), Virtue("sensitivity"), Virtue("gentleness"), Virtue("sincerity")]

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

    print("The trait goals of Player 1 are: ")

    for i, trait in enumerate(player1_traits_sample):
        print(str(i) + ". " + str(trait.get_personality().get_name()) + " " +  str(trait.get_value()))

    player2_traits_sample = random.sample(traits, k=3)

    print("The trait goals of Player 2 are: ")

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
    print("The occupation of Player 1 is: ")
    print(str(i) + ". " + str(player1_occupations_sample[0].get_personality().get_name()) + " " +  str(player1_occupations_sample[0].get_value()))

    player2_occupations_sample = random.sample(occupations, k=1)
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

    print("The features Player 2 liked about Player 1 are: ")
    for i, feature in enumerate(player1_features_sample):
        print(str(i) + ". " + str(feature.get_personality().get_name()) + " " +  str(feature.get_value()))


    player2_features_sample = random.sample(features, k=3)

    print("The features Player 1 liked about Player 2 are: ")
    for i, feature in enumerate(player2_features_sample):
        print(str(i) + ". " + str(feature.get_personality().get_name()) + " " +  str(feature.get_value()))


    player_1 = Player(player1_traits_sample, player1_occupations_sample[0], player1_features_sample)
    print(player_1.get_personality_tokens(), player_1.get_satisfaction().get_score())

    player_2 = Player(player2_traits_sample, player2_occupations_sample[0], player2_features_sample)
    print(player_2.get_personality_tokens(), player_2.get_satisfaction().get_score())

    for i in range(3):
        print("Scene #" + str(i+1))
        options, match_condition = scene_generator()
        scene = Scene(options, match_condition)

        print(scene.get_scene())

        player1_choice = -1
        while player1_choice == -1 or player1_choice > str(len(options) - 1):
            player1_choice = input("Player 1 choose an option between 0 and " + str(len(options) - 1) + ": ")

        player_1.set_choice(player1_choice)

        player2_choice = -1
        while player2_choice == -1 or player2_choice > str(len(options) - 1):
            player2_choice = input("Player 2 choose an option between 0 and " + str(len(options) - 1) + ": ")

        player_2.set_choice(player2_choice)

        scene.play_turn(player_1, player_2)
        
        print(player_1.get_personality_tokens(), player_1.get_satisfaction().get_score())
        print(player_2.get_personality_tokens(), player_2.get_satisfaction().get_score())




    # Overall goals: Traits, Satisfaction, Destiny
    # Trait goals can be shared
    # +3 for success -1 for none
    # Equal partners = Difference of satisfaction is <=3, player's satisfaction is >20
    # Love Team = Gentleness >= 6, sum of satisfaction >= 60





