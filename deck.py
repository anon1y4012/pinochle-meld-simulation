import random
import config as config

#A class to represent a Card deck. Functions: Shuffle, deal
class Deck:
    
    #Create a deck based on deck configuration
    def __init__(self):
        self.cards = [ value+suit for value in config.values for suit in config.suits ]
        self.cards *= config.occurrences

    #shuffle the deck. No return value
    def shuffle(self):
        random.shuffle(self.cards)

    #deal the deck to the given number of players. Return as a list of lists   
    def deal(self, numPlayers):
        results = [[] for player in range(numPlayers)]
        for (index, card) in enumerate(self.cards):
            playerIndex = index % numPlayers
            results[playerIndex].append(card)
        return results
            
            