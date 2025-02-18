import random
import config

class Deck:
    def __init__(self):
        self.cards = [value + suit for value in config.values for suit in config.suits] * config.occurrences

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, numPlayers):
        return [self.cards[i::numPlayers] for i in range(numPlayers)]