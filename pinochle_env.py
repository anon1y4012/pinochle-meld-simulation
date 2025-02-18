import gym
from gym import spaces
import numpy as np
import random
from deck import Deck
from hand_evaluator import evaluate
import config
from itertools import combinations

class PinochleEnv(gym.Env):
    """
    Custom Environment for Pinochle Passing Decision.
    Observation: 24-dimensional vector representing partner's hand counts.
    Action: Discrete action mapping to a combination of 3 card indices from partner's 12-card hand.
    Reward: +1 for win, -1 for loss.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PinochleEnv, self).__init__()
        self.deck = Deck()
        # Precompute all possible combinations of 3 indices out of 12
        self.combinations = list(combinations(range(12), 3))
        self.action_space = spaces.Discrete(len(self.combinations))
        # Observation: 24-dim count vector (one per card type, e.g. '9C', '10C', ... 'AC')
        self.observation_space = spaces.Box(low=0, high=12, shape=(24,), dtype=np.int32)
        self.reset()

    def encode_hand(self, hand):
        card_types = [v+s for s in config.suits for v in config.values]
        vec = np.zeros(len(card_types), dtype=np.int32)
        for card in hand:
            index = card_types.index(card)
            vec[index] += 1
        return vec

    def simulate_trick_taking(self, bidder_hand):
        # Calculate meld score and add a random trick-taking factor.
        meld_score = evaluate(bidder_hand)
        trick_factor = random.uniform(-5, 5)
        final_score = meld_score + trick_factor
        # Arbitrary threshold for win/loss outcome.
        return 1 if final_score >= 10 else -1

    def step(self, action):
        # Map action to 3 card indices in partner's hand.
        indices = self.combinations[action]
        # Sort partner's hand in a canonical order.
        partner_hand_sorted = sorted(
            self.partner_hand,
            key=lambda card: (config.suits.index(card[-1]), config.values.index(card[:-1]))
        )
        # Selected cards to pass.
        passed_cards = [partner_hand_sorted[i] for i in indices]
        # Remove passed cards from partner's hand.
        for card in passed_cards:
            self.partner_hand.remove(card)
        # Bidder receives these cards.
        self.bidder_hand.extend(passed_cards)
        # Simulate bidder passing back 3 cards: remove 3 lowest value cards.
        value_mapping = {v: i for i, v in enumerate(config.values)}
        def card_value(card):
            return value_mapping[card[:-1]]
        self.bidder_hand.sort(key=card_value)
        passed_back = self.bidder_hand[:3]
        self.bidder_hand = self.bidder_hand[3:]
        # (Optional: the passed_back cards could be added to partner's hand.)
        reward = self.simulate_trick_taking(self.bidder_hand)
        done = True  # one-step episode
        obs = self.encode_hand(self.partner_hand)
        info = {"passed_cards": passed_cards, "passed_back": passed_back, "bidder_hand": self.bidder_hand}
        return obs, reward, done, info

    def reset(self):
        # Deal two hands (12 cards each) for bidder and partner.
        self.deck = Deck()
        self.deck.shuffle()
        hands = self.deck.deal(2)
        self.bidder_hand = hands[0]
        self.partner_hand = hands[1]
        return self.encode_hand(self.partner_hand)

    def render(self, mode="human"):
        print("Bidder Hand:", self.bidder_hand)
        print("Partner Hand:", self.partner_hand)