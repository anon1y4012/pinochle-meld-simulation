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
    Custom Pinochle Environment incorporating:
      - 3-card passing (bidder and partner exchange)
      - Trick-taking simulation with actual pinochle rules.
    
    Assumptions:
      - Four hands are dealt (12 cards each).
      - ML-controlled team (bidder and partner) makes the passing decision.
      - Opponents follow simple heuristics.
      - No memory of previously played cards is kept during trick-taking.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PinochleEnv, self).__init__()
        self.deck = Deck()
        # Precompute all possible combinations of 3 indices from a 12-card hand.
        self.combinations = list(combinations(range(12), 3))
        self.action_space = spaces.Discrete(len(self.combinations))
        # Observation: partner's hand encoded as a count vector over 24 card types.
        self.observation_space = spaces.Box(low=0, high=12, shape=(24,), dtype=np.int32)
        self.trump_suit = config.trumpSuit
        self.reset()

    def encode_hand(self, hand):
        card_types = [v + s for s in config.suits for v in config.values]
        vec = np.zeros(len(card_types), dtype=np.int32)
        for card in hand:
            index = card_types.index(card)
            vec[index] += 1
        return vec

    def card_rank(self, card):
        # Returns a numeric rank for card (e.g., '10H' => rank value).
        rank = card[:-1]
        order = {'9': 0, 'J': 1, 'Q': 2, 'K': 3, '10': 4, 'A': 5}
        return order[rank]

    def simulate_trick_taking(self):
        """
        Simulate 12 tricks.
         - Each trick: players must follow suit if possible.
         - If unable, they must play a trump if available.
         - Highest card in led suit wins unless trump is played.
         - Trump cards beat non-trump cards.
         - Points: 1 point per King, 10, or Ace; extra 1 point for the last trick.
         - If a team scores 0 trick points, their meld is forfeited (round score = 0).
        """
        # Prepare players: 0 = bidder, 1 = partner (ML team); 2,3 = opponents.
        players = [
            {'hand': self.bidder_hand.copy(), 'team': 'ML'},
            {'hand': self.partner_hand.copy(), 'team': 'ML'},
            {'hand': self.opponent1_hand.copy(), 'team': 'OPP'},
            {'hand': self.opponent2_hand.copy(), 'team': 'OPP'}
        ]
        team_trick_points = {'ML': 0, 'OPP': 0}

        # Meld points calculated from remaining hands.
        ml_meld = evaluate(self.bidder_hand) + evaluate(self.partner_hand)
        opp_meld = evaluate(self.opponent1_hand) + evaluate(self.opponent2_hand)

        # Bidder leads first trick.
        leader = 0
        for trick in range(12):
            trick_cards = []  # List of (player_index, card played)
            led_suit = None

            for i in range(4):
                player_idx = (leader + i) % 4
                hand = players[player_idx]['hand']
                # Heuristic for card selection:
                if led_suit is None:
                    # Leader: choose highest non-trump if available; else lowest trump.
                    non_trump = [c for c in hand if c[-1] != self.trump_suit]
                    if non_trump:
                        card = max(non_trump, key=self.card_rank)
                    else:
                        card = min(hand, key=self.card_rank)
                    led_suit = card[-1]
                else:
                    # Must follow led suit if possible.
                    follow = [c for c in hand if c[-1] == led_suit]
                    if follow:
                        card = min(follow, key=self.card_rank)
                    else:
                        # If no card in led suit, must play a trump if available.
                        trumps = [c for c in hand if c[-1] == self.trump_suit]
                        if trumps:
                            # If a trump has already been played, try to beat it.
                            played_trumps = [tc for (_, tc) in trick_cards if tc[-1] == self.trump_suit]
                            if played_trumps:
                                current_high = max(played_trumps, key=self.card_rank)
                                better = [c for c in trumps if self.card_rank(c) > self.card_rank(current_high)]
                                if better:
                                    card = min(better, key=self.card_rank)
                                else:
                                    card = min(trumps, key=self.card_rank)
                            else:
                                card = min(trumps, key=self.card_rank)
                        else:
                            # Otherwise, play the lowest card available.
                            card = min(hand, key=self.card_rank)
                hand.remove(card)
                trick_cards.append((player_idx, card))

            # Determine winner of the trick.
            # If any trump was played, highest trump wins.
            trump_plays = [(p, c) for p, c in trick_cards if c[-1] == self.trump_suit]
            if trump_plays:
                winner, winning_card = max(trump_plays, key=lambda x: self.card_rank(x[1]))
            else:
                # Otherwise, highest card in led suit wins.
                led_plays = [(p, c) for p, c in trick_cards if c[-1] == led_suit]
                winner, winning_card = max(led_plays, key=lambda x: self.card_rank(x[1]))
            leader = winner

            # Compute trick points: King, 10, and Ace yield 1 point each.
            trick_points = sum(1 for (_, c) in trick_cards if c[:-1] in ['K', '10', 'A'])
            if trick == 11:  # Last trick bonus.
                trick_points += 1
            team = players[winner]['team']
            team_trick_points[team] += trick_points

        # If a team scores 0 in trick taking, they lose their meld.
        ml_total = (ml_meld + team_trick_points['ML']) if team_trick_points['ML'] > 0 else 0
        opp_total = (opp_meld + team_trick_points['OPP']) if team_trick_points['OPP'] > 0 else 0

        return ml_total, opp_total

    def step(self, action):
        """
        Action: index corresponding to a choice of 3 cards from partner's 12-card hand.
        Passing phase:
          - Partner passes selected 3 cards to bidder.
          - Bidder then passes back 3 cards (selected heuristically).
        Followed by trick taking simulation.
        """
        # Determine which 3 cards to pass (using sorted partner hand for consistency).
        indices = self.combinations[action]
        partner_sorted = sorted(self.partner_hand)
        passed_cards = [partner_sorted[i] for i in indices]
        for card in passed_cards:
            self.partner_hand.remove(card)
        self.bidder_hand.extend(passed_cards)

        # Bidder passes back 3 cards: heuristic – pass the 3 lowest cards.
        self.bidder_hand.sort(key=lambda c: self.card_rank(c))
        passed_back = self.bidder_hand[:3]
        self.bidder_hand = self.bidder_hand[3:]
        self.partner_hand.extend(passed_back)

        # Simulate trick taking for the round.
        ml_total, opp_total = self.simulate_trick_taking()

        # Reward is based on round outcome.
        reward = 1 if ml_total > opp_total else -1 if ml_total < opp_total else 0
        done = True
        obs = self.encode_hand(self.partner_hand)
        info = {
            "ml_total": ml_total,
            "opp_total": opp_total,
            "passed_cards": passed_cards,
            "passed_back": passed_back
        }
        return obs, reward, done, info

    def reset(self):
        # Deal four hands (12 cards each) for the round.
        self.deck = Deck()
        self.deck.shuffle()
        hands = self.deck.deal(4)
        self.bidder_hand = hands[0]
        self.partner_hand = hands[1]
        self.opponent1_hand = hands[2]
        self.opponent2_hand = hands[3]
        return self.encode_hand(self.partner_hand)

    def render(self, mode="human"):
        print("Bidder Hand:", self.bidder_hand)
        print("Partner Hand:", self.partner_hand)
        print("Opponent1 Hand:", self.opponent1_hand)
        print("Opponent2 Hand:", self.opponent2_hand)