import gym
from gym import spaces
import numpy as np
import random
from deck import Deck
from hand_evaluator import evaluate
import config
from itertools import combinations

class PinochleBiddingEnv(gym.Env):
    """
    Pinochle environment with sequential bidding, passing, and trick-taking.
    
    Player indices:
      0: ML Bidder (Player 1)
      1: Opponent (Player 2)
      2: ML Partner (Player 3)
      3: Opponent (Player 4)
    
    Bidding: Starts at 20; each player in turn can either bid (must exceed current bid) or pass.
      • If only ML teammates remain, the partner will pass.
      • The RL agent (Player 1) acts on its turn using its action:
            Action 0 means “pass”,
            Action > 0 means bidding that exact amount (adjusted upward if below current bid+1).
      • Aggressive bias: if the current bid is below 30, force a bid of at least 30.
    
    Passing: If ML wins the bid, Player 3 (partner) passes 3 cards to Player 1,
              then Player 1 returns 3 cards (via a simple heuristic).
    
    Trick-taking: The bid winner leads. If they hold the Ace of trump, they’ll lead it.
    Opponents and ML select cards using heuristics.
    
    Reward (in trick phase): For an ML win, if total (meld+trick points) meets the winning bid,
      reward = (ML trick points – Opponent trick points) + winning_bid;
      otherwise, a penalty of –winning_bid is applied.
    
    NEW RULE in Trick Taking:
      When following suit (and if no trump has been played), if a player has a card that can beat 
      the current highest card in the led suit, they must play the lowest such card. 
      (If they have no card that can beat it, they are free to play any card in that suit.)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PinochleBiddingEnv, self).__init__()
        self.deck = Deck()
        self.combinations = list(combinations(range(12), 3))
        # For bidding, we allow actions 0..50 (0 = pass, 1..50 = bid amount).
        self.action_space = spaces.Discrete(51)
        # Observation space is a flat dict with 5 keys.
        self.observation_space = spaces.Dict({
            'phase': spaces.Discrete(3),
            'hand': spaces.Box(low=0, high=12, shape=(24,), dtype=np.int32),
            'current_bid': spaces.Discrete(100),
            'active': spaces.MultiBinary(4),
            'current_turn': spaces.Discrete(4)
        })
        self.phase = 'bidding'
        self.deal_hands()
        self.init_bidding()

    def deal_hands(self):
        self.deck = Deck()
        self.deck.shuffle()
        hands = self.deck.deal(4)
        # Assign:
        #  Player 1 (index 0): ML Bidder
        #  Player 2 (index 1): Opponent
        #  Player 3 (index 2): ML Partner
        #  Player 4 (index 3): Opponent
        self.hands = hands

    def init_bidding(self):
        self.phase = 'bidding'
        self.current_bid = 20
        self.active = [True, True, True, True]
        # Assume dealer is Player 4 so bidding starts with Player 1.
        self.current_turn = 0  
        self.bidding_history = []  # List of tuples: (player_index, bid)
    
    def encode_hand(self, hand):
        card_types = [v + s for s in config.suits for v in config.values]
        vec = np.zeros(len(card_types), dtype=np.int32)
        for card in hand:
            index = card_types.index(card)
            vec[index] += 1
        return vec

    def get_bidding_observation(self):
        return {
            'phase': 0,
            'hand': self.encode_hand(self.hands[0]),  # ML Bidder’s hand
            'current_bid': self.current_bid,
            'active': np.array(self.active, dtype=np.int8),
            'current_turn': self.current_turn
        }

    def get_passing_observation(self):
        return {
            'phase': 1,
            'hand': self.encode_hand(self.hands[2]),
            'current_bid': self.current_bid,
            'active': np.array(self.active, dtype=np.int8),
            'current_turn': self.current_turn
        }

    def get_trick_observation(self):
        return {
            'phase': 2,
            'hand': self.encode_hand(self.hands[0]),
            'current_bid': self.current_bid,
            'active': np.array(self.active, dtype=np.int8),
            'current_turn': self.current_turn
        }

    def step(self, action):
        if self.phase == 'bidding':
            # Bidding turn:
            if self.current_turn == 0:  # ML Bidder’s turn
                if action == 0:
                    bid = 0
                else:
                    bid = action
                    if bid <= self.current_bid:
                        bid = self.current_bid + 1
                    if self.current_bid < 30 and bid < 30:
                        bid = max(30, self.current_bid + 1)
                self.bidding_history.append((0, bid))
                if bid == 0:
                    self.active[0] = False
                else:
                    self.current_bid = bid
                self.current_turn = (self.current_turn + 1) % 4
            else:
                # Non-ML players use heuristic bidding.
                if self.current_turn == 2 and self.active[0] and self.active[2] and (not self.active[1] and not self.active[3]):
                    bid = 0
                else:
                    meld_score = evaluate(self.hands[self.current_turn])
                    threshold = 10
                    if meld_score >= self.current_bid - threshold:
                        bid = self.current_bid + 1 + random.randint(0, 3)
                        if self.current_bid < 30 and bid < 30:
                            bid = max(30, self.current_bid + 1)
                        if bid > 50:
                            bid = 50
                    else:
                        bid = 0
                self.bidding_history.append((self.current_turn, bid))
                if bid == 0:
                    self.active[self.current_turn] = False
                else:
                    self.current_bid = bid
                self.current_turn = (self.current_turn + 1) % 4

            if sum(self.active) == 1:
                winner = self.active.index(True)
                self.winning_bid = 20  
                self.winning_player = winner
                self.winning_team = 'ML' if winner in [0, 2] else 'OPP'
                if winner in [0, 2]:
                    self.trump_suit = self.select_trump(self.hands[0])
                else:
                    self.trump_suit = self.select_trump(self.hands[winner])
                self.phase = 'passing' if self.winning_team == 'ML' else 'trick'
            if len(self.bidding_history) >= 4 and all(bid == 0 for (_, bid) in self.bidding_history[-4:]):
                return self.reset(), 0, True, {"info": "All players passed; redeal."}
            return self.get_bidding_observation(), 0, False, {"bidding_history": self.bidding_history}

        elif self.phase == 'passing':
            action = action % len(self.combinations)
            indices = self.combinations[action]
            partner_sorted = sorted(self.hands[2])
            passed_cards = [partner_sorted[i] for i in indices]
            for card in passed_cards:
                self.hands[2].remove(card)
            self.hands[0].extend(passed_cards)
            self.hands[0].sort(key=lambda c: self.card_rank(c))
            passed_back = self.hands[0][:3]
            self.hands[0] = self.hands[0][3:]
            self.hands[2].extend(passed_back)
            self.phase = 'trick'
            return self.get_trick_observation(), 0, False, {"passed_cards": passed_cards, "passed_back": passed_back}

        elif self.phase == 'trick':
            ml_total, opp_total, ml_trick_points, opp_trick_points = self.simulate_round()
            if self.winning_team == 'ML':
                if ml_total >= self.winning_bid:
                    reward = (ml_trick_points - opp_trick_points) + self.winning_bid
                else:
                    reward = -self.winning_bid
            else:
                reward = ml_total
            done = True
            return self.get_trick_observation(), reward, done, {"ml_total": ml_total, "opp_total": opp_total,
                                                                 "winning_bid": self.winning_bid,
                                                                 "winning_team": self.winning_team,
                                                                 "ml_trick_points": ml_trick_points,
                                                                 "opp_trick_points": opp_trick_points}

    def select_trump(self, hand):
        counts = {s: 0 for s in config.suits}
        for card in hand:
            counts[card[-1]] += 1
        return max(counts, key=counts.get)

    def simulate_round(self):
        # Trick-taking simulation over 12 tricks.
        players = [
            {"hand": self.hands[0].copy(), "team": "ML", "label": "Player 1 (ML Bidder)"},
            {"hand": self.hands[1].copy(), "team": "OPP", "label": "Player 2 (Opponent)"},
            {"hand": self.hands[2].copy(), "team": "ML", "label": "Player 3 (ML Partner)"},
            {"hand": self.hands[3].copy(), "team": "OPP", "label": "Player 4 (Opponent)"}
        ]
        team_trick_points = {"ML": 0, "OPP": 0}
        leader = self.winning_player  # Bid winner leads.
        for trick in range(12):
            trick_cards = []
            led_suit = None
            for i in range(4):
                player_idx = (leader + i) % 4
                hand = players[player_idx]["hand"]
                if not hand:
                    continue
                if led_suit is None:
                    if player_idx == self.winning_player and any(c[-1] == self.trump_suit and c.startswith("A") for c in hand):
                        card = [c for c in hand if c[-1] == self.trump_suit and c.startswith("A")][0]
                    else:
                        non_trump = [c for c in hand if c[-1] != self.trump_suit]
                        card = max(non_trump, key=self.card_rank) if non_trump else min(hand, key=self.card_rank)
                    led_suit = card[-1]
                else:
                    follow = [c for c in hand if c[-1] == led_suit]
                    if follow:
                        # Check if a trump has been played in this trick.
                        trump_played = any(c[-1] == self.trump_suit for (_, c) in trick_cards)
                        if not trump_played:
                            # Identify the current highest card in led suit.
                            current_follow = [c for (_, c) in trick_cards if c[-1] == led_suit]
                            if current_follow:
                                current_high = max(current_follow, key=self.card_rank)
                                # Determine cards in hand that can beat current_high.
                                can_beat = [c for c in follow if self.card_rank(c) > self.card_rank(current_high)]
                                if can_beat:
                                    # Forced play: must play the lowest card that beats current_high.
                                    card = min(can_beat, key=self.card_rank)
                                else:
                                    # If no card can beat current_high, free to choose any card in follow.
                                    card = min(follow, key=self.card_rank)
                            else:
                                card = min(follow, key=self.card_rank)
                        else:
                            card = min(follow, key=self.card_rank)
                    else:
                        trumps = [c for c in hand if c[-1] == self.trump_suit]
                        if trumps:
                            played_trumps = [tc for (_, tc) in trick_cards if tc[-1] == self.trump_suit]
                            if played_trumps:
                                current_high = max(played_trumps, key=self.card_rank)
                                better = [c for c in trumps if self.card_rank(c) > self.card_rank(current_high)]
                                card = min(better, key=self.card_rank) if better else min(trumps, key=self.card_rank)
                            else:
                                card = min(trumps, key=self.card_rank)
                        else:
                            card = min(hand, key=self.card_rank)
                hand.remove(card)
                trick_cards.append((player_idx, card))
            trump_plays = [(p, c) for p, c in trick_cards if c[-1] == self.trump_suit]
            if trump_plays:
                winner, winning_card = max(trump_plays, key=lambda x: self.card_rank(x[1]))
            else:
                led_plays = [(p, c) for p, c in trick_cards if c[-1] == led_suit]
                winner, winning_card = max(led_plays, key=lambda x: self.card_rank(x[1]))
            leader = winner
            trick_points = sum(1 for (_, c) in trick_cards if c[:-1] in ['K', '10', 'A'])
            if trick == 11:
                trick_points += 1  # Bonus for last trick.
            team = players[winner]["team"]
            team_trick_points[team] += trick_points
        ml_meld = evaluate(self.hands[0]) + evaluate(self.hands[2])
        opp_meld = evaluate(self.hands[1]) + evaluate(self.hands[3])
        ml_trick_points = team_trick_points["ML"]
        opp_trick_points = team_trick_points["OPP"]
        ml_total = ml_meld + ml_trick_points
        opp_total = opp_meld + opp_trick_points
        return ml_total, opp_total, ml_trick_points, opp_trick_points

    def card_rank(self, card):
        order = {'9': 0, 'J': 1, 'Q': 2, 'K': 3, '10': 4, 'A': 5}
        for r in ['10', '9', 'J', 'Q', 'K', 'A']:
            if card.startswith(r):
                return order[r]
        return 0

    def reset(self):
        self.deal_hands()
        self.init_bidding()
        return self.get_bidding_observation()

    def render(self, mode="human"):
        print("Phase:", self.phase)
        print("Hands:")
        print(" Player 1 (ML Bidder):", self.hands[0])
        print(" Player 2 (Opponent):", self.hands[1])
        print(" Player 3 (ML Partner):", self.hands[2])
        print(" Player 4 (Opponent):", self.hands[3])
        if self.phase != 'bidding':
            print("Trump Suit:", self.trump_suit)
        if self.phase == 'bidding':
            print("Bidding History:", self.bidding_history)