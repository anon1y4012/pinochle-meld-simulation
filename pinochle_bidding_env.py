import gym
from gym import spaces
import numpy as np
import random
from deck import Deck
from hand_evaluator import evaluate
import config
from itertools import combinations

def card_rank(card):
    order = {'9': 0, 'J': 1, 'Q': 2, 'K': 3, '10': 4, 'A': 5}
    for r in ['10', '9', 'J', 'Q', 'K', 'A']:
        if card.startswith(r):
            return order[r]
    return 0

class PinochleBiddingEnv(gym.Env):
    """
    Pinochle environment with sequential bidding, passing, and trick-taking.

    Bidding:
      - Starts at 20 and can go up to 50.
      - ML Bidder (Player 1) uses its action.
      - Non‑ML players bid randomly: with 50% chance they pass (bid 0) or, if current_bid < 50, choose a random bid between (current_bid+1) and 50.
      - Teammates (Players 1 and 3) won’t bid against each other if they’re the only ones left.
    
    Passing:
      - If ML wins the bid, Player 3 passes 3 cards to Player 1, then Player 1 returns 3 cards (heuristically).
      - An oracle function (compute_optimal_pass) evaluates all legal passes and gives an auxiliary reward shaping signal.
    
    Trick-taking:
      - Follows forced-play rules (when following suit and no trump has been played, if a player holds any card in the led suit that can beat the current highest card, they must play the lowest such card).
    
    Reward:
      - In the trick phase, main_reward is computed as before plus an auxiliary term.
      - The auxiliary term is now computed via a fast MCTS-based lookahead (compute_optimal_value) that runs a limited number of random playouts.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PinochleBiddingEnv, self).__init__()
        self.deck = Deck()
        self.combinations = list(combinations(range(12), 3))
        self.action_space = spaces.Discrete(51)  # 0 = pass; 1..50 = bid amount
        self.observation_space = spaces.Dict({
            'phase': spaces.Discrete(3),  # 0: bidding, 1: passing, 2: trick-taking
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
        self.hands = self.deck.deal(4)

    def init_bidding(self):
        self.phase = 'bidding'
        self.current_bid = 20
        self.active = [True, True, True, True]
        self.current_turn = 0
        self.bidding_history = []

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
            'hand': self.encode_hand(self.hands[0]),
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
            if self.current_turn == 0:
                # ML Bidder uses its action.
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
                # Non-ML players bid randomly.
                if self.current_turn == 2 and self.active[0] and self.active[2] and (not self.active[1] and not self.active[3]):
                    bid = 0
                else:
                    if self.current_bid >= 50:
                        bid = 0
                    else:
                        if random.random() < 0.5:
                            bid = 0
                        else:
                            bid = random.randint(self.current_bid + 1, 50)
                self.bidding_history.append((self.current_turn, bid))
                if bid == 0:
                    self.active[self.current_turn] = False
                else:
                    self.current_bid = bid
                self.current_turn = (self.current_turn + 1) % 4

            if sum(self.active) == 1:
                winner = self.active.index(True)
                self.winning_bid = 20  # Force winning bid to be 20.
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
            # Passing phase: Agent selects a 3-card combination to pass from partner's hand.
            action = action % len(self.combinations)
            chosen_combo = self.combinations[action]
            partner_sorted = sorted(self.hands[2])
            passed_cards = [partner_sorted[i] for i in chosen_combo]
            for card in passed_cards:
                self.hands[2].remove(card)
            self.hands[0].extend(passed_cards)
            self.hands[0].sort(key=lambda c: self.card_rank(c))
            passed_back = self.hands[0][:3]
            self.hands[0] = self.hands[0][3:]
            self.hands[2].extend(passed_back)
            # Oracle for passing.
            optimal_combo, optimal_value = self.compute_optimal_pass()
            agent_ml_total, _, _, _ = self.simulate_round()
            pass_reward = -abs(optimal_value - agent_ml_total)
            self.phase = 'trick'
            return self.get_trick_observation(), pass_reward, False, {
                "passed_cards": passed_cards,
                "passed_back": passed_back,
                "optimal_pass": optimal_combo,
                "optimal_value": optimal_value
            }

        elif self.phase == 'trick':
            optimal_value = self.compute_optimal_value()
            ml_total, opp_total, ml_trick_points, opp_trick_points = self.simulate_round()
            if self.winning_team == 'ML':
                if ml_total >= self.winning_bid:
                    main_reward = (ml_trick_points - opp_trick_points) + self.winning_bid
                else:
                    main_reward = -self.winning_bid
            else:
                main_reward = ml_total
            auxiliary_reward = -abs(optimal_value - ml_total)
            alpha = 0.1
            reward = main_reward + alpha * auxiliary_reward
            done = True
            return self.get_trick_observation(), reward, done, {
                "ml_total": ml_total,
                "opp_total": opp_total,
                "winning_bid": self.winning_bid,
                "winning_team": self.winning_team,
                "ml_trick_points": ml_trick_points,
                "opp_trick_points": opp_trick_points,
                "optimal_value": optimal_value
            }

    def select_trump(self, hand):
        counts = {s: 0 for s in config.suits}
        for card in hand:
            counts[card[-1]] += 1
        return max(counts, key=counts.get)

    def compute_optimal_pass(self):
        best_value = -float('inf')
        best_combo = None
        original_partner = list(self.hands[2])
        for combo in self.combinations:
            if max(combo) >= len(original_partner):
                continue
            partner_sorted = sorted(original_partner)
            chosen_cards = [partner_sorted[i] for i in combo]
            partner_after = list(original_partner)
            for card in chosen_cards:
                partner_after.remove(card)
            bidder_after = list(self.hands[0])
            bidder_after.extend(chosen_cards)
            bidder_after.sort(key=lambda c: self.card_rank(c))
            returned = bidder_after[:3]
            bidder_after = bidder_after[3:]
            partner_after.extend(returned)
            saved_bidder = self.hands[0]
            saved_partner = self.hands[2]
            self.hands[0] = bidder_after
            self.hands[2] = partner_after
            ml_total, _, _, _ = self.simulate_round()
            self.hands[0] = saved_bidder
            self.hands[2] = saved_partner
            if ml_total > best_value:
                best_value = ml_total
                best_combo = combo
        return best_combo, best_value

    def compute_optimal_value(self):
        # Use a limited-iteration MCTS to approximate optimal trick outcome.
        state = self._get_current_state()
        return self.mcts_optimal_value(state, iterations=50)

    def _get_current_state(self):
        return {
            'hands': [list(h) for h in self.hands],
            'leader': self.winning_player,
            'current_trick': [],
            'trick_number': 0,
            'trump': self.trump_suit
        }

    def mcts_optimal_value(self, state, iterations=50):
        total = 0
        for _ in range(iterations):
            total += self._simulate_random_playout(state)
        return total / iterations

    def _simulate_random_playout(self, state):
        current_state = self._copy_state(state)
        total_score = 0
        while not self._is_terminal(current_state):
            player = (current_state['leader'] + len(current_state['current_trick'])) % 4
            legal_moves = self._get_legal_moves(current_state, player)
            move = random.choice(legal_moves) if legal_moves else None
            current_state, score = self._simulate_move(current_state, player, move)
            total_score += score
        return total_score

    def _copy_state(self, state):
        return {
            'hands': [list(h) for h in state['hands']],
            'leader': state['leader'],
            'current_trick': list(state['current_trick']),
            'trick_number': state['trick_number'],
            'trump': state['trump']
        }

    def _is_terminal(self, state):
        return state['trick_number'] == 12

    def _get_legal_moves(self, state, player):
        hand = state['hands'][player]
        if len(state['current_trick']) == 0:
            return hand
        led_suit = state['current_trick'][0][1][-1]
        candidates = [c for c in hand if c[-1] == led_suit]
        if candidates:
            current_cards = [c for (_, c) in state['current_trick'] if c[-1] == led_suit]
            if current_cards:
                current_high = max(current_cards, key=card_rank)
                forced = [c for c in candidates if card_rank(c) > card_rank(current_high)]
                return forced if forced else candidates
            else:
                return candidates
        else:
            return hand

    def _simulate_move(self, state, player, card):
        new_state = self._copy_state(state)
        if card is not None:
            new_state['hands'][player].remove(card)
            new_state['current_trick'].append((player, card))
        score = 0
        if len(new_state['current_trick']) == 4:
            trick = new_state['current_trick']
            led_suit = trick[0][1][-1]
            trump_cards = [(p, c) for p, c in trick if c[-1] == new_state['trump']]
            if trump_cards:
                winner, winning_card = max(trump_cards, key=lambda x: card_rank(x[1]))
            else:
                valid = [(p, c) for p, c in trick if c[-1] == led_suit]
                winner, winning_card = max(valid, key=lambda x: card_rank(x[1]))
            trick_points = sum(1 for (_, c) in trick if c[:-1] in ['K','10','A'])
            if state['trick_number'] == 11:
                trick_points += 1
            score = trick_points if winner in [0, 2] else -trick_points
            new_state['leader'] = winner
            new_state['current_trick'] = []
            new_state['trick_number'] += 1
        return new_state, score

    def simulate_round(self):
        players = [
            {"hand": self.hands[0].copy(), "team": "ML", "label": "Player 1 (ML Bidder)"},
            {"hand": self.hands[1].copy(), "team": "OPP", "label": "Player 2 (Opponent)"},
            {"hand": self.hands[2].copy(), "team": "ML", "label": "Player 3 (ML Partner)"},
            {"hand": self.hands[3].copy(), "team": "OPP", "label": "Player 4 (Opponent)"}
        ]
        team_trick_points = {"ML": 0, "OPP": 0}
        leader = self.winning_player
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
                        trump_played = any(c[-1] == self.trump_suit for (_, c) in trick_cards)
                        if not trump_played:
                            current_follow = [c for (_, c) in trick_cards if c[-1] == led_suit]
                            if current_follow:
                                current_high = max(current_follow, key=self.card_rank)
                                can_beat = [c for c in follow if self.card_rank(c) > self.card_rank(current_high)]
                                card = min(can_beat, key=self.card_rank) if can_beat else min(follow, key=self.card_rank)
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
            trick_points = sum(1 for (_, c) in trick_cards if c[:-1] in ['K','10','A'])
            if trick == 11:
                trick_points += 1
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