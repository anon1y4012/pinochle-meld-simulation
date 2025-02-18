import config

def evaluate(hand):
    points = 0
    for name, (cards, value, occurrences) in config.meldHands.items():
        if all(hand.count(card) >= occurrences for card in cards):
            points += value
    return points