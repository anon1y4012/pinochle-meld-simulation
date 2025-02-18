import config

def evaluate(hand):
    points = 0
    for name, meld_info in config.meldHands.items():
        cards, value, occurrences = meld_info
        if all(hand.count(card) >= occurrences for card in cards):
            points += value  # Ensure value is correctly accessed
    return points