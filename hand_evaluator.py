import config as config

# Determine the meld points for the given hand
def evaluate(hand):
    points = 0
    #loop through all meld possibilities from configuration
    for name, (cards, value, occurrences) in config.meldHands.iteritems():
        #if all cards are in this hand the required number of times then award the points
        if all(hand.count(card) >= occurrences for card in cards):
            points += value
    return points