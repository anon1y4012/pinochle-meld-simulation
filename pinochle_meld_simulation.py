from deck import Deck
from hand_evaluator import evaluate
import config
from counter import Counter

d = Deck()

# Create counters
individualPoints = Counter("Individual Points", config.numSimulations)
teamPoints = Counter("Team Points", config.numSimulations)
allPoints = Counter("All Points in a Hand", config.numSimulations)

# Run simulations
for _ in range(config.numSimulations):
    d.shuffle()
    hands = d.deal(4)
    scores = list(map(evaluate, hands))
    individualPoints.extend(scores)
    teamPoints.extend([scores[0] + scores[2], scores[1] + scores[3]])
    allPoints.extend([sum(scores)])

# Plot results
individualPoints.plot()
teamPoints.plot()
allPoints.plot()