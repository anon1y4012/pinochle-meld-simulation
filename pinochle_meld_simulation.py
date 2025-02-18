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

# Print hard data for individual, team, and total scores
def print_statistics(counter, label):
    print(f"\n{label} Statistics:")
    print(f"  Min: {counter.min()}")
    print(f"  Max: {counter.max()}")
    print(f"  Mean: {counter.mean():.2f}")
    print(f"  Median: {counter.median()}")
    print(f"  Mode: {counter.mode()}")
    print(f"  Standard Deviation: {counter.standardDeviation():.2f}")

# Display statistics in Jupyter Notebook
print_statistics(individualPoints, "Individual Points")
print_statistics(teamPoints, "Team Points")
print_statistics(allPoints, "Total Hand Points")

# Plot results
individualPoints.plot()
teamPoints.plot()
allPoints.plot()