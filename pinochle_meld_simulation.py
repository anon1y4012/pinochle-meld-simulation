from deck import *
from hand_evaluator import *
import config as config
from counter import *

d = Deck()
#Create counters for different types of points to track
individualPoints = Counter("Individual Points", config.numSimulations)
teamPoints = Counter("Team Points", config.numSimulations)
allPoints = Counter("All Points in a Hand", config.numSimulations)
#Run through number of simulations from configuration
#For each: shuffle, deal, score meld, then add scores to counters
for i in range(config.numSimulations):
    d.shuffle()
    hands = d.deal(4)
    scores = map(evaluate, hands)
    individualPoints.extend(scores)
    teamPoints.extend([scores[0] + scores[2], scores[1] + scores[3]])
    allPoints.extend([sum(scores)])

#Plot all of our data
individualPoints.plot()
teamPoints.plot()
allPoints.plot()