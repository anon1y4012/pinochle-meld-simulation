
# Pinochle Meld Simulation

## What is Pinochle?

Pinochle is a trick-taking game played by my extended family. It has some interesting rules, being played with a deck of cards from 9-Ace with two copies of each. The game is played with four people, two on each team. For each played hand, players are delt equal hands (of 12 cards), after which they determine their "Meld." Meld points are earned for each player by getting different combinations of cards delt to them (e.g. a King and Queen of the same suit earn 2 points). The meld points are added up for each team before playing out the hand. You can learn more about the game at https://en.wikipedia.org/wiki/Pinochle.  

## Meld Simulations
The goal of this program is to analyze and visualize the distribution of meld points as an individual, team, and overall in a hand. This program simulates the dealing of 1 million (by default) hands, scores them, and then figures out the following statistics:
* min
* max
* mean
* median
* mode
* standard deviation

It also creates a histogram of all results (for each category) to visualize the distribution and display the statistics. Currently these histograms filter out outliers to facilitate easier viewing, but this can be turned off in the configuration.


## Development
You must use Python 2.7 to run. You can run with just: `python pinochle_meld_simulation.py`.
Basic configuration, such as number of simulations and meld point values can be changed in config.py.

## Future Work
I'd like to make the following improvements in the future:
* Analyze the affect of my own meld on that of my teammate and opponent. If I have low meld, are they likely to have high meld (and vice versa). This could be very helpful knowledge for the bidding process in Pinochle.
* Create further visualizations of the data. This data is so heavily weighted towards the 0-10 range that other visualizations might be useful as well.
* Convert standard deviation to sample standard deviation. I used the population value for the sake of speed, but this really should be changed.