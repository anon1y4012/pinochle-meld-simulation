numSimulations = 1000000

#deck configuration - card values, suits, and number of occurrences of each
values = ['9', '10', 'J', 'Q', 'K', 'A']
suits = ['C', 'D', 'H', 'S']
occurrences = 2

meldHands = {
    #Name: ([card list], points, occurrences required for each card)
    'Round Of Aces': (['AC','AD','AH','AS'], 10, 1),
    'Round Of Kings': (['KC','KD','KH','KS'], 8, 1),
    'Round Of Queens': (['QC','QD','QH','QS'], 6, 1),
    'Round Of Jacks': (['JC','JD','JH','JS'], 4, 1),
    'All Aces': (['AC','AD','AH','AS'], 90, 2),
    'All Kings': (['KC','KD','KH','KS'], 72, 2),
    'All Queens': (['QC','QD','QH','QS'], 54, 2),
    'All Jacks': (['JC','JD','JH','JS'], 36, 2),
    'Marriage Clubs': (['KC', 'QC'], 2, 1),
    'Double Marriage Clubs': (['KC', 'QC'], 2, 2),
    'Marriage Diamonds': (['KD', 'QD'], 2, 1),
    'Double Marriage Diamonds': (['KD', 'QD'], 2, 2),
    'Marriage Hearts': (['KH', 'QH'], 2, 1),
    'Double Marriage Hearts': (['KH', 'QH'], 2, 2),
    'Marriage Spades': (['KS', 'QS'], 2, 1),
    'Double Marriage Spades': (['KS', 'QS'], 2, 2),
    'Run in Clubs': (['JC','QC','KC','10C','AC'], 15, 1),
    'Run in Diamonds': (['JD','QD','KD','10D','AD'], 15, 1),
    'Run in Hearts': (['JH','QH','KH','10H','AH'], 15, 1),
    'Run in Spades': (['JS','QS','KS','10S','AS'], 15, 1),
    'Double Run in Clubs': (['JC','QC','KC','10C','AC'], 135, 2),
    'Double Run in Diamonds': (['JD','QD','KD','10D','AD'], 135, 2),
    'Double Run in Hearts': (['JH','QH','KH','10H','AH'], 135, 2),
    'Double Run in Spades': (['JS','QS','KS','10S','AS'], 135, 2),
    'Pinochle': (['QS','JD'], 4, 1),
    'Double Pinochle': (['QS','JD'], 26, 2)
}


#graphing config
showOutliers = False
outliersCutoff = 5 # number of standard deviations from the mean to cut off beyond
outputDirectory = "output"
xAxisLabel="Meld Score"
yAxisLabel="Number of Occurrences"
title="Distribution of Meld"