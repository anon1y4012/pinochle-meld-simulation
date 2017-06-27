from __future__ import division
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config as config

#A class to easily count occurrences of values and find basic stats on that data
class Counter:
    def __init__(self, name, numTrials):
        self.name = name
        self.myData = {}
        self.numTrials = numTrials
        #use to make sure we don't double calculate, which is costly
        self.calculated = False
        
    #add a list of data to the counter
    def extend(self, counts):
        for count in counts:
            if not count in self.myData:
                self.myData[count] = 0
            self.myData[count] += 1
    
    #calculate some basic stats on our data. 
    #Since this can be intensive, it will only calculate once       
    def calc(self):
        if not self.calculated:
            self.totalSum = 0
            self.totalCount = 0
            self.maxCount = 0
            self.maxCountValue = 0
            #loop through all counts, building a sum, count of numbers, and keeping track of mode
            for number, count in self.myData.iteritems():
                self.totalSum += number*count
                self.totalCount += count
                if count > self.maxCount:
                    self.maxCount = count
                    self.maxCountValue = number
            self.calculated = True

    #find the minimum value in our data
    def min(self):
        return min(self.myData.keys())
        
    #find the maximum value in our data
    def max(self):
        return max(self.myData.keys())
        
    #find the mean 
    def mean(self):
        return self.totalSum / self.totalCount
        
    #find the median
    def median(self):
        currentCount = 0
        stoppingPoint = self.totalCount / 2
        #loop through data in order until we reach halfway through
        for number in sorted(self.myData.keys()):
            count = self.myData[number]
            currentCount += count
            if currentCount >= stoppingPoint:
                return number
            
    #find the mode
    def mode(self):
        return self.maxCountValue #return pre-calculated value
        
    #calculate the population standard deviation 
    #TODO: Change this to use standard deviation for a sample
    def standardDeviation(self):
        totalSquares = sum(number*number* count for number, count in self.myData.iteritems())
        mean = self.mean()
        variance = (totalSquares / self.totalCount) - (mean * mean)
        return math.sqrt(variance)
    
    #Plot this data on a bar chart, as configured in config.py
    #image name will be: <counter name>.png
    def plot(self):
        self.calc() #calculate data before display
        (stdDev, mean) = (self.standardDeviation(), self.mean())
        #if configured, remove outliers based on number of standard deviations from the mean
        dataWithoutOutliers = sorted(self.myData.iteritems()) if config.showOutliers else filter(lambda (num, count): num <= mean + config.outliersCutoff*stdDev, sorted(self.myData.iteritems()))
        #X axis is the numbers in our data
        markers =  [num for (num, count) in dataWithoutOutliers]
        #Y axis is the count for each number in our data
        vals =     [count for (num, count) in dataWithoutOutliers]
        matplotlib.rcParams['xtick.labelsize'] = 7  #reduce all x-axis labels
        #create bar chart, then configure it for display
        plt.bar(markers, vals, 1, align="center", linewidth=0) 
        plt.xticks(markers)
        plt.margins(0)
        plt.xlabel(config.xAxisLabel)
        plt.ylabel(config.yAxisLabel)
        plt.title("%s: %s (%s trials)" % (config.title, self.name, self.numTrials))
        #Add text summary of data in upper right
        plt.text(max(markers)*.5, max(vals)*.7, self, fontsize=12, family="monospace")
        #Save chart to file (as configured)
        plt.savefig("%s/%s.png" % (config.outputDirectory, self.name), dpi=200)
        plt.clf() # clear the graph we have been building
        
    def __str__(self):
        self.calc()
        return  """
        Min:        %s
        Max:        %s
        Mean:       %s
        Median:     %s
        Mode:       %s
        Std. Dev.:  %s
        """ % (self.min(), self.max(), round(self.mean(),2), self.median(), self.mode(), round(self.standardDeviation(),2))
