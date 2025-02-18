import math
import matplotlib.pyplot as plt
import config

class Counter:
    def __init__(self, name, numTrials):
        self.name = name
        self.myData = {}
        self.numTrials = numTrials
        self.calculated = False  # To avoid unnecessary recalculations

    def extend(self, counts):
        for count in counts:
            self.myData[count] = self.myData.get(count, 0) + 1

    def calc(self):
        if not self.calculated:
            self.totalSum = sum(number * count for number, count in self.myData.items())
            self.totalCount = sum(self.myData.values())
            self.maxCountValue = max(self.myData, key=self.myData.get)
            self.calculated = True

    def min(self):
        return min(self.myData.keys())

    def max(self):
        return max(self.myData.keys())

    def mean(self):
        self.calc()
        return self.totalSum / self.totalCount

    def median(self):
        self.calc()
        sorted_keys = sorted(self.myData.keys())
        mid = self.totalCount // 2
        running_count = 0
        for key in sorted_keys:
            running_count += self.myData[key]
            if running_count >= mid:
                return key

    def mode(self):
        self.calc()
        return self.maxCountValue

    def standardDeviation(self):
        self.calc()
        mean = self.mean()
        variance = sum((num - mean) ** 2 * count for num, count in self.myData.items()) / self.totalCount
        return math.sqrt(variance)

    def plot(self):
        self.calc()
        markers = list(self.myData.keys())
        vals = list(self.myData.values())

        plt.bar(markers, vals, align="center", linewidth=0)
        plt.xlabel(config.xAxisLabel)
        plt.ylabel(config.yAxisLabel)
        plt.title(f"{config.title}: {self.name} ({self.numTrials} trials)")

        plt.savefig(f"{config.outputDirectory}/{self.name}.png", dpi=200)
        plt.clf()