#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm

"""
    File name: results-analysis.py
    Author: German Malsagov
    Last modified: 04/09/2018
    Python Version: 2.7
"""


def plot_stats():
    """This script allows to evaluate the results of our testing
    It provides measures for dispersion and presence of outliers"""
    # Load data
    with open('analysis.txt', 'r') as file:
        lines = file.readlines()
        header = ['Length (words)', 'Inference Delay (ms)']
        length = []
        delay = []
        for x in lines:
            length.append(x.split(',')[0])
            delay.append(x.split(',')[1].rstrip())

        length = map(int, length)
        delay = map(int, delay)

        data_array = list(zip(length, delay))
        # length = np.array(length)
        # delay = np.array(delay)

    # Determine variables' mean and std
    explore_var_means(data_array, header)

    # Determine variables' dispersion
    explore_var_dispersion(delay, header[1])

    # Explore outliers
    explore_data_outliers(delay, header[1])


def explore_var_means(data_array, header):
    """Explores mean of the variables"""

    fig, ax = plt.subplots()

    # Compute means of the list
    mean_array = []
    for i in range(2):
        data_list = list(data_array[i][:])
        mean = np.mean(data_list)
        mean_array.append(mean)

    #store the variable names in the objects array
    objects = []
    for i in range(len(header)):
        objects.append(header[i])

    print(objects)
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, mean_array, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation=90)

    plt.ylabel('Mean value')
    plt.yticks(np.arange(0, max(mean_array)+15, 5))
    # create a list to collect the plt.patches data
    totals = []
    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_height())
    # set individual bar lables using above list
    total = sum(totals)
    # set individual bar lables using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x()+.17, i.get_height()+10, \
                str(round((i.get_height()/total)*100, 2)), fontsize=12,
                    color='dimgrey', rotation=90)

    plt.title('Measure of Central Tendency: variable means')
    fig.savefig('Measure of Central Tendency - variable means.pdf', bbox_inches = 'tight')
    # plt.show()


def explore_var_dispersion(data_array, header):
    """Explores dispersion of variables"""

    fig2, ax = plt.subplots()

    # Compute mean std
    (mu, sigma) = norm.fit(data_array[:])

    # Histogram data
    n, bins, patches = plt.hist(data_array[:], 20, facecolor='darkblue', alpha=0.75)

    # Add a trend line
    y = mlab.normpdf( bins, mu, sigma)
    plt.plot(bins,y, 'r--', linewidth=1)

    varname = header

    # Plot histogram
    plt.xlabel(varname)
    plt.ylabel('Frequency')
    plt.title(r'$\mathrm{Histogram\ of\ %s - }\ \mu=%.2f,\ \sigma=%.2f$' %(varname,mu, sigma))
    fig2.savefig('Measure of Dispersion - %s .pdf' %(varname), bbox_inches = 'tight')
    plt.show()


def explore_data_outliers(data_array, header):
    """Explores data for potential outliers"""

    fig, ax = plt.subplots()

    # Using log scale to fit data into one boxplot
    data = []
    for i in range(2):
        data.append(data_array[:][i])

    plt.ylim((5,14))
    ax.boxplot(data_array, sym='.')
    plt.xlabel('Variables')
    plt.ylabel('log(ms)')
    plt.xticks(range(len(header)), (' ','delay'), rotation=0)
    fig.savefig('Outliers.pdf', bbox_inches = 'tight')
    plt.show()


if __name__ == '__main__':
    plot_stats()
