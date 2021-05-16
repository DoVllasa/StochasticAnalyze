import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import argparse
import json

from sklearn.linear_model import LinearRegression

"""
Aufgabe 1.2
"""
def plot_diagramms(urliste):
    # x = np.array([1,2,3,4,5])
    # y = np.array([3,1,3,1,2])
    # y2 = np.array([3,4,7,8,10])
    urliste = json.loads(urliste)
    x, y = np.unique(urliste, return_counts=True)

    y2 = np.cumsum(y)
    d = np.array(urliste)
    d = np.sort(d)
    print(d)

    """
        Plot Histogram
    """
    plt.bar(x,y)
    plt.show()

    """
        Plot Empirical cdf
    """
    plt.step(x,y2)
    plt.show()


def characteristics_of_sample(urliste):
    urliste = json.loads(urliste)
    urliste = np.array(urliste)
    a = np.mean(urliste)
    b = np.median(urliste)
    c = stats.mode(urliste)
    d = np.var(urliste, ddof=1)
    e = np.std(urliste, ddof=1)
    q75, q25 = np.percentile(urliste, [75, 25])
    q66, q30 = np.percentile(urliste, [66, 30])
    q90 = np.percentile(urliste, 90)
    q10 = np.percentile(urliste, 10)
    iqr = q75 - q25
    smax = np.max(urliste)
    smin = np.min(urliste)
    spannweite = smax - smin

    print("### Befehl - Arithmetisches Mittel: urliste = json.loads(urliste);  urliste = np.array(urliste);  arithmetisches_mittel = np.mean(urliste) ### \n",
          "Arithmetisches Mittel: ", a, "\n\n",

          "### Befehl - Media: urliste = json.loads(urliste);  urliste = np.array(urliste);  Median = np.median(urliste) ### \n"
          "Median               : ", b, "\n\n",

          "### Befehl - Modalwert: urliste = json.loads(urliste);  urliste = np.array(urliste);  Modalwert = np.mode(urliste) ### \n"
          "Modalwert            : ", c, "\n\n",
          "### Befehl - Varianz: urliste = json.loads(urliste);  urliste = np.array(urliste);  varianz = np.var(urliste, ddof=1) ###  \n"

          "Varianz              : ", d, "\n\n",
          "### Befehl - Standardabweichung: urliste = json.loads(urliste);  urliste = np.array(urliste);  Standarabweichung = np.std(urliste, ddof=1) ### \n"
          "Standardabweichung   : ", e, "\n\n",

          "### Befehl - 90% Quantil: urliste = json.loads(urliste);  urliste = np.array(urliste);  Quantil_90 = np.std(urliste, ddof=1) ### \n"
          "90% Quantil          : ", q90, "\n\n",

          "### Befehl - 75% Quantil: urliste = json.loads(urliste);  urliste = np.array(urliste);  Quantil_75 = np.std(urliste, ddof=1) ### \n"
          "75% Quantil          : ", q75, "\n\n",

          "### Befehl - 66% Quantil: urliste = json.loads(urliste);  urliste = np.array(urliste);  Quantil_66 = np.std(urliste, ddof=1) ### \n"
          "66% Quantil          : ", q66, "\n\n",

          "### Befehl - 30% Quantil: urliste = json.loads(urliste);  urliste = np.array(urliste);  Quantil_30 = np.std(urliste, ddof=1) ### \n"
          "30% Quantil          : ", q30, "\n\n",

          "### Befehl - 25% Quantil: urliste = json.loads(urliste);  urliste = np.array(urliste);  Quantil_25 = np.std(urliste, ddof=1) ###  \n"
          "25% Quantil          : ", q25, "\n\n",

          "### Befehl - 10% Quantil: urliste = json.loads(urliste);  urliste = np.array(urliste);  Quantil_10 = np.std(urliste, ddof=1) ### \n"
          "10% Quantil          : ", q10, "\n\n",

          "### Befehl - Interquartilabstand: urliste = Interquartilabstand = q75-q25 ### \n"
          "Interquartilabstand  : ", iqr, "\n\n",

          "### Befehl - Spannweite: smax = np.max(urliste); smin = np.min(urliste); spannweite = smax - smin ### \n"
          "Spannweite           : ", spannweite, "\n\n", )



def calculate_coefficient(cof1, cof2):
    cof1 = json.loads(cof1)
    cof1 = np.array(cof1)
    cof2 = json.loads(cof2)
    cof2 = np.array(cof2)

    print(np.corrcoef(cof1,cof2)[0,1])

    plt.plot(cof1, cof2, 'bo')

    cof1 = cof1.reshape(-1, 1)
    cof2 = cof2.reshape(-1, 1)

    linear_regressor = LinearRegression()
    linear_regressor.fit(cof1, cof2)  # perform linear regression
    Y_pred = linear_regressor.predict(cof1)  # make predictions
    plt.plot(cof1, Y_pred, color='red')


    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--urliste")
    parser.add_argument("--cof1")
    parser.add_argument("--cof2")
    parser.add_argument("--plot", type=bool)
    args = parser.parse_args()
    urliste = args.urliste
    cof1 = args.cof1
    cof2 = args.cof2
    plot = args.plot
    if urliste:
        characteristics_of_sample(urliste)
    if plot == True:
        plot_diagramms(urliste)
    if cof1 and cof2:
        calculate_coefficient(cof1,cof2)
