import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import argparse
import json

"""
Aufgabe 1.2
"""
def plot_diagramms(plot_y,plot_x, plot_y_cum="", plot_urlist=""):
    # x = np.array([1,2,3,4,5])
    plot_x = json.loads(plot_x)
    plot_y = json.loads(plot_y)



    x = np.array(plot_x)
    # y = np.array([3,1,3,1,2])
    y = np.array(plot_y)
    y2 = np.array([3,4,7,8,10])
    # d = np.array([1,1,1,2,3,3,3,4,5,5])


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
    q10 = np.percentile(urliste, 10)
    iqr = q75 - q25
    smax = np.max(urliste)
    smin = np.min(urliste)
    spannweite = smax - smin

    print(" Arithmetisches Mittel: ", a, "\n",
          "Median               : ", b, "\n",
          "Modalwert            : ", c, "\n",
          "Varianz              : ", d, "\n",
          "Standardabweichung   : ", e, "\n",
          "10% Quantil          : ", q10, "\n",
          "Interquartilabstand  : ", iqr, "\n",
          "Spannweite           : ", spannweite, "\n", )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--urliste")
    parser.add_argument("--plot-x", required=False)
    parser.add_argument("--plot-y", required=False)
    # parser.add_argument("--plot-cum-y", required=False)
    # parser.add_argument("--plot-urliste", required=False)
    args = parser.parse_args()
    urliste = args.urliste
    plot_x = args.plot_x
    plot_y = args.plot_y
    # plot_cum = args.plot_y

    # plot_urliste = args.plot_urliste
    characteristics_of_sample(urliste)
    plot_diagramms(plot_y,plot_x)