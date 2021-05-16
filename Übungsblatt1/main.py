import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import argparse
import json
from sklearn.linear_model import LinearRegression

"""
Übungsblatt1
"""


def convert_param_to_list(urliste):
    urliste = json.loads(urliste)
    urliste = np.array(urliste)
    return urliste


def plot_diagramms(urliste):
    # x = np.array([1,2,3,4,5])
    # y = np.array([3,1,3,1,2])
    # y2 = np.array([3,4,7,8,10])
    urliste = convert_param_to_list(urliste)

    x, y = np.unique(urliste, return_counts=True) # [1,1,2,2,1,5]  --> x= [1,2,3,4,5]; y= [3,2,1]

    y2 = np.cumsum(y) # Cumulated list [2,3,5,2] --> [2,5,10,12]

    sorted_urliste = np.sort(urliste) # Sort list [2,2,1,1,4,] --> [1,1,2,2,4]
    print("Sortierte Urliste: \n",
          sorted_urliste)

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
    urliste = convert_param_to_list(urliste)

    a = np.mean(urliste) # Mittelwert
    b = np.median(urliste) # Median
    c = stats.mode(urliste) # Modalwert
    d = np.var(urliste, ddof=1) # Varianz
    e = np.std(urliste, ddof=1) # Standardabweichung
    q75, q25 = np.percentile(urliste, [75, 25]) # Quantile
    q66, q30 = np.percentile(urliste, [66, 30]) # Quantile
    q90 = np.percentile(urliste, 90) # Quantile
    q10 = np.percentile(urliste, 10) # Quantile
    iqr = q75 - q25 # Interquartilabstand
    smax = np.max(urliste)
    smin = np.min(urliste)
    spannweite = smax - smin # Spannweite

    print("Arithmetisches Mittel: ", a, "\n\n",
          "Median               : ", b, "\n\n",
          "Modalwert            : ", c, "\n\n",
          "Varianz              : ", d, "\n\n",
          "Standardabweichung   : ", e, "\n\n",
          "90% Quantil          : ", q90, "\n\n",
          "75% Quantil          : ", q75, "\n\n",
          "66% Quantil          : ", q66, "\n\n",
          "30% Quantil          : ", q30, "\n\n",
          "25% Quantil          : ", q25, "\n\n",
          "10% Quantil          : ", q10, "\n\n",
          "Interquartilabstand  : ", iqr, "\n\n",
          "Spannweite           : ", spannweite, "\n\n", )


def calculate_coefficient(cof1, cof2):
    cof1 = convert_param_to_list(cof1)
    cof2 = convert_param_to_list(cof2)

    print("Korrelationskoeffizient: \n",
          np.corrcoef(cof1,cof2)[0,1])

    """
        Punktediagramm
    """
    plt.plot(cof1, cof2, 'bo')

    """
        Regressionsgerade
    """
    cof1 = cof1.reshape(-1, 1)
    cof2 = cof2.reshape(-1, 1)

    linear_regressor = LinearRegression()
    linear_regressor.fit(cof1, cof2)  # perform linear regression
    Y_pred = linear_regressor.predict(cof1)  # make predictions
    plt.plot(cof1, Y_pred, color='red')
    plt.show()


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
