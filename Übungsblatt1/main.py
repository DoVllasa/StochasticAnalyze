import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import argparse
import json
from sklearn.linear_model import LinearRegression
import math

"""
    Übungsblatt2
"""

class Uebungsblatt2():

    def __init__(self, n, k, repetition=False):
            self.n = int(n)
            self.k = int(k)
            self.repetition = repetition

    def combinatoric(self):
        """
            Matematical combinatoric:
            C(n,k) = n! / k!*(n-k)! --> Without repetition # 1
            C'(n,k) = (n + k -1)! / (n - 1)! * k! --> With repitition # 2
        """

        if self.repetition:
            comb_result = math.factorial(self.n + self.k -1) / math.factorial(self.n - 1) * math.factorial(self.k)  # 2
            print("C'(n,k) = (n + k -1)! / (n - 1)! * k! --> With repitition: ", comb_result)
        else:
            comb_result = math.factorial(self.n) / math.factorial(self.k) * math.factorial(self.n - self.k) # 1
            print("C(n,k) = n! / k!*(n-k)! --> Without repetition: ", comb_result)

    def variation(self):
        """
            Mathematical variation:
            V(n,k) = n!/(n-k)! --> without repitition # 1
            V'(n,k) = n^k # 2 --> With repitition # 2
        """

        if self.repetition:
            var_result = self.n ** self.k # 2
            print('V(n,k) = n!/(n-k)! --> without repitition: ', var_result)
        else:
            var_result = math.factorial(self.n) / math.factorial(self.n - self.k) # 1
            print("V'(n,k) = n^k --> With repitition: ", var_result)

    def permutation(self):
        """
            Mathematical permutation:
            P(n) = n!
        """
        permutation_result = math.factorial(self.n)
        print(permutation_result)


"""
    Übungsblatt1
"""

class Uebungsblatt1():

    def __init__(self, urliste):
            self.urliste = urliste

    def convert_param_to_list(self, cof1=None, cof2=None):
        if self.urliste is not None:
            urliste = json.loads(self.urliste)
            urliste = np.array(urliste)
            return urliste
        if (cof1 and cof2) is not None:
            cof1 = json.loads(cof1)
            cof2 = json.loads(cof2)
            cof1 = np.array(cof1)
            cof2 = np.array(cof2)
            return cof1, cof2

    def plot_diagramms(self):
        # x = np.array([1,2,3,4,5])
        # y = np.array([3,1,3,1,2])
        # y2 = np.array([3,4,7,8,10])
        urliste = self.convert_param_to_list()
        x, y = np.unique(urliste, return_counts=True) # [1,1,2,2,1,5]  --> x= [1,2,3,4,5]; y= [3,2,1]
        y2 = np.cumsum(y)                             # Cumulated list [2,3,5,2] --> [2,5,10,12]
        sorted_urliste = np.sort(urliste)             # Sort list [2,2,1,1,4,] --> [1,1,2,2,4]
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


    def characteristics_of_sample(self):
        urliste = self.convert_param_to_list()

        a = np.mean(urliste)                        # Mittelwert
        b = np.median(urliste)                      # Median
        c = stats.mode(urliste)                     # Modalwert
        d = np.var(urliste, ddof=1)                 # Varianz
        e = np.std(urliste, ddof=1)                 # Standardabweichung
        q75, q25 = np.percentile(urliste, [75, 25]) # Quantile
        q66, q30 = np.percentile(urliste, [66, 30]) # Quantile
        q90 = np.percentile(urliste, 90)            # Quantile
        q10 = np.percentile(urliste, 10)            # Quantile
        iqr = q75 - q25                             # Interquartilabstand
        smax = np.max(urliste)
        smin = np.min(urliste)
        spannweite = smax - smin                    # Spannweite

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


    def calculate_coefficient(self, cof1, cof2):
        cof1, cof2 = self.convert_param_to_list(cof1, cof2)

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
        linear_regressor.fit(cof1, cof2)         # perform linear regression
        Y_pred = linear_regressor.predict(cof1)  # make predictions
        plt.plot(cof1, Y_pred, color='red')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--urliste")
    parser.add_argument("--cof1")
    parser.add_argument("--cof2")
    parser.add_argument("--plot", type=bool)
    parser.add_argument("--n")
    parser.add_argument("--k")
    parser.add_argument('--rep', action='store_true')
    parser.add_argument('--no-rep', action='store_false')
    parser.add_argument("--comb", action='store_true')
    parser.add_argument("--var", action='store_true')
    parser.add_argument("--per", action='store_true')

    args = parser.parse_args()
    urliste = args.urliste
    cof1 = args.cof1
    cof2 = args.cof2
    plot = args.plot
    n = args.n
    k = args.k
    rep = args.rep
    no_rep = args.no_rep
    comb = args.comb
    var = args.var
    per = args.per

    """
           Characterization of samples
    """
    if cof1 and cof2:
        urliste_uebung1 = Uebungsblatt1(urliste)
        urliste_uebung1.calculate_coefficient(cof1, cof2)
    if urliste:
        urliste_uebung1 = Uebungsblatt1(urliste)
        urliste_uebung1.characteristics_of_sample()
    if plot == True:
        urliste_uebung1 = Uebungsblatt1(urliste)
        urliste_uebung1.plot_diagramms()

    """
        Korrelationskoeffizienten
    """
    if cof1 and cof2:
        urliste_uebung1 = Uebungsblatt1(urliste)
        urliste_uebung1.calculate_coefficient(cof1,cof2)

    """
        Variation, Combination and permutation
    """
    if n and k:
        if comb:
            if rep:
                uebungsblatt2 = Uebungsblatt2(n, k, repetition=rep)
            else:
                uebungsblatt2 = Uebungsblatt2(n, k, repetition=no_rep)
            uebungsblatt2.combinatoric()
        if var:
            if rep:
                uebungsblatt2 = Uebungsblatt2(n, k, repetition=rep)
            else:
                uebungsblatt2 = Uebungsblatt2(n, k, repetition=no_rep)
            uebungsblatt2.variation()
        if per:
            uebungsblatt2 = Uebungsblatt2(n, 0, repetition=rep)
            uebungsblatt2.permutation()

