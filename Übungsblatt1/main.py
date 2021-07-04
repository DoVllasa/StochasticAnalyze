import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
from sklearn.linear_model import LinearRegression
import math
from scipy import stats, special

class Uebungsblatt6():
    def __init__(self, x):
        self.x = float(x)

    def stetige_gleichverteilung(self, a, b, p=0):
        a = float(a)
        b = float(b)
        p = float(p)

        print("Verteilungsdichte: ", stats.uniform(a, b - a).pdf(self.x))
        print("Verteilungsfunktion: ", stats.uniform(a, b - a).cdf(self.x))
        print("Erwartungswert: ", stats.uniform(a, b-a).expect())
        print("Varianz: ", stats.uniform(a, b-a).var())
        print("Standardabweichung: ", math.sqrt(stats.uniform(a, b-a).var()))
        print("p-Quantil: ", stats.uniform(a, b-a).ppf(p))

    def stetige_exponentialverteilung(self, l, p=0):
        l = float(l)
        print("Verteilungsdichte: ", stats.expon(scale=1/l).pdf(self.x))
        print("Verteilungsfunktion: ", stats.expon(scale=1/l).cdf(self.x))
        print("Erwartungswert: ", stats.expon(scale=1/l).expect())
        print("Varianz: ", stats.expon(scale=1/l).var())
        print("Standardabweichung: ", math.sqrt(stats.expon(scale=1/l).var()))
        print("p-Quantil: ", stats.expon(scale=1/l).ppf(p))

    def stetige_normalverteilung(self, mu, sigma, p=0):
        mu = float(mu)
        sigma = float(sigma)

        print("Verteilungsdichte: ", stats.norm(mu, sigma).pdf(self.x))
        print("Verteilungsfunktion: ", stats.norm(mu, sigma).cdf(self.x))
        print("Erwartungswert: ", stats.norm(mu, sigma).expect())
        print("Varianz: ",stats.norm(mu, sigma).var())
        print("Standardabweichung: ", math.sqrt(stats.norm(mu, sigma).var()))
        print("p-Quantil: ", stats.norm(mu, sigma).ppf(p))




class Uebungsblatt5():
    def __init__(self, x):
        self.x = json.loads(x)

    def binom(self, n, p):
        print("Verteilung:")
        n = float(n)
        p = float(p)

        for i in range(len(self.x)):
            print(self.x[i], ": ", stats.binom(n, p).pmf(self.x[i]))

        print("Verteilungsfunktion:")

        for i in range(len(self.x)):
            print(self.x[i], ": ", stats.binom(n, p).cdf(self.x[i]))

        print("Erwartungswert: ", stats.binom(n, p).expect())
        print("Varianz", stats.binom(n, p).var())

    def geom(self, p):
        p = float(p)
        print("Verteilung:")

        for i in range(len(self.x)):
            print(self.x[i], ": ", stats.geom(p).pmf(self.x[i]))

        print("Verteilungsfunktion:")

        for i in range(len(self.x)):
            print(self.x[i], ": ", stats.geom(p).cdf(self.x[i]))

        print("Erwartungswert: ", stats.geom(p).expect())
        print("Varianz", stats.geom(p).var())

    def poisson(self, l):
        l = float(l)
        print("Verteilung:")

        for i in range(len(self.x)):
            print(self.x[i], ": ", stats.poisson(l).pmf(self.x[i]))

        print("Verteilungsfunktion:")

        for i in range(len(self.x)):
            print(self.x[i], ": ", stats.poisson(l).cdf(self.x[i]))

        print("Erwartungswert: ", stats.poisson(l).expect())
        print("Varianz", stats.poisson(l).var())

    def bernoulli(self, p):
        p = float(p)
        print("Verteilung:")

        for i in range(len(self.x)):
            print(self.x[i], ": ", stats.bernoulli(p).pmf(self.x[i]))

        print("Verteilungsfunktion:")

        for i in range(len(self.x)):
            print(self.x[i], ": ", stats.bernoulli(p).cdf(self.x[i]))

        print("Erwartungswert: ", stats.bernoulli(p).expect())
        print("Varianz", stats.bernoulli(p).var())

    def hypergeom(self, N, M, n):
        N = float(N)
        M = float(M)
        n = float(n)
        print("Verteilung:")

        for i in range(len(self.x)):
            print(self.x[i], ": ", stats.hypergeom(N, M, n).pmf(self.x[i]))

        print("Verteilungsfunktion:")

        for i in range(len(self.x)):
            print(self.x[i], ": ", stats.hypergeom(N, M, n).cdf(self.x[i]))

        print("Erwartungswert: ", stats.hypergeom(N, M, n).expect())
        print("Varianz", stats.hypergeom(N, M, n).var())




"""
    Übungsblatt4
"""
class Uebungsblatt4():
    def __init__(self, Xi, Pi):
        self.Xi = json.loads(Xi)
        self.Pi = json.loads(Pi)

    def erwartungswert_zufallsvariablen(self):
        if len(self.Xi) == len(self.Pi):
            erw = 0
            for i, j in zip(self.Xi, self.Pi):
                erw += i * j

            var_ex_hoch2 = 0
            funktion_ex_hoch2 = erw * erw

            for i, j in zip(self.Xi, self.Pi):
                var_ex_hoch2 += (i*i*j)
            var = var_ex_hoch2 - funktion_ex_hoch2

            kovarianz = np.cov(self.Xi, self.Pi)

            standardabweichung = 0
            print(var)
            if var > 0:
                standardabweichung = math.sqrt(var)

            print("Erwartungswert: ", erw, "\n",
                  "Varianz: ", var, "\n",
                  "Standardabweichung: ",standardabweichung , "\n",
                  "Kovarianz: ", kovarianz, "\n")
        else:
            print("Both lists have to be the same length")

"""
    Übungsblatt3
"""
class Uebungsblatt3():
    pass


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
        comb_result = special.comb(self.n, self.k, repetition=self.repetition, exact=True)  # 2

        if self.repetition:
            print("C'(n,k) = (n + k -1)! / (n - 1)! * k! --> With repitition: ", comb_result)
        else:
            print("C(n,k) = n! / k!*(n-k)! --> Without repetition: ", comb_result)

    def variation(self):
        """
            Mathematical variation:
            V(n,k) = n!/(n-k)! --> without repitition # 1
            V'(n,k) = n^k # 2 --> With repitition # 2
        """

        if self.repetition:
            var_result = pow(self.n,self.k) # 2
            print("V'(n,k) = n^k --> With repitition: ", var_result)
        else:
            var_result = math.factorial(self.n) / math.factorial(self.n - self.k) # 1
            print('V(n,k) = n!/(n-k)! --> without repitition: ', var_result)



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
        verteilungsfkt = np.cumsum(urliste)         # Cumulated list [2,3,5,2] --> [2,5,10,12]
        erwartungswert = np.sum(urliste)

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
              "Spannweite           : ", spannweite, "\n\n",
              "Verteilunggsfunktion           : ", verteilungsfkt, "\n\n", )


    def calculate_coefficient(self, cof1, cof2):
        cof1, cof2 = self.convert_param_to_list(cof1, cof2)

        emp_correlationcoefficient = np.corrcoef(cof1,cof2)[0,1]
        emp_covariance = np.cov(cof1, cof2)
        Sx = np.var(cof1, ddof=1)
        Sy = np.var(cof2, ddof=1)

        print("(Empirischer) Korrelationskoeffizient: \n",
              emp_correlationcoefficient)


        print("(Empirische) Kovarianz: \n",
              emp_covariance)

        print("Varianz-cof1 Sx: \n",
              Sx)

        print("Varianz-cof2 Sy: \n",
              Sy)

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
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--n")
    parser.add_argument("--k")
    parser.add_argument('--rep', action='store_true')
    parser.add_argument('--no-rep', action='store_false')
    parser.add_argument("--comb", action='store_true')
    parser.add_argument("--var", action='store_true')
    parser.add_argument("--per", action='store_true')
    parser.add_argument("--Xi")
    parser.add_argument("--Pi")

    parser.add_argument("--Ber",action='store_true')
    parser.add_argument("--geom",action='store_true')
    parser.add_argument("--Bin",action='store_true')
    parser.add_argument("--H",action='store_true')
    parser.add_argument("--Po",action='store_true')

    parser.add_argument("--x")
    parser.add_argument("--nn")
    parser.add_argument("--p")
    parser.add_argument("--l")
    parser.add_argument("--N")
    parser.add_argument("--M")

    parser.add_argument("--gleichver",action='store_true')
    parser.add_argument("--exponentver", action='store_true')
    parser.add_argument("--normalver", action='store_true')
    parser.add_argument("--a")
    parser.add_argument("--b")
    parser.add_argument("--mu")
    parser.add_argument("--sigma")

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
    Xi = args.Xi
    Pi = args.Pi

    Ber = args.Ber
    geom = args.geom
    Bin = args.Bin
    H = args.H
    Po = args.Po

    gleichver = args.gleichver
    exponentver = args.exponentver
    normalver = args.normalver
    a = args.a
    b = args.b
    mu = args.mu
    sigma = args.sigma

    x = args.x
    nn = args.nn
    p = args.p
    l = args.l
    N = args.N
    M = args.M

    """
           Characterization of samples
    """
    if cof1 and cof2:
        urliste_uebung1 = Uebungsblatt1(urliste)
        urliste_uebung1.calculate_coefficient(cof1, cof2)
    if urliste and plot:
        urliste_uebung1 = Uebungsblatt1(urliste)
        urliste_uebung1.characteristics_of_sample()
        urliste_uebung1.plot_diagramms()
    if urliste:
        urliste_uebung1 = Uebungsblatt1(urliste)
        urliste_uebung1.characteristics_of_sample()

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

    """
        Berechnung des Erwartungswertes
    """

    if Xi and Pi:
        berechnung_characteristica_zv = Uebungsblatt4(Xi,Pi)
        berechnung_characteristica_zv.erwartungswert_zufallsvariablen()


    """
        Wichtige dieskrete Verteilungen
    """
    if x:
        uebung5 = Uebungsblatt5(x)
        if Bin and nn and p:
            uebung5.binom(nn, p)
        if geom and p:
            uebung5.geom(p)
        if Po and l:
            uebung5.poisson(l)
        if Ber and p:
            uebung5.bernoulli(p)
        if H and N and M and nn:
            uebung5.hypergeom(N, M, nn)

    """
        Kontinuierliche Wahrscheinlichkeitstheorie
        - Wichtige stetige Verteilungen
    """

    if gleichver and x and a and b:
        uebung6 = Uebungsblatt6(x)
        uebung6.stetige_gleichverteilung(a,b)
    if exponentver and x and l:
        uebung6 = Uebungsblatt6(x)
        uebung6.stetige_exponentialverteilung(l)
    if normalver and x and mu and sigma:
        uebung6 = Uebungsblatt6(x)
        uebung6.stetige_normalverteilung(mu, sigma)