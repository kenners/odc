"""
odc.py

An implementation of the Siggaard-Andersen TANH oxyhaemoglobin dissociation curve model in Python.

Siggaard-Andersen O, Siggaard-Andersen M, Fogh-Andersen N. The TANH-equation modified for the hemoglobin, oxygen, and carbon monoxide equilibrium. Scand J Clin Lab Invest Suppl 1993;214:113–9. doi: 10.1080/00365519309090687
http://dx.doi.org/10.1080/00365519309090687
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import scipy.optimize

class ODC(object):
    """
    An oxyhaemoglobin dissociation curve model.
    """
    def __init__(self, pO2, sO2, T=37.0, pH=7.40, pCO2=5.33, cDPG=5.00, ctHb=15.0, FCOHb=0, FMetHb=0, FHbF=0):
        #TODO: Set COHb, MetHb, HbF to 0 rather than 0.005

        # Constants
        self.p0 = 7         # kPa
        self.s0 = 0.867     # Fraction
        self.h0 = 3.5       #
        self.T0 = 37.0      # ºC
        self.k0 = 0.5342857 #
        self.y0 = 1.8747    #
        self.pH0 = 7.40     #
        self.cDPG0 = 5.0    # mmol/L
        self.pCO20 = 5.33   # kPa
        self.p050 = 3.578   # kPa - Estimated normal p50 in human adults
        self.f0 = 1.121     #

        # Chemical allosteric affinity constants
        self.a10 = -0.88    # pH allosteric affinity coefficient
        self.a20 = 0.048    # pCO2 allosteric affinity coefficient
        self.a30 = -0.7     # MetHb allosteric affinity coefficient
        self.a40 = -0.25    # HbF allosteric affinity coefficient
        self.a500 = 0.06
        self.a501 = -0.02
        self.a60 = 0.06     # cDPG allosteric affinity coefficient
        self.a70 = -0.02

        self.dbdT = 0.055
        self.dadcDPG0 = 0.3
        self.dadcDPGxHbF = -0.1

        # Input variables
        self.pO2 = pO2
        self.sO2 = sO2
        self.T = T
        self.pH = pH
        self.pCO2 = pCO2
        self.cDPG = cDPG
        self.FCOHb = FCOHb
        self.FMetHb = FMetHb
        self.FHbF = FHbF
        self.ctHb = ctHb

        # Accuracy of any iterative calculations
        self.epsilon = 1e-6

    # y = y0 + x - x0 + (h * tanh(k0 * (x - x0)))

    def __str__(self):
        # TODO
        pass

    @property
    def p(self):
        """
        pO2CO
        Combined partial pressure of oxygen and carbon monoxide
        """
        return self.pO2 * self.s / (self.s - self.sCO)

    @property
    def s(self):
        """
        sO2CO
        Combined oxygen/carbon monoxide saturation of haemoglobin.
        """
        return self.sO2 + ((1 - self.sO2) * self.sCO)

    @property
    def h(self):
        """
        Positive, homotropic, allosteric ligand interaction factor.
        Varies with changes in affinity
        """
        return self.h0 + self.a

    @property
    def x0(self):
        """
        Combined allosteric affinity factor.
        Sum of chemical and thermal affinity.
        """
        return self.a + self.b

    @property
    def a1(self):
        """pH allosteric affinity factor"""
        return self.a10 * (self.pH - self.pH0)

    @property
    def a2(self):
        """pCO2 allosteric affinity factor"""
        return self.a20 * math.log(self.pCO2 / self.pCO20)

    @property
    def a3(self):
        """MetHb allosteric affinity factor"""
        return self.a30 * self.FMetHb

    @property
    def a4(self):
        """HbF allosteric affinity factor"""
        return self.a40 * self.FHbF

    @property
    def a5(self):
        """DPG allosteric affinity factor"""
        return (self.a500 + (self.a501 * self.FHbF)) * (self.cDPG - self.cDPG0)

    def calculate_a(self, pH, pCO2, FMetHb, FHbF, cDPG):
        a1 = self.a10 * (pH - self.pH0)
        a2 = self.a20 * math.log(pCO2 / self.pCO20)
        a3 = self.a30 * FMetHb
        a4 = self.a40 * FHbF
        a5 = (self.a500 + (self.a501 * FHbF)) * (cDPG - self.cDPG0)
        return a1 + a2 + a3 + a4 + a5

    @property
    def a(self):
        """
        Chemical allosteric affinity factor.
        Sum of factors for pH, pCO2, COHb, MetHb, and HbF
        """
        return self.a1 + self.a2 + self.a3 + self.a4 + self.a5

    @property
    def a_lam(self):
        # TODO: ?remove
        """
        Chemical allosteric affinity factor.
        Calculated from p, s, and T using LambertW function.
        """
        return (-self.k0 * (self.b + self.h0 - self.x + self.y - self.y0) + scipy.special.lambertw(self.k0 * (-self.b + self.h0 + self.x - self.y + self.y0) * math.exp(self.k0 * (-self.b + self.h0 + self.x + self.y - self.y0)))) / (2 * self.k0)

    @property
    def b(self):
        """Thermal allosteric affinity factor"""
        return self.dbdT * (self.T - self.T0)

    @property
    def ceHb(self):
        """Concentration of effective haemoglobin"""
        return self.ctHb * (1 - self.FCOHb - self.FMetHb)

    @property
    def sCO(self):
        """Saturation of haemoglobin with carbon monoxide"""
        return self.FCOHb / 1 - self.FMetHb

    @property
    def y(self):
        return math.log(self.s / 1 - self.s)

    @property
    def x(self):
        return math.log(self.p / self.p0)

    def calc_x(self, p, a, T):
        # From OSA pascal
        """Calculates x for p, a, and T"""
        return math.log(p / self.p0) - a - (self.dbdT * (T - self.T0))

    def calc_y(self, p, a, T):
        # From OSA pascal
        """Calculates y for p, a, and T"""
        x = self.calc_x(p, a, T)
        h = self.calc_h(a)
        return self.y0 + x + (h * math.tanh(self.k0 * x))

    def calc_h(self, a):
        """Calculates h for a"""
        return self.h0 + a

    def calc_dydx(self, p, a, T):
        # From OSA pascal
        return 1 + self.calc_h(a) * self.k0 * (1 - (math.tanh(self.k0 * self.calc_x(p, a, T)))**2)

    def calc_dyda(self, p, a, T):
        # From OSA pascal
        return math.tanh(self.k0 * self.calc_x(p, a, T)) - self.calc_dydx(p, a, T)

    @property
    def a_est(self):
        # From 1990 model
        """
        Estimates 'a' from sO2 and pO2 using IFCC 1990 Guidelines.
        Assumes no dyshaemoglobins (MetHb, COHb, HbF).
        Inaccurate if sO2 > 0.97.
        """
        if self.sO2 > 0.97:
            raise ValueError('sO2 > 0.97')
        x = math.log(self.pO2 / self.p0)
        y = math.log(self.sO2 / (1 - self.sO2)) - self.y0
        t = math.tanh(self.k0 * x)
        return (y - x - (self.h0 * t)) * ((1.87 * t**2) + t - 2.87)**-1

    @property
    def p50_est(self):
        # From 1990 model
        """
        Estimate of p50 from sO2 and pO2 using IFCC 1990 Guidelines.
        Assumes no dyshaemoglobins (MetHb, COHb, HbF).
        Inaccurate if sO2 > 0.97.
        """
        return self.p050 * math.exp(self.f0 * self.a_est)

    @property
    def cDPG_est(self):
        # From OSA pascal
        """Estimates cDPG from other variables"""
        aDPG0 = self.calculate_a(self.pH, self.pCO2, self.FMetHb, self.FHbF, self.cDPG0)
        a = aDPG0
        sO2CO = self.s
        pO2CO = self.p
        ym = self._logit(sO2CO)
        yc = self.calc_y(pO2CO, a, self.T)
        while (abs(ym - yc) > self.epsilon) or (a < -self.h0):
            yc = self.calc_y(pO2CO, a, self.T)
            if abs(ym - yc) > 2:
                a = a + (0.5 * (ym - yc) / self.calc_dyda(pO2CO, a, self.T))
            else:
                a = a + (ym - yc) / self.calc_dyda(pO2CO, a, self.T)
        if a < -self.h0:
            raise ValueError('Unable to calculate cDPG')
            #return self.cDPG0
        else:
            return self.cDPG0 * (1 + ((a - aDPG0) / (self.dadcDPG0 + (self.dadcDPGxHbF * self.FHbF))))

    @property
    def a_iter(self):
        """Calculates 'a' using an iterative approach"""
        # From 1993 tanh paper

        start = self.a # Temporary guess with DPG = 5
        return scipy.optimize.newton(lambda a: self.y0 - self.y + self.x - (a + self.b) + (self.h * math.tanh(self.k0 * (self.x - (a + self.b)))), start)

    def _tanh(self, y0, y, x, a, b, h, k0):
        return y0 - y + x - (a + b) + (h * math.tanh(k0 * (x - (a + b))))

    @property
    def cDPG_from_a_iter(self):
        """Calculates cDPG from iterative value of a"""
        return self.a_iter - self.a1 - self.a2 - self.a3 - self.a4

    @property
    def p50(self):
        """
        p50
        """
        return self.calculate_pO2(0.5)

    def calculate_pO2(self, sO2):
        """Calculates pO2 from sO2"""
        # Requires: sO2, T, FCOHb, FMetHb, FHbF, pH, pCO2, and cDPG.
        # Calculate a and b
        a = self.calculate_a(self.pH, self.pCO2, self.FMetHb, self.FHbF, self.cDPG_est)
        b = self.b
        # Calculate the 'measured' sO2CO (s) from sO2, FCOHb and FMetHb
        #s_m = self.s
        s_m = sO2 + ((1 - sO2) * self.sCO)
        # Make a guess of a temporary pO2CO (p) (preferably choose the
        # point of symmetry of the TANH function) and calculate a temporary
        # sO2CO from the TANH equation
        p_temp = self.p0
        h = self.calc_h(a)
        s_temp = 0
        while abs(s_m - s_temp) > self.epsilon:
            # Calculate temporary sO2CO from TANH equation
            x = math.log(p_temp/self.p0)
            y = self.y0 + x - (a + b) + (h * math.tanh(self.k0 * (x - (a + b))))
            s_temp = math.exp(y)/(math.exp(y) + 1)

            #print(s_temp, s_m - s_temp, p_temp)
            # The difference between the temporary sO2CO and the 'measured'
            # sO2CO allows the calculation of a new temporary pO2CO using a
            # fast Newton-Raphson procedure. Repeat until difference is less
            # than a given limit.
            s_diff = s_m - s_temp
            s_temp2 = s_temp + s_diff
            y = math.log(s_temp2 / (1 - s_temp2))
            # Calculate new p
            top = self.y0 - y + x - (a + b) + (h * math.tanh(self.k0 * (x - (a + b))))
            bottom = (h * self.k0 * (-math.tanh(self.k0 * (x - (a + b)))**2 + 1)) + 1
            x_temp = x - (top / bottom)
            p_temp = self.p0 * math.exp(x_temp)
        p = p_temp
        # Finally M* pCO is calculated (Eq 7) and subtracted from
        # pO2CO to give pO2
        MpCO = (p/s_m) * self.sCO
        pO2 = p - MpCO
        # Round answer to epsilon precision
        precision = abs(round(math.log(self.epsilon, 10))) - 1
        return round(pO2, precision)

    def calculate_sO2(self, pO2):
        """Calculates pO2 from sO2"""
        # Requires: sO2, T, FCOHb, FMetHb, FHbF, pH, pCO2, and cDPG.
        # TODO
        pass

    @property
    def curve_data(self):
        """Tuple of pO2 and sO2 for the curve"""
        sO2 = np.arange(0, 1, 0.01, dtype=np.float64)
        pO2 = np.array([self.calculate_pO2(x) for x in sO2])
        return pO2, sO2


    def plot_curve(self):
        """Matplotlib plot of the oxyhaemoglobin curve"""
        plt.plot(self.curve_data[0], self.curve_data[1], 'r-')
        plt.plot(self.pO2, self.sO2, 'bx')
        plt.xlabel('pO2 (kPa)')
        plt.ylabel('sO2')
        plt.yticks(np.arange(0,1.1,0.1))
        plt.axis(xmax=20)
        #plt.legend(loc='best')
        plt.grid(True)
        return plt

    # Helper functions
    def _logit(self, x):
        return math.log(x / (1 - x))

    def _antilogit(self, x):
        return math.exp(x) / (1 + math.exp(x))
