import numpy as np
import pywt 
from PyEMD import EMD,EEMD,CEEMDAN


def Walvet_transformation(signal,lev):
    signal = np.sin(np.linspace(0, 10, 1000))
    coeffs = pywt.wavedec(signal, 'db1', level=lev)
    return coeffs

def Ceemdan(signal):
    ceemdan = CEEMDAN()
    return ceemdan(signal)

def Emd(signal):
    emd = EMD()
    return emd(signal)
def Eemd(signal):
    eemd = EEMD()
    return eemd(signal)
