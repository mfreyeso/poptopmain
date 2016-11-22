#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import norm
import scipy.stats as st
from mpmath import mp

mp.dps = 30

# ###########################################DATOS#########################
# muestra #1
S0 = pd.read_csv(
    'POP/Datos/lon160E-80W/SST_N_PROM_80W.csv', index_col=['major', 'minor'])
# SST_80W_masc = pd.read_csv('MAR(1)\Datos\lon160E-80W\SST_MASK_PROM_80W.csv',index_col=[0])
date = pd.date_range('1982', '2016', freq='M')
a = sorted(S0.index.levels[0], reverse=True)
b = S0.index.levels[1]
multin = pd.MultiIndex.from_product([a, b], names=['Lat', 'Lon'])
SST_80W = pd.DataFrame(S0.values, index=multin, columns=date)

# muestra#2
S1 = pd.read_csv(
    'POP/Datos/lon165E-125W/SST_N_PROM_125W.csv', index_col=['major', 'minor'])
a1 = sorted(S1.index.levels[0], reverse=True)
b1 = S1.index.levels[1]
multin1 = pd.MultiIndex.from_product([a1, b1], names=['Lat', 'Lon'])
SST_125W = pd.DataFrame(S1.values, index=multin1, columns=date)

pos = np.where(SST_80W.T.index.month == 1)[0]
d0 = SST_80W.T.ix[np.where(SST_80W.T.index.month == 1)[0]]
d1 = d0.T
# Separo los datos por meses


def B(matriz):
    E = mp.eig(matriz, left=False, right=False)
    E = mp.eig_sort(E)
    bd = matriz
    a = 0
    for y in range(5):
        a = a + 1
        d = np.identity(100)
        d[d == 1.0] = np.abs(mp.re(E[y]))
        d1 = mp.matrix(d)
        bd = bd + d1
        o = mp.eig(bd, left=False, right=False)
        o = mp.eig_sort(o)
        if mp.re(o[0]) > 0.001:
            break
    B1 = mp.cholesky(bd)
    return B1, a


def Mes(matriz, i):
    d0 = matriz.ix[np.where(matriz.index.month == i)[0]]
    d = mp.matrix(d0.values)
    m = mp.matrix([np.mean(d.T[ii, :]) for ii in range(d.T.rows)])
    return d.T, m


Ene_80W, M_ene = Mes(SST_80W.T, 1)
Feb_80W, M_Feb = Mes(SST_80W.T, 2)
Mar_80W, M_Mar = Mes(SST_80W.T, 3)
Abr_80W, M_Abr = Mes(SST_80W.T, 4)
May_80W, M_may = Mes(SST_80W.T, 5)
Jun_80W, M_Jun = Mes(SST_80W.T, 6)
Jul_80W, M_Jul = Mes(SST_80W.T, 7)
Ago_80W, M_Ago = Mes(SST_80W.T, 8)
Sep_80W, M_Sep = Mes(SST_80W.T, 9)
Oct_80W, M_Oct = Mes(SST_80W.T, 10)
Nov_80W, M_Nov = Mes(SST_80W.T, 11)
Dic_80W, M_Dic = Mes(SST_80W.T, 12)

Ene_125W, M125W_ene = Mes(SST_125W.T, 1)
Feb_125W, M125W_Feb = Mes(SST_125W.T, 2)
Mar_125W, M125W_Mar = Mes(SST_125W.T, 3)
Abr_125W, M125W_Abr = Mes(SST_125W.T, 4)
May_125W, M125W_may = Mes(SST_125W.T, 5)
Jun_125W, M125W_Jun = Mes(SST_125W.T, 6)
Jul_125W, M125W_Jul = Mes(SST_125W.T, 7)
Ago_125W, M125W_Ago = Mes(SST_125W.T, 8)
Sep_125W, M125W_Sep = Mes(SST_125W.T, 9)
Oct_125W, M125W_Oct = Mes(SST_125W.T, 10)
Nov_125W, M125W_Nov = Mes(SST_125W.T, 11)
Dic_125W, M125W_Dic = Mes(SST_125W.T, 12)

Y = Mar_80W  # presente
X = Feb_80W  # pasado

# estimar M0 yM1


M0_act = ((Y * Y.T) / 100)
M0_in = (M0_act**-1)
MP = (M0_act * M0_in)
o = MP[0:5, 0:5]
o0 = M0_in[0:5, 0:5]
M1_act = ((Y * X.T) / 100)
M0_ant = ((X * X.T) / 100)
A = M1_act * (M0_ant ** -1)
BBT = M0_act - (A * M1_act.T)

####
E = mp.eig(BBT, left=False, right=False)
E = mp.eig_sort(E)
# reajuste 0

d = np.identity(100)
d[d == 1.0] = abs(-24.7174268506955680247383180094514)
d1 = mp.matrix(d)
BBT0 = BBT + d1
E0 = mp.eig(BBT0, left=False, right=False)
E0 = mp.eig_sort(E0)

# reajuste 0
d0 = np.identity(100)
d0[d0 == 1.0] = abs(-0.00000062521499573893501604735293439538)
d2 = mp.matrix(d0)
BBT1 = BBT0 + d2
E1 = mp.eig(BBT1, left=False, right=False)
E1 = mp.eig_sort(E1)

# choslesky
B0 = mp.cholesky(BBT0)
B1 = mp.cholesky(BBT1)

# el modelo
x_ant = Feb_80W.T[0, :].T
x0 = x_ant - M_Feb
alea = mp.randmatrix(100, 1)
med_sim = M_Mar
mes_sim = ((A * x0) + (B1 * alea)) + med_sim

s0 = mp.chop(BBT0 - B1 * B1.H)
s1 = mp.chop(BBT1 - B1 * B1.H)

Z_ant = np.array(x_ant, dtype=float)
Z_sim = np.array(mes_sim, dtype=float)
SK = st.ks_2samp(Z_ant, Z_sim)


# funciones

def parametros(Y, X):
    M0_act = ((Y * Y.T) / 100)
    M1_act = ((Y * X.T) / 100)
    M0_ant = ((X * X.T) / 100)
    A = M1_act * (M0_ant**-1)
    BBT = M0_act - (A * M1_act.T)
    return A, BBT


# Y=PRESEM¿NTE
# x=PASADO

A_00, BBT_00 = parametros(Ene_80W, Dic_80W)
A_01, BBT_01 = parametros(Feb_80W, Ene_80W)
A_02, BBT_02 = parametros(Mar_80W, Feb_80W)
A_03, BBT_03 = parametros(Abr_80W, Mar_80W)
A_04, BBT_04 = parametros(May_80W, Abr_80W)
A_05, BBT_05 = parametros(Jun_80W, May_80W)
A_06, BBT_06 = parametros(Jul_80W, Jun_80W)
A_07, BBT_07 = parametros(Ago_80W, Jul_80W)
A_08, BBT_08 = parametros(Sep_80W, Ago_80W)
A_09, BBT_09 = parametros(Oct_80W, Sep_80W)
A_10, BBT_10 = parametros(Nov_80W, Oct_80W)
A_11, BBT_11 = parametros(Dic_80W, Nov_80W)


B_0 = B(BBT_00)
