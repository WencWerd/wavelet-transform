import numpy as np
from scipy.special import j1 as BesselJ
from matplotlib import pyplot
import matplotlib

#name_data - like array type

#-------------------------------Формула Симпсона-------------------------------
def simpson_rule(f_data, h):
    """
    Формула Симпсона для таблично заданной
    функции f с постоянным шагом аргумента h
    """
    s1 = np.sum(f_data[2 : -1 : 2])
    s2 = np.sum(f_data[1 : -1 : 2])
    return (h / 3) * (f_data[0] + 2 * s1 + 4 * s2 + f_data[-1])
#------------------------------------------------------------------------------


#------------------------------------Вейвлет-----------------------------------
def wavelet(x_data):
    """
    Вейвлет Морле
    """
    t0_data = np.cos(x_data * (np.pi * (2 / np.log(2)) ** 0.5))
    t1_data = np.exp(-x_data ** 2 / 2)
    t2_data = np.pi ** 0.25
    return t0_data * t1_data / t2_data
#------------------------------------------------------------------------------


#-----------------------------Вейвлет-пребразование----------------------------    
def wavelet_transform(f_data, t_data, a_data, b_data):
    """
    Вейвлет-пребразование для таблично
    заданной функции f на промежутке t
    с вейвлетом Морле на промежутке (t - b) / a\n
    Возвращает двумерный набор данных.
    """
    len_a = len(a_data)
    len_b = len(b_data)
    h = t_data[1] - t_data[0]
    data = np.empty((len_a,  len_b), dtype=f_data.dtype)
    for i in range(len_a):
        _a = a_data[i]
        for j in range(len_b):
            psi_data = wavelet((t_data - b_data[j]) / _a)
            integrand_data = f_data * psi_data
            data[i][j] = simpson_rule(integrand_data, h) / _a
    return data
#------------------------------------------------------------------------------


#----------------Преобразование Лапласа от заданной функции p(t)---------------
def P(w_data):
    """
    Преобразование Лапласа от заданной функции p(t)\n
    integral( p(t) * e^(-i * w * t) )dt, t = 0..+infinity
    """
    data = np.empty(len(w_data), dtype=np.complex128)
    for i, w in enumerate(w_data):
        w2 = w ** 2
        e = np.exp(2j * w)

        if w == 2 or w == -2:
            v_1 = 0
        else:
            v_1 = (     w + e * (   -w * np.cos(4) + 2j * np.sin(4))) / (    w2 - 4)

        if w == 3 / 2 or w == -3 / 2:
            v_2 = 0
        else:
            v_2 = (-2 * w + e * (2 * w * np.cos(3) - 3j * np.sin(3))) / (4 * w2 - 9)

        if w == 5 / 2 or w == -5 / 2:
            v_3 = 0
        else:
            v_3 = (-2 * w + e * (2 * w * np.cos(5) - 5j * np.sin(5))) / (4 * w2 - 25)
        data[i] = 1j * (v_1 + v_2 + v_3)
    return data
#------------------------------------------------------------------------------


#----------------Преобразование Фурье от заданной функции q(x1)----------------
def Q(a_data):
    """
    Преобразование Фурье от заданной функции q(x1)\n
    integral( q(x1) * e^(i * a * x1) )d(x1), x1 = -infinity..+infinity
    """
    a_data_abs = np.abs(a_data)
    value = np.pi * BesselJ(a_data_abs) / a_data_abs #BesselJ(1, |a|)
    return value
#------------------------------------------------------------------------------


#--------------Корни выражения 1 + e^(2*H*Sqrt(a^2 - k^2)) по "a"--------------
def alphaN(k, h):
    """
    Корни выражения 1 + e^(2*H*Sqrt(a^2 - k^2)) по "a"
    """
    t0 = np.pi if h == 1 else np.pi / h
    t1 = np.square(k)
    t2 = np.square(np.pi / h)
    #Количество вещественных корней + мнимых
    n = int(1/2 + abs(k) / np.pi * h) + 20
    data = np.empty(n, dtype=np.complex128)
    for i in range(1, n + 1):
        data[i - 1] = np.sqrt(t1 - t2 * np.square(i - 0.5), dtype=np.complex128)    
    return data
#-----------------------------------------------------------------------------


#---------------------Вычеты функции K() в точках xi_data---------------------
def resK(xi_data, x2, k, h):
    """
    Вычеты функции K() в точках xi_data
    """
    lambda_data = np.sqrt(xi_data ** 2 - k ** 2, dtype=np.complex128)
    if x2 == 0:
        t0_data = Q(xi_data) * lambda_data
        t1_data = lambda_data * h
        t2_data = np.cosh(t1_data) / np.sinh(t1_data)
        t3_data = xi_data * (t2_data + t1_data)
        return t0_data / t3_data
    else:
        G = Q(xi_data) * np.sinh(lambda_data * (h + x2)) * np.exp(-lambda_data * x2)
        H1 =   xi_data * np.cosh(lambda_data * h) / lambda_data \
             + h * xi_data * np.sinh(lambda_data * h)
        return G / H1 #array
#-----------------------------------------------------------------------------


#---------------------------------Интеграл (1)--------------------------------
def u_w(x1, x2, w_data, h, m0, p0):
    """
    Интеграл (1)
    """
    t0 = 1 if m0 == p0 else np.sqrt(m0 / p0)
    data = np.empty(len(w_data), dtype=np.complex128)
    for i, w in enumerate(w_data):
        k = w / t0
        poles = alphaN(k, h)
        data[i] = 1j * np.sum(resK(-poles, x2, k, h) * np.exp(1j * poles * x1))
    return data
#-----------------------------------------------------------------------------


#---------------------------------Интеграл (2)--------------------------------
def u_t(x1, x2, w_data, h, m0, p0, t_data):
    """
    Интеграл (2)
    """
    _h = w_data[1] - w_data[0]
    #u_w(x1, x2, w_data, h, m0, p0)
    t0_data = np.multiply(u_w(x1, x2, w_data, h, m0, p0), P(w_data), dtype=np.complex128)
    data = np.empty(len(t_data), dtype=np.complex128)
    for i, _t in enumerate(t_data):    
        e = np.exp((-1j * _t) * w_data)
        integrand = np.multiply(t0_data, e)
        data[i] = simpson_rule(integrand, _h) / (2 * np.pi)
    return data
#-----------------------------------------------------------------------------


#-------------------------------------main-------------------------------------
x1 = np.int64(5)
x2 = np.int64(0)
h  = np.int64(1)
m0 = np.int64(1)
p0 = np.int64(1)

w_n = 10**7 + 1#2**20 + 1
w_a = -100
w_b =  100
t_n = 2 * 10**4 + 1#2**14 + 1
t_a = 0
t_b = 1000

w = np.linspace(w_a, w_b, w_n, dtype=np.float64)
t = np.linspace(t_a, t_b, t_n, dtype=np.float64)

u_t_data = u_t(x1, x2, w, h, m0, p0, t)

#Графики интеграла (2)
#первый с учётом знака вещ. части, второй без

#№pyplot.figure("u_{0}_{1}_{2}".format(x1, t_b, t_n))
#pyplot.plot(t, [abs(z) * np.sign(z.real) for z in u_t_data])
#fig = pyplot.gcf()
#fig.set_size_inches((19.20,10.80), forward=False)
#pyplot.savefig("screenshots\\u(t)_abs_and_sign_of_real.jpg", dpi=500, format='jpg')

#pyplot.figure("u_{0}_{1}_{2}_abs".format(x1, t_b, t_n))
#pyplot.plot(t, np.abs(u_t_data))
#fig = pyplot.gcf()
#fig.set_size_inches((19.20,10.80), forward=False)
#pyplot.savefig("screenshots\\u(t)_abs.jpg", dpi=500, format='jpg')

"""
#Проверка интеграла (1)
for n in u_w(x1, x2, [i for i in range(-3, 3+1)], h, m0, p0):
    print("{0.real:0.9f} + {0.imag:0.9f}i".format(n))
print()
"""
"""
#График интеграла (1)
u_w_data = u_w(x1, x2, w, h, m0, p0)
pyplot.figure("u_w")
pyplot.plot(w, np.abs(u_w_data))
"""

"""
#Проверка интеграла (2)
pyplot.figure("p")
pyplot.plot(t, [abs(z) * np.sign(np.real(z)) for z in u_t_data])
x = np.linspace(0, 2, 2000)
f1 = [np.cos(2 * x_) * (1 - np.cos(x_ / 2)) for x_ in x]
f2 = [0 for x_ in x]
xx = np.linspace(0, 4, 4000)
pyplot.plot(xx, f1 + f2)
"""

"""
#Проверка здания функции P
pyplot.figure("P")
x = np.linspace(-20, 20, 10000)
f = np.abs(P(x))
pyplot.plot(x, f)
"""

"""
#Проверка здания функции Q
pyplot.figure("Q")
x = np.linspace(-20, 20, 10000)
f = Q(x)
pyplot.plot(x, f)
"""

"""
#Построение вейвлет преобразования
scales_n = 501
scales_a = 1.5
scales_b = 5.0
scales = np.linspace(scales_a, scales_b, scales_n, dtype=np.float64)
f_wavelet_transform = wavelet_transform(u_t_data, t, scales, t)

f_wavelet_transform_abs = np.abs(f_wavelet_transform)
pyplot.figure("Вейвлет-преобразование (модуль)")
cmap = pyplot.get_cmap("inferno_r")
pyplot.pcolormesh(t, scales, f_wavelet_transform_abs, shading='gouraud', cmap=cmap)
pyplot.colorbar()
pyplot.ylabel('Scales')
pyplot.xlabel('Times')
fig = pyplot.gcf()
fig.set_size_inches((19.20,10.80), forward=False)
"""
#pyplot.savefig("screenshots\\wavelet_transform_abs.jpg", dpi=500, format='jpg')

pyplot.show()
#------------------------------------------------------------------------------