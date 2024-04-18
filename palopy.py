import time
from os.path import exists
from inspect import getsource
from types import FunctionType
import numpy as np
from scipy.stats import chi2
from scipy.special import comb
import sympy as sp


### GENERAL USEFUL STUFF ###

# Get current time (UTC-3)
def get_time():
    t = time.gmtime()
    h = int(time.strftime('%H', t)) - 3
    m_s = time.strftime('%M:%S', t)
    tiempo = f'{h}:{m_s}'
    return tiempo


# Save csv without overwriting
def save(df, filename, path='.', time=True):
    
    if time:
        df['Hora'] = get_time()
    
    if exists(f'{path}/{filename}.csv') or exists(f'{path}/{filename} (0).csv') or exists(f'{path}/{filename}(0).csv'):
        i = 1
        while True:
            if exists(f'{path}/{filename} ({i}).csv'):
                i += 1
                continue
            else:
                df.to_csv(f'{path}/{filename} ({i}).csv')
                break
    else:
        df.to_csv(f'{path}/{filename}.csv')



### ESTIMADORES ###

# Promedio y estimador de la varianza
def mu_sigma(X):
    
    X = np.array(X)
    
    E_X = np.mean(X)
    Var_X = np.sum(X - E_X)**2 / (len(X) - 1)
    
    return E_X, np.sqrt(Var_X)


# Promedio pesado y su varianza
def weighted(X, sigma):
    
    X = np.array(X)
    sigma = np.array(sigma)
    
    E_X = np.sum(X / sigma**2) / np.sum(1 / sigma**2)
    Var_X = 1 / np.sum(1 / sigma**2)
    
    return E_X, np.sqrt(Var_X)


# Correlacion
def correlation(x, y):
    
    x_med = np.mean(x)
    y_med = np.mean(y)
    
    cov = np.sum((x - x_med) * (y - y_med))
    stds = np.sum((x - x_med)**2) * np.sum((y - y_med)**2)
    
    return cov / np.sqrt(stds)


# Suma en cuadratura
def quadsum(*args):
    
    args = np.array(args)
    sum_sq = np.sqrt(np.sum(args**2))
    
    return sum_sq



### LEAST SQUARES AND ERROR PROPAGATION ###

# General error propagation for multiple functions of data
def propagation(Funcs, Data, Cov_Data):
    '''
    Funcs is list of functions to which propagate error. Each must have the same parameters
    Data is experimental data, parameters of every F in order
    Cov_Data is covariance matrix or 1D array of sigmas of Data
    '''
    # Check if only 1 function and make subscriptable
    is_callable = callable(Funcs)
    if is_callable:
        Funcs = [Funcs]
    
    n_Funcs = len(Funcs)  # N of functions
    n_params = Funcs[0].__code__.co_argcount  # N of parameters
    params = sp.symbols(Funcs[0].__code__.co_varnames[:n_params])  # Parameters
    
    # Convert to array
    Data = np.array(Data)
    Cov_Data = np.array(Cov_Data)
    # Only if Cov_Data is 1D assume independent and sqrt of variance
    if len(Cov_Data.shape) == 1:
        Cov_Data = np.diag(Cov_Data**2)
    
    # Expected value for each function result [RETURN]
    Expected = np.array([Func(*Data) for Func in Funcs])
    
    # Function correction for sympy if necessary
    for i, Func in enumerate(Funcs):
        if 'np.' in getsource(Func):
            F_code = compile(getsource(Func).replace('np.', 'sp.'), '', 'exec')
            Func = FunctionType(F_code.co_consts[0], globals(), "gfg")
        Funcs[i] = Func(*params)  # Make functions to sympy expresion
    
    # Definition of matrix with function derivatives for each parameter
    F_primes = np.zeros((n_Funcs, n_params))
    for i, Func in enumerate(Funcs):
        for j, param in enumerate(params):
            F_primes[i, j] = sp.lambdify(params, sp.diff(Func, param).simplify())(*Data)
    
    # Covariance matrix of function results [RETURN]
    Covariance = F_primes @ Cov_Data @ F_primes.T
    
    # Make into single numbers if only 1 function
    if is_callable:
        Expected = Expected[0]
        Covariance = np.sqrt(Covariance[0,0])  # sqrt of variance
    
    return Expected, Covariance


# Least squares for lineal parameters with correlated measurements
def least_squares(f, x, y, sigma):
    
    n = len(x)  # Numero de mediciones
    k = f.__code__.co_argcount - 1  # Numero de parametros
    parametros = f.__code__.co_varnames[1:k+1]
    
    
    # Conversion a array y definicion matriz sigma
    x = np.array(x)
    y = np.array([np.array(y)]).T
    sigma = np.array(sigma)
    # Si cov no es una matriz asumo que es raiz de var
    if len(sigma.shape) == 1:
        sigma = np.identity(n) * sigma**2
    
    
    # Correccion de la funcion
    if 'np.' in getsource(f):
        f_code = compile(getsource(f).replace('np.', 'sp.'), '', 'exec')
        f = FunctionType(f_code.co_consts[0], globals(), "gfg")
    
    f = f(sp.symbols('x'), *sp.symbols(parametros))
    
    
    # Definicion de la matriz A
    A = []
    for parametro in sp.symbols(parametros):
        fprima_i = sp.diff(f, parametro).simplify()
        if 'x' in str(fprima_i):
            fprima_i = sp.lambdify('x', fprima_i)
            A.append(fprima_i(x))
        else:
            fprima_i = n*[float(fprima_i)]
            A.append(np.array(fprima_i))
    
    A = np.array(A).T
    
    
    # Resultados
    pcov = np.linalg.inv(A.T @ np.linalg.inv(sigma) @ A)
    popt = pcov @ A.T @ np.linalg.inv(sigma) @ y
    
    chi_sq = (y - A @ popt).T @ np.linalg.inv(sigma) @ (y - A @ popt)
    P = 1 - chi2.cdf(chi_sq[0,0], n-k)
    
    return popt.T[0], pcov, P


# Lineal least squares
def lineal(x, a1, a2):
    return a1 + a2*x

def lin_least_squares(x, y, sigma):
    
    n = len(x)
    Delta = n*np.sum(x**2) - np.sum(x)**2
    
    a1 = (np.sum(x**2)*np.sum(y) - np.sum(x)*np.sum(x*y)) / Delta
    a2 = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / Delta
    cov = ((sigma**2)/Delta)*np.array([[np.sum(x**2), -np.sum(x)], [-np.sum(x), n]])
    
    return np.array([a1, a2]), cov



### TESTS DE HIPOTESIS ###

# Test chi cuadrado para datos en un histograma
def chisq_test_hist(obs, exp, params_fit=0):
    
    df = len(obs) - 1 - params_fit
    
    chisq = np.sum((obs - exp)**2 / exp)
    
    P_chisq = 1 - chi2.cdf(chisq, df)
    
    return chisq, P_chisq


# Correccion de bineado para que no haya menos de 5 eventos en un bin
def correccion_bineado(datos, bins='auto', min=5):
    
    entr_temp, bins_temp = np.histogram(datos, bins=bins)
    
    bins_finales = [bins_temp[0]]
    counts = 0
    for i in range(len(entr_temp)):
        counts += entr_temp[i]
        if counts >= min:
            bins_finales.append(bins_temp[i+1])
            counts = 0
    
    return bins_finales


# Wald–Wolfowitz runs test
def run_test(x):
    
    N = len(x)
    n_pos = len(np.where(x >= 0)[0])
    n_neg = len(np.where(x < 0)[0])
    
    runs = (x[:-1] * x[1:] < 0).sum() + 1
    
    P_runs = 0
    for num in range(2, runs+1):
        if num % 2 == 0:
            P_runs += 2*comb(n_pos-1, num-1)*comb(n_neg-1, num-1) / comb(N, n_pos)
        else:
            P_runs += (comb(n_pos-1, num-2)*comb(n_neg-1, num-1) + comb(n_pos-1, num-1)*comb(n_neg-1, num-2)) / comb(N, n_pos)
    
    return runs, P_runs


# Combinar p-values
def comb_test(*args):
    chisq = -2*np.log(np.prod(np.array(args)))
    P_chisq = 1 - chi2.cdf(chisq, 2*len(args))
    return chisq, P_chisq