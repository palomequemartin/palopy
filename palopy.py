import time
from os.path import exists
from inspect import getsource
from types import FunctionType
import numpy as np
from scipy.stats import chi2
from scipy.special import comb
from pandas import Series
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
        df['Hora'] = Series([get_time()], index=[0])
    
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
    
    N = len(X)
    
    E_X = np.mean(X)
    sigma_E_X = np.sqrt(np.sum(X - E_X)**2 / (N**2 - N))
    
    return E_X, sigma_E_X


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
def propagate(functions, data, covariances):
    '''
    Propagate error of multiple functions with the same parameters

    functions:
        Single function or list of functions to which propagate error.
        Each must have the same parameters
    data:
        Experimental data, parameters of every f in order
    covariances:
        Covariance matrix or 1D array of sigmas of data

    Returns:
        Expected value and covariance matrix of function results
    '''
    # Convert input to arrays
    functions = list(functions)
    data = np.array(data)
    covariances = np.array(covariances)
    # If covariances is 1D assume array of sigmas
    if covariances.ndim == 1:
        covariances = np.diag(covariances**2)

    # Expected value of function results [RETURN]
    expected = np.array([f(*data) for f in functions])

    # Find number of functions and parameters
    n_funcs = len(functions)  # N of functions
    n_params = functions[0].__code__.co_argcount  # N of parameters
    params = sp.symbols(functions[0].__code__.co_varnames[:n_params])  # Parameters

    # Define matrix with function derivatives for each parameter
    derivatives = np.zeros((n_funcs, n_params))
    for i, f in enumerate(functions):
        for j, param in enumerate(params):
            derivatives[i, j] = sp.lambdify(params, sp.diff(f, param).simplify())(*data)

    # Covariance matrix of function results [RETURN]
    covariance = derivatives @ covariances @ derivatives.T

    # Make into single numbers if only 1 function
    if n_funcs == 1:
        expected = expected[0]
        covariance = np.sqrt(covariance[0, 0])  # Sigma

    return expected, covariance



# Least squares for lineal parameters with correlated measurements
def least_squares(F, X_data, Y_data, Cov_Y, p_value=False):
    
    n_data = len(X_data)  # N of data points
    n_params = F.__code__.co_argcount - 1  # N of parameters
    params = sp.symbols(F.__code__.co_varnames[1:n_params+1])  # Parameters (like curve_fit)
        
    # Convert to array
    X_data = np.array(X_data)
    Y_data = np.array([np.array(Y_data)]).T
    Cov_Y = np.array(Cov_Y)
    # Only if Cov_Y is 1D assume independent and sqrt of variance
    if len(Cov_Y.shape) == 1:
        Cov_Y = np.diag(Cov_Y**2)
    
    # Function correction for sympy if necessary
    if 'np.' in getsource(F):
        F_code = compile(getsource(F).replace('np.', 'sp.'), '', 'exec')
        F = FunctionType(F_code.co_consts[0], globals(), "gfg")
    
    F = F(sp.symbols('X'), *params)  # Make functions to sympy expresion
    
    # A matrix definition
    A = []
    for param in params:
        F_prime_i = sp.diff(F, param).simplify()
        # If derivative is not constant evaluate on data
        if 'X' in str(F_prime_i):
            F_prime_i = sp.lambdify('X', F_prime_i)
            A.append(F_prime_i(X_data))
        # Else append n_data sized list of constant
        else:
            F_prime_i = n_data*[float(F_prime_i)]
            A.append(np.array(F_prime_i))
    A = np.array(A).T
    
    # Results
    pcov = np.linalg.inv(A.T @ np.linalg.inv(Cov_Y) @ A)
    popt = pcov @ A.T @ np.linalg.inv(Cov_Y) @ Y_data
    
    if p_value:
        
        chi_sq = (Y_data - A @ popt).T @ np.linalg.inv(Cov_Y) @ (Y_data - A @ popt)
        P = 1 - chi2.cdf(chi_sq[0,0], n_data - n_params)
        
        RETURN = (popt.T[0], pcov, P)
    
    else:
        
        RETURN = (popt.T[0], pcov)
    
    return RETURN


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



### FREQUENTIST HYPOTHESIS TESTS ###

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