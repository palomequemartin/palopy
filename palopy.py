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
    sigma_E_X = np.sqrt(np.sum((X - E_X)**2) / (N**2 - N))
    
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

def propagation(functions, data, sigma):
    """
    Propagate error of multiple functions with the same parameters

    Parameters
    ----------
    functions : callable or list of callables
        Single function or list of functions to which propagate error.
        Each must have the same parameters, even if some are not used.
    data : array_like
        Experimental data, parameters of every f in order.
    sigma : array_like
        If `sigma` is a 2-D array it should contain the covariance matrix
        of errors of `data`. If `sigma` is a 1-D array it should contain
        values of standard deviations of errors of `data`.

    Returns
    -------
    expected : array_like
        Expected value of function results.
    covariance : array_like
        Covariance matrix of function results.
    """
    # Convert data and sigma to arrays and make functions subscriptable
    if callable(functions):
        functions = [functions]
    data = np.array(data)
    sigma = np.array(sigma)
    # If sigma is 1D assume array of standard deviations
    if sigma.ndim == 1:
        sigma = np.diag(sigma**2)

    # Expected value of function results [RETURN]
    expected = np.array([f(*data) for f in functions])

    # Find number of functions and parameters
    n_funcs = len(functions)  # Number of functions
    n_params = functions[0].__code__.co_argcount  # Number of parameters
    params = sp.symbols(functions[0].__code__.co_varnames[:n_params])  # Parameters
    
    # Function correction for sympy if necessary
    for i, f in enumerate(functions):
        if 'np.' in getsource(f):
            f_code = compile(getsource(f).replace('np.', 'sp.'), '', 'exec')
            f = FunctionType(f_code.co_consts[0], globals(), "gfg")
        functions[i] = f(*params)  # Make functions to sympy expresion
    
    # Define matrix with function derivatives for each parameter
    derivatives = np.zeros((n_funcs, n_params))
    for i, f in enumerate(functions):
        for j, param in enumerate(params):
            derivatives[i, j] = sp.lambdify(params, sp.diff(f, param).simplify())(*data)

    # Covariance matrix of function results [RETURN]
    covariance = derivatives @ sigma @ derivatives.T

    # Make into single numbers if only 1 function
    if n_funcs == 1:
        expected = expected[0]
        covariance = np.sqrt(covariance[0, 0])  # standard deviation

    return expected, covariance


def least_squares(f, xdata, ydata, sigma, chi2_test=True):
    """
    Least squares for a linear function in the parameters and for
    measurements with covariance.

    Parameters
    ----------
    f : callable
        Function to fit, must be defined as f(x, *parameters), must be
        lineal in the parameters.
    xdata : array_like
        Experimental data in the independent variable.
    ydata : array_like
        Experimental data in the dependent variable.
    sigma : array_like
        If `sigma` is a 2-D array it should contain the covariance matrix
        of errors of `ydata`. If `sigma` is a 1-D array it should contain
        values of standard deviations of errors of `ydata`.
    chi2_test : bool, optional
        If True, returns the p-value of a chi-squared test of the fit
        (default is True).

    Returns
    -------
    popt : array_like
        Optimal values of parameters.
    pcov : array_like
        Covariance matrix of parameters.
    p_value : float, optional
        p-value of the test if chi2_test is True.
    """
    # Convert inputs to array, ydata must be a column vector
    xdata = np.array(xdata)
    sigma = np.array(sigma)
    # If sigma is 1-D assume array of standard deviations
    if sigma.ndim == 1:
        sigma = np.diag(sigma**2)
    
    # Find number of data points and parameters
    n_data = len(xdata)  # Number of data points
    n_params = f.__code__.co_argcount - 1  # Number of parameters
    
    # Constant term in f does not affect the optimization of parameters
    constant = f(xdata, *np.zeros(n_params))
    ydata = np.array([np.array(ydata) - constant]).T
    
    # Define matrix with every f(x) evaluated in data by isolating each parameter
    functions_x = np.array([f(xdata, *row) - constant for row in np.identity(n_params)])
    
    # Results
    sigma_inv = np.linalg.inv(sigma)
    pcov = np.linalg.inv(functions_x @ sigma_inv @ functions_x.T)
    popt = pcov @ functions_x @ sigma_inv @ ydata
    
    if chi2_test:
        
        chi_sq = (ydata - functions_x.T @ popt).T @ sigma_inv @ (ydata - functions_x.T @ popt)
        p_value = chi2.sf(chi_sq[0,0], n_data - n_params)
        
        return popt.T[0], pcov, p_value
    
    return popt.T[0], pcov


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