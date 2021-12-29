import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Find implied volatility

def bs_call(S, K, T, vol):
    d1 = (np.log(S/K) + (0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - K * norm.cdf(d2)

def bs_put(S, K, T, vol):
    d1 = (np.log(S/K) + (0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return -S * norm.cdf(-d1) + K * norm.cdf(-d2)

def bs_vega(S, K, T, sigma):
    d1 = (np.log(S / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def find_vol_call(target_value, S, K, T, *args):
    MAX_ITERATIONS = 200
    PRECISION = 1/1000000
    sigma = 0.3
    for i in range(0, MAX_ITERATIONS):
        price = bs_call(S, K, T, sigma)
        vega = bs_vega(S, K, T, sigma)
        #print('V:' + str(vega))
        diff = target_value - price  # our root
        #print(diff)
        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)
        #print(sigma)
    return sigma # value wasn't found, return best guess so far

def find_vol_put(target_value, S, K, T, *args):
    MAX_ITERATIONS = 200
    PRECISION = 1/1000000
    sigma = 0.3
    for i in range(0, MAX_ITERATIONS):
        price = bs_put(S, K, T, sigma)
        vega = bs_vega(S, K, T, sigma)
        diff = target_value - price  # our root
        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)
    return sigma # value wasn't found, return best guess so far


def Risk_Neutral_Mils_col_deterministic(len_t, dt, Nsims, v0, S0, kappa, theta, eta, z_sim_v, z_sim_s):
    var_path_det = np.zeros((Nsims, len_t))
    var_path_det[:, 0] = v0

    S_path_det = np.zeros((Nsims, len_t))
    S_path_det[:, 0] = S0

    # Euler X and Mils v
    for i in range(len_t - 1):
        var_path_det[:, i + 1] = var_path_det[:, i] + kappa * (theta - var_path_det[:, i]) * dt

        # Euler Simulation of log(Stock Price)
        S_path_det[:, i + 1] = S_path_det[:, i] * np.exp( \
            - 0.5 * np.maximum(var_path_det[:, i], 0) * dt + \
            np.sqrt(np.maximum(var_path_det[:, i], 0)) * np.sqrt(dt) * (z_sim_s[:, i]))

    return var_path_det, S_path_det


def Risk_Neutral_Mils_col(len_t, dt, Nsims, v0, S0, kappa, theta, eta, z_sim_v, z_sim_s):
    var_path = np.zeros((Nsims, len_t))
    var_path[:, 0] = v0

    S_path = np.zeros((Nsims, len_t))
    S_path[:, 0] = S0

    # Euler X and Mils v
    for i in range(len_t - 1):
        # Milstein Simulation of variance
        var_path[:, i + 1] = var_path[:, i] + \
                             kappa * (theta - np.maximum(var_path[:, i], 0)) * dt + \
                             eta * np.sqrt(np.maximum(var_path[:, i], 0)) * np.sqrt(dt) * z_sim_v[:, i] + \
                             0.25 * eta ** 2 * (dt * (z_sim_v[:, i]) ** 2 - dt)

        # Euler Simulation of log(Stock Price)
        S_path[:, i + 1] = S_path[:, i] * np.exp( \
            - 0.5 * np.maximum(var_path[:, i], 0) * dt + \
            np.sqrt(np.maximum(var_path[:, i], 0)) * np.sqrt(dt) * (z_sim_s[:, i]))

    return var_path, S_path


if __name__ == '__main__':
    r = 0
    S0 = 1
    v0_sqrt = 0.2
    v0 = v0_sqrt ** 2
    kappa = 3
    theta_sqrt = 0.4
    theta = theta_sqrt ** 2
    eta = 1.5
    rho = -0.5

    Nsims = 5000
    dt = 1 / 1000

    T = [0.25, 0.5, 1]
    # len_t = int(T/dt)

    K_steps = 5
    K_call = np.linspace(1, 1.2, K_steps)
    K_put = np.linspace(0.8, 1, K_steps)

    # 2c simulate stock price with heston
    T1 = 0.25
    len_t1 = int(T1 / dt)
    t_line = np.linspace(0, T1, len_t1)

    mu = np.array([0, 0])
    cov = np.array([[1, rho], [rho, 1]])

    z_sim_v = np.zeros((Nsims, len_t1))
    z_sim_s = np.zeros((Nsims, len_t1))

    for i in range(0, Nsims):
        W = np.random.multivariate_normal(mu, cov, size=len_t1)
        z_sim_v[i, :] = W[:, 0]
        z_sim_s[i, :] = W[:, 1]

    [var_path_1, S_path_1] = Risk_Neutral_Mils_col(len_t1, dt, Nsims, v0, S0, kappa, theta, eta, z_sim_v, z_sim_s)

    # plt.plot(t_line, S_path_1.T)
    # plt.xlabel('Time')
    # plt.ylabel('Simulated Paths')
    # plt.title('Simulated Path with Heston Model')
    # plt.savefig('2c_stock_price_heston.jpg')

    [var_path_2, S_path_2] = Risk_Neutral_Mils_col_deterministic(len_t1, dt, Nsims, v0, S0, kappa, theta, eta, z_sim_v, z_sim_s)

    # plt.plot(t_line, S_path_2.T)
    # plt.xlabel('Time')
    # plt.ylabel('Simulated Paths')
    # plt.title('Simulated Path with Deterministic Variance')
    # plt.savefig('2c_stock_price_determin.jpg')

    # 2c1 option price with deterministic variance path
    # T = 0.25
    fin_vol_call_1 = np.zeros(K_steps)
    fin_vol_call_2 = np.zeros(K_steps)
    for i in range(0, K_steps):
        option_price = np.maximum(S_path_1[:, -1] - K_call[i], 0)
        mean_price = np.mean(option_price)
        # some confidence interval
        fin_vol_call_1[i] = find_vol_call(mean_price, S0, K_call[i], T1)

        option_price_2 = np.maximum(S_path_2[:, -1] - K_call[i], 0)
        mean_price_2 = np.mean(option_price_2)
        # some confidence interval
        fin_vol_call_2[i] = find_vol_call(mean_price_2, S0, K_call[i], T1)

    fin_vol_put_1 = np.zeros(K_steps)
    fin_vol_put_2 = np.zeros(K_steps)
    for i in range(0, K_steps):
        option_price = np.maximum(-S_path_1[:, -1] + K_put[i], 0)
        mean_price = np.mean(option_price)
        # some confidence interval
        fin_vol_put_1[i] = find_vol_put(mean_price, S0, K_put[i], T1)

        option_price_2 = np.maximum(-S_path_2[:, -1] + K_put[i], 0)
        mean_price_2 = np.mean(option_price_2)
        # some confidence interval
        fin_vol_put_2[i] = find_vol_put(mean_price_2, S0, K_put[i], T1)

    plt.plot(K_call, fin_vol_call_1, label='Risk-Netural Simulated IV_call without control')
    plt.plot(K_put, fin_vol_put_1, label='Risk-Netural Simulated IV_put without control')
    plt.plot(K_call, fin_vol_call_2, label='Risk-Netural Simulated IV_call with control')
    plt.plot(K_put, fin_vol_put_2, label='Risk-Netural Simulated IV_put with control')
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.title('Volatility Simulation for T = 0.25')
    plt.legend()
    plt.savefig('2ci_a.jpg')
