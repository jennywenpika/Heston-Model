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
    MAX_ITERATIONS = 1000
    PRECISION = 1/1000000
    sigma = 0.15
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
    MAX_ITERATIONS = 1000
    PRECISION = 1/1000000
    sigma = 0.1
    for i in range(0, MAX_ITERATIONS):
        price = bs_put(S, K, T, sigma)
        vega = bs_vega(S, K, T, sigma)
        diff = target_value - price  # our root
        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)
    return sigma # value wasn't found, return best guess so far


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


def Risk_Neutral_Mils_col_deterministic(len_t, dt, Nsims, v0, S0, kappa, theta, eta, z_sim_v, z_sim_s):
    var_path_det = np.zeros((Nsims, len_t))
    var_path_det[:, 0] = v0

    S_path_det = np.zeros((Nsims, len_t))
    S_path_det[:, 0] = S0

    # Euler X and Mils v
    for i in range(len_t - 1):
        var_path_det[:, i + 1] = np.maximum(var_path_det[:, i], 0) + kappa * (theta - var_path_det[:, i]) * dt

        # Euler Simulation of log(Stock Price)
        S_path_det[:, i + 1] = S_path_det[:, i] * np.exp( \
            - 0.5 * np.maximum(var_path_det[:, i], 0) * dt + \
            np.sqrt(np.maximum(var_path_det[:, i], 0)) * np.sqrt(dt) * (z_sim_s[:, i]))

    return var_path_det, S_path_det


def Risk_Neutral_Mils_col_b(len_t, dt, Nsims, v0, S0, kappa, theta, eta, z_sim_v, z_sim_s, strike_price = 0):
    var_path = np.zeros((Nsims, len_t))
    var_path[:, 0] = v0

    payoff_path = np.zeros((Nsims, len_t))
    payoff_path[:, 0] = S0 - strike_price

    # Euler X and Mils v
    for i in range(len_t - 1):
        # Milstein Simulation of variance
        var_path[:, i + 1] = var_path[:, i] + \
                             kappa * (theta - np.maximum(var_path[:, i], 0)) * dt + \
                             eta * np.sqrt(np.maximum(var_path[:, i], 0)) * np.sqrt(dt) * z_sim_v[:, i] + \
                             0.25 * eta ** 2 * (dt * (z_sim_v[:, i]) ** 2 - dt)

        # Euler Simulation of log(Stock Price)
        payoff_path[:, i + 1] = payoff_path[:, i] + np.maximum(var_path[:, i], 0) * dt

    return var_path, payoff_path


def variance_analytic(T):
    return theta + (v0 - theta) * np.exp(-kappa * T)

def contingent_Analytical(T):
    cont_claim = theta * T + (1 - np.exp(-kappa *T)) * (v0 - theta)/kappa
    return cont_claim


def contingent_Analytical_sqr(T):
    part = (2 * kappa * theta + eta **2) / (2 * kappa)
    part1 = part * theta * T
    part2 = (v0**2 - part * (2*v0 - theta)) * (1 - np.exp(-2*kappa*T))/ (2 *kappa)
    part3 = part *2*(v0 - theta) * (1-np.exp(- kappa * T))/ kappa
    cont_claim = part1 + part2 + part3
    return cont_claim


def contingent_Sim(T, dt, var_path_1):
    nSim = len(var_path_1[:, 1])

    simulated_vals = []
    for ii in range(nSim):
        # for each iteration we get a list of values (250)
        y_value = var_path_1[ii, :]
        num_of_y = len(y_value)

        # calculate contingent by left riemann sum

        left_riemann_sum = np.sum(y_value[0:num_of_y - 1] * dt)
        simulated_vals.append(max(left_riemann_sum, 0))
    return simulated_vals


def contingent_Sim_sqr(T, dt, var_path_1):
    nSim = len(var_path_1[:, 1])

    simulated_vals = []
    for ii in range(nSim):
        # for each iteration we get a list of values (250)
        y_value = var_path_1[ii, :]
        num_of_y = len(y_value)

        # calculate contingent by left riemann sum
        left_riemann_sum = np.sum(y_value[0:num_of_y - 1] ** 2 * dt)
        simulated_vals.append(max(left_riemann_sum, 0))
    return simulated_vals


def optimal_gmma(x_sim, cv_sim):
    gamma = np.cov(x_sim, cv_sim) / np.var(cv_sim)
    return gamma[0][1]


def cv_gamma(price_sim, cv1_sim, cv2_sim, cv3_sim, cv4_sim):
    a = np.array([[np.var(cv1_sim), np.cov(cv1_sim, cv2_sim)[0][1], np.cov(cv1_sim, cv3_sim)[0][1], np.cov(cv1_sim, cv4_sim)[0][1]],
                 [np.cov(cv2_sim, cv1_sim)[0][1], np.var(cv2_sim), np.cov(cv2_sim, cv3_sim)[0][1], np.cov(cv2_sim, cv4_sim)[0][1]],
                 [np.cov(cv3_sim, cv1_sim)[0][1], np.cov(cv3_sim, cv2_sim)[0][1], np.var(cv3_sim), np.cov(cv3_sim, cv4_sim)[0][1]],
                 [np.cov(cv4_sim, cv1_sim)[0][1], np.cov(cv4_sim, cv2_sim)[0][1], np.cov(cv4_sim, cv3_sim)[0][1], np.var(cv4_sim)]])
    b = np.array([np.cov(price_sim, cv1_sim)[0][1], np.cov(price_sim, cv2_sim)[0][1], np.cov(price_sim, cv3_sim)[0][1], np.cov(price_sim, cv4_sim)[0][1]])

    return np.linalg.solve(a, b.T)


def optimal_gamma_cv(price_sim, cv_list, cv_sim, cv1_sim, cv2_sim, cv3_sim):
    optimal_cv_gamma = np.cov(price_sim, cv_sim)
    - (cv_list[0] * np.cov(cv_sim, cv1_sim) + cv_list[1] * np.cov(cv_sim, cv2_sim) + cv_list[2] * np.cov(cv_sim, cv3_sim))
    return optimal_cv_gamma[0][1] / np.var(cv_sim)


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

    plt.figure(6)
    plt.plot(t_line, S_path_1.T)
    plt.xlabel('Time')
    plt.ylabel('Simulated Paths')
    plt.title('Simulated Path with Heston Model')
    plt.savefig('2c_stock_price_heston.jpg')

    [var_path_2, S_path_2] = Risk_Neutral_Mils_col_deterministic(len_t1, dt, Nsims, v0, S0, kappa, theta, eta, z_sim_v, z_sim_s)

    plt.figure(7)
    plt.plot(t_line, S_path_2.T)
    plt.xlabel('Time')
    plt.ylabel('Simulated Paths')
    plt.title('Simulated Path with Deterministic Variance')
    plt.savefig('2c_stock_price_determin.jpg')

    # 2c1 option price with deterministic variance path
    # T = 0.25



    fin_vol_call_1 = np.zeros(K_steps)
    fin_vol_call_2 = np.zeros(K_steps)
    fin_vol_call_3 = np.zeros(K_steps)
    fin_vol_call_4 = np.zeros(K_steps)
    fin_vol_call_5 = np.zeros(K_steps)
    fin_vol_call_6 = np.zeros(K_steps)
    for i in range(0, K_steps):
        option_price = np.maximum(S_path_1[:, -1] - K_call[i], 0)
        mean_price = np.mean(option_price)
        # some confidence interval
        fin_vol_call_1[i] = find_vol_call(mean_price, S0, K_call[i], T1)

        mean_price_2 = mean_price + optimal_gmma(option_price, var_path_1[:, -1]) * (variance_analytic(T1) - np.mean(var_path_1[:, -1]))
        print(mean_price, mean_price_2)
        # some confidence interval
        fin_vol_call_2[i] = find_vol_call(mean_price_2, S0, K_call[i], T1)

        contingent_sims = contingent_Sim(T1, dt, var_path_1)
        mean_price_3 = mean_price + optimal_gmma(option_price, contingent_sims) * (contingent_Analytical(T1) - np.mean(contingent_sims))
        fin_vol_call_3[i] = find_vol_call(mean_price_3, S0, K_call[i], T1)

        contingent_sims_sq = contingent_Sim_sqr(T1, dt, var_path_1)
        mean_price_4 = mean_price + optimal_gmma(option_price, contingent_sims_sq) * (
                    contingent_Analytical_sqr(T1) - np.mean(contingent_sims_sq))
        fin_vol_call_4[i] = find_vol_call(mean_price_4, S0, K_call[i], T1)

        mean_price_5 = mean_price + optimal_gmma(option_price, S_path_1[:, -1]) * (
                S0 - np.mean(S_path_1[:, -1]))
        fin_vol_call_5[i] = find_vol_call(mean_price_5, S0, K_call[i], T1)

        gamma_cv = cv_gamma(option_price, var_path_1[:, -1], contingent_sims, contingent_sims_sq, S_path_1[:, -1])

        mean_price_6 = mean_price \
                       + optimal_gamma_cv(option_price, gamma_cv[1:], var_path_1[:, -1], contingent_sims, contingent_sims_sq, S_path_1[:, -1]) * (variance_analytic(T1) - np.mean(var_path_1[:, -1])) \
                       + optimal_gamma_cv(option_price, list(gamma_cv[i] for i in [0,2,3]), contingent_sims, var_path_1[:, -1], contingent_sims_sq, S_path_1[:, -1]) * (contingent_Analytical(T1) - np.mean(contingent_sims))\
                       + optimal_gamma_cv(option_price, list(gamma_cv[i] for i in [0,1,3]), contingent_sims_sq, var_path_1[:, -1], contingent_sims, S_path_1[:, -1]) * (contingent_Analytical_sqr(T1) - np.mean(contingent_sims_sq))\
                       + optimal_gamma_cv(option_price, gamma_cv[:3], S_path_1[:, -1], var_path_1[:, -1], contingent_sims, contingent_sims_sq) * (S0 - np.mean(S_path_1[:, -1]))
        fin_vol_call_6[i] = find_vol_call(mean_price_6, S0, K_call[i], T1)
        print(mean_price, mean_price_2, mean_price_3, mean_price_4)

    fin_vol_put_1 = np.zeros(K_steps)
    fin_vol_put_2 = np.zeros(K_steps)
    fin_vol_put_3 = np.zeros(K_steps)
    fin_vol_put_4 = np.zeros(K_steps)
    fin_vol_put_5 = np.zeros(K_steps)
    fin_vol_put_6 = np.zeros(K_steps)
    for i in range(0, K_steps):
        option_price = np.maximum(-S_path_1[:, -1] + K_put[i], 0)
        mean_price = np.mean(option_price)
        # some confidence interval
        fin_vol_put_1[i] = find_vol_put(mean_price, S0, K_put[i], T1)

        # some confidence interval
        mean_price_2 = mean_price + optimal_gmma(option_price, var_path_1[:, -1]) * (
                    variance_analytic(T1) - np.mean(var_path_1[:, -1]))
        fin_vol_put_2[i] = find_vol_put(mean_price_2, S0, K_put[i], T1)

        contingent_sims = contingent_Sim(T1, dt, var_path_1)
        mean_price_3 = mean_price + optimal_gmma(option_price, contingent_sims) * (contingent_Analytical(T1) - np.mean(contingent_sims))
        fin_vol_put_3[i] = find_vol_put(mean_price_3, S0, K_put[i], T1)

        contingent_sims_sq = contingent_Sim_sqr(T1, dt, var_path_1)
        mean_price_4 = mean_price + optimal_gmma(option_price, contingent_sims_sq) * (
                    contingent_Analytical_sqr(T1) - np.mean(contingent_sims_sq))
        fin_vol_put_4[i] = find_vol_put(mean_price_4, S0, K_put[i], T1)

        mean_price_5 = mean_price + optimal_gmma(option_price, S_path_1[:, -1]) * (
                S0 - np.mean(S_path_1[:, -1]))
        fin_vol_put_5[i] = find_vol_put(mean_price_5, S0, K_put[i], T1)

        gamma_cv = cv_gamma(option_price, var_path_1[:, -1], contingent_sims, contingent_sims_sq, S_path_1[:, -1])

        mean_price_6 = mean_price \
                       + optimal_gamma_cv(option_price, gamma_cv[1:], var_path_1[:, -1], contingent_sims, contingent_sims_sq, S_path_1[:, -1]) * (variance_analytic(T1) - np.mean(var_path_1[:, -1])) \
                       + optimal_gamma_cv(option_price, list(gamma_cv[i] for i in [0,2,3]), contingent_sims, var_path_1[:, -1], contingent_sims_sq, S_path_1[:, -1]) * (contingent_Analytical(T1) - np.mean(contingent_sims))\
                       + optimal_gamma_cv(option_price, list(gamma_cv[i] for i in [0,1,3]), contingent_sims_sq, var_path_1[:, -1], contingent_sims, S_path_1[:, -1]) * (contingent_Analytical_sqr(T1) - np.mean(contingent_sims_sq))\
                       + optimal_gamma_cv(option_price, gamma_cv[:3], S_path_1[:, -1], var_path_1[:, -1], contingent_sims, contingent_sims_sq) * (S0 - np.mean(S_path_1[:, -1]))
        fin_vol_put_6[i] = find_vol_put(mean_price_6, S0, K_put[i], T1)

    plt.figure(1)
    plt.plot(K_call, fin_vol_call_1, label='Risk-Netural Simulated IV_call without control')
    plt.plot(K_put, fin_vol_put_1, label='Risk-Netural Simulated IV_put without control')
    plt.plot(K_call, fin_vol_call_2, label='Control on Variance call')
    plt.plot(K_put, fin_vol_put_2, label='Control on Variance put')
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.title('Set Variance as Control Variable Simulation for T = 0.25')
    plt.legend()
    plt.savefig('2ci_a.jpg')

    # 2ci b
    plt.figure(2)
    plt.plot(K_call, fin_vol_call_1, label='Risk-Netural Simulated IV_call without control')
    plt.plot(K_put, fin_vol_put_1, label='Risk-Netural Simulated IV_put without control')
    plt.plot(K_call, fin_vol_call_3, label='Control on Payoff call')
    plt.plot(K_put, fin_vol_put_3, label='Control on Payoff put')
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.title('Set Payoff as Control Variable Simulation for T = 0.25')
    plt.legend()
    plt.savefig('2ci_b.jpg')

    plt.figure(3)
    plt.plot(K_call, fin_vol_call_1, label='Risk-Netural Simulated IV_call without control')
    plt.plot(K_put, fin_vol_put_1, label='Risk-Netural Simulated IV_put without control')
    plt.plot(K_call, fin_vol_call_4, label='Control on Payoff call')
    plt.plot(K_put, fin_vol_put_4, label='Control on Payoff put')
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.title('Set Payoff Square as Control Variable Simulation for T = 0.25')
    plt.legend()
    plt.savefig('2ci_c.jpg')

    plt.figure(4)
    plt.plot(K_call, fin_vol_call_1, label='Risk-Netural Simulated IV_call without control')
    plt.plot(K_put, fin_vol_put_1, label='Risk-Netural Simulated IV_put without control')
    plt.plot(K_call, fin_vol_call_5, label='Control on Stock call')
    plt.plot(K_put, fin_vol_put_5, label='Control on Stock put')
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.title('Set Stock Price as Control Variable Simulation for T = 0.25')
    plt.legend()
    plt.savefig('2ci_d.jpg')

    plt.figure(5)
    plt.plot(K_call, fin_vol_call_1, label='Risk-Netural Simulated IV_call without control')
    plt.plot(K_put, fin_vol_put_1, label='Risk-Netural Simulated IV_put without control')
    plt.plot(K_call, fin_vol_call_6, label='Multiple Control Variables call')
    plt.plot(K_put, fin_vol_put_6, label='Multiple Control Variables put')
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.title('Multiple Control Variables Simulation for T = 0.25')
    plt.legend()
    plt.savefig('2cii.jpg')
