# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


def run_simulation(seed, a, b, c, d, e, f, g, h, i, max_demand,
                   min_allocation_ratio=0.1, fairness_weight=1):
    np.random.seed(seed)

    N = h
    kstep = a
    alpha = b
    gamma = c
    theta = d
    beta = e
    gamma_trans = f
    gamma_irricost = g

    areas_km2 = np.random.uniform(50, 500, size=N)
    areas = areas_km2 * 1000000

    demands_et_mm = np.random.uniform(60, 80, size=N)
    demands_et = demands_et_mm / 1000

    demand = areas * demands_et

    distances = np.random.uniform(20, 200, size=N)
    i = np.random.uniform(0.5, 0.9)

    total_demand_true = np.sum(demand)
    total_supply = total_demand_true * i

    base_crops = ['wheat', 'maize', 'sunflower', 'potato']
    base_Ky = [1, 0.9, 1, 1.2]
    crop_max_yield = {'wheat': 1.05, 'maize': 1.80, 'sunflower': 0.45, 'potato': 9.01}

    crop_lists, crop_props, crop_kys, crop_max_yields = [], [], [], []
    for _ in range(N):
        num_crops = np.random.randint(1, 4)
        selected_crops = np.random.choice(base_crops, size=num_crops, replace=False)
        props = np.random.dirichlet(np.ones(num_crops))
        kys = [base_Ky[base_crops.index(cc)] for cc in selected_crops]
        max_ys = [crop_max_yield[cc] for cc in selected_crops]
        crop_lists.append(selected_crops)
        crop_props.append(props)
        crop_kys.append(kys)
        crop_max_yields.append(max_ys)

    Ky = np.array([np.average(crop_kys[t], weights=crop_props[t]) for t in range(N)])
    max_yield_per_acre = np.array([np.average(crop_max_yields[t], weights=crop_props[t]) for t in range(N)])

    irrigation_methods = ['traditional', 'sprinkler', 'drip']
    irrigation_params = {
        'traditional': {'efficiency': 0.30, 'cost_per_m3': 0.5},
        'sprinkler': {'efficiency': 0.70, 'cost_per_m3': 0.2},
        'drip': {'efficiency': 0.90, 'cost_per_m3': 0.1}
    }
    irr_lists, irr_props, irr_effs, irr_costs = [], [], [], []
    for _ in range(N):
        num_irr = np.random.randint(1, 3)
        selected_irr = np.random.choice(irrigation_methods, size=num_irr, replace=False)
        props = np.random.dirichlet(np.ones(num_irr))
        effs = [irrigation_params[m]['efficiency'] for m in selected_irr]
        costs = [irrigation_params[m]['cost_per_m3'] for m in selected_irr]
        irr_lists.append(selected_irr)
        irr_props.append(props)
        irr_effs.append(effs)
        irr_costs.append(costs)

    max_distance = np.max(distances)
    max_loss_rate = 0.1
    loss_rate = (distances / max_distance) * max_loss_rate
    effective_ratio = 1 - loss_rate

    adj = np.zeros((N, N))
    for t in range(N):
        neighbors = [(t + j) % N for j in [-2, -1, 0, 1, 2]]
        for j in neighbors:
            adj[t, j] = 1 / 5

    consensus_steps = 300
    z = np.zeros(N)
    for t in range(N):
        z[t] = demand[t]
    ts = np.zeros(N)
    ts[0] = total_supply
    for k in range(consensus_steps):
        z_new = np.zeros(N)
        ts_new = np.zeros(N)
        for t in range(N):
            z_new[t] = sum(adj[t, j] * z[j] for j in range(N))
            ts_new[t] = sum(adj[t, j] * ts[j] for j in range(N))
        z = z_new
        ts = ts_new

    total_demand_est = z[0] * N
    total_supply_est = ts[0] * N
    target_avg = total_supply_est / total_demand_est

    r = np.zeros(N)
    for t in range(N):
        r[t] = demand[t] / total_demand_est * N

    x = np.zeros((N, kstep + 1))
    s = np.zeros((N, kstep + 1))
    y = np.zeros((N, kstep + 1))
    f = np.zeros(kstep + 1)

    def phi(x_val):
        return x_val

    def nabla_1(x_val, sigma, r_val, gamma_val, ky, eff_ratio, eff_irr, dro_level):
        effective_x = x_val * eff_ratio * eff_irr
        return (2 * ky * gamma_val * dro_level * (effective_x - r_val) * (eff_ratio * eff_irr)
                + fairness_weight * 2 * (x_val - sigma))

    def nabla_2(x_val, sigma, beta_val, target_avg_val):
        return fairness_weight * 2 * (sigma - x_val) + 2 * beta_val * (sigma - target_avg_val)

    def transportation_cost(x_val, i_val):
        return theta * distances[i_val] * x_val

    for t in range(N):
        x[t, 0] = max(r[t] * target_avg, min_allocation_ratio * r[t])
        s[t, 0] = phi(x[t, 0])
        y[t, 0] = nabla_2(x[t, 0], s[t, 0], beta, target_avg)

        irrigation_efficiency = np.average(irr_effs[t], weights=irr_props[t])
        cost_per_unit_water = np.average(irr_costs[t], weights=irr_costs[t])

        effective_x = x[t, 0] * effective_ratio[t] * irrigation_efficiency
        dro_level = (demands_et[t] * 1000 / max_demand) ** 2
        shortage_cost = gamma * dro_level * Ky[t] * (effective_x - r[t]) ** 2
        fairness_cost = fairness_weight * (x[t, 0] - s[t, 0]) ** 2
        global_constraint_cost = beta * (s[t, 0] - target_avg) ** 2
        transport_cost = gamma_trans * transportation_cost(x[t, 0], t)
        irrigation_cost_val = gamma_irricost * cost_per_unit_water * x[t, 0]
        f[0] += (shortage_cost + fairness_cost + global_constraint_cost
                 + transport_cost + irrigation_cost_val)

    pi_convergence = 0
    for k in range(kstep):
        x_new = np.zeros(N)
        s_new = np.zeros(N)
        y_new = np.zeros(N)

        for t in range(N):
            irrigation_efficiency = np.average(irr_effs[t], weights=irr_props[t])
            dro_level = (demands_et[t] * 1000 / max_demand) ** 2
            grad1 = nabla_1(x[t, k], s[t, k], r[t], gamma, Ky[t],
                            effective_ratio[t], irrigation_efficiency, dro_level)
            y_term = y[t, k]
            x_new[t] = x[t, k] - alpha * (grad1 + y_term)
            min_alloc = min_allocation_ratio * r[t]
            x_new[t] = np.clip(x_new[t], min_alloc, r[t])

            s_new[t] = phi(x_new[t]) - phi(x[t, k])
            for j in range(N):
                s_new[t] += adj[t, j] * s[j, k]

            y_new[t] = nabla_2(x_new[t], s_new[t], beta, target_avg) - nabla_2(x[t, k], s[t, k], beta, target_avg)
            for j in range(N):
                y_new[t] += adj[t, j] * y[j, k]

        x[:, k + 1] = x_new
        s[:, k + 1] = s_new
        y[:, k + 1] = y_new

        f_current = 0
        for t in range(N):
            irrigation_efficiency = np.average(irr_effs[t], weights=irr_props[t])
            cost_per_unit_water = np.average(irr_costs[t], weights=irr_costs[t])
            effective_x = x_new[t] * effective_ratio[t] * irrigation_efficiency
            dro_level = (demands_et[t] * 1000 / max_demand) ** 2
            shortage_cost = gamma * dro_level * Ky[t] * (effective_x - r[t]) ** 2
            fairness_cost = fairness_weight * (x_new[t] - s_new[t]) ** 2
            global_constraint_cost = beta * (s_new[t] - target_avg) ** 2
            transport_cost = gamma_trans * transportation_cost(x_new[t], t)
            irrigation_cost_val = gamma_irricost * cost_per_unit_water * x_new[t]
            f_current += (shortage_cost + fairness_cost + global_constraint_cost
                          + transport_cost + irrigation_cost_val)
        f[k + 1] = f_current

    final_x_pi = x[:, -1]
    final_s_pi = s[:, -1]                      
    final_water_pi = final_x_pi * (total_supply_est / np.sum(final_x_pi))

    irrigation_efficiencies = np.array([np.average(irr_effs[t], weights=irr_props[t]) for t in range(N)])
    cost_per_unit_waters = np.array([np.average(irr_costs[t], weights=irr_props[t]) for t in range(N)])

    # ---------------- ADMM ----------------
    rho_admm = 0.5
    admm_steps = 600
    x_admm = np.zeros(N)
    z_admm = np.zeros(N)
    u_admm = np.zeros(N)
    for t in range(N):
        x_admm[t] = max(r[t] * target_avg, min_allocation_ratio * r[t])
        z_admm[t] = x_admm[t]

    for k in range(admm_steps):
        x_new_admm = x_admm.copy()
        for t in range(N):
            dro_level = (demands_et_mm[t] / max_demand) ** 2
            grad_fi = (2 * gamma * dro_level * Ky[t]
                       * (x_admm[t] * effective_ratio[t] * irrigation_efficiencies[t] - r[t])
                       * (effective_ratio[t] * irrigation_efficiencies[t])
                       + 2 * beta * (np.sum(x_admm) - N * target_avg)
                       + gamma_trans * theta * distances[t])
            grad_admm = grad_fi + rho_admm * (x_admm[t] - z_admm[t] + u_admm[t])
            x_new_admm[t] -= alpha * grad_admm
            min_alloc = min_allocation_ratio * r[t]
            x_new_admm[t] = np.clip(x_new_admm[t], min_alloc, r[t])

        avg_constraint = (np.sum(x_new_admm + u_admm) - N * target_avg) / N
        z_admm_new = x_new_admm + u_admm - avg_constraint
        for t in range(N):
            min_alloc = min_allocation_ratio * r[t]
            z_admm_new[t] = np.clip(z_admm_new[t], min_alloc, r[t])

        x_admm = x_new_admm
        z_admm = z_admm_new
        u_admm += x_admm - z_admm

    final_x_admm = z_admm.copy()
    final_water_admm = final_x_admm * (total_supply_est / np.sum(final_x_admm))

    
    x_area = (areas / np.sum(areas)) * target_avg * N
    water_area = x_area * (total_supply_est / np.sum(x_area))
    x_demand = r.copy() * target_avg
    water_demand = x_demand * (total_supply_est / np.sum(x_demand))


    def calculate_yield(effective_water, demand_water, Ky_vals, areas_vals, areas_km2_vals=None):
        yields = np.zeros(len(effective_water))
        for t in range(len(effective_water)):
            eta = effective_water[t] / demand_water[t] if demand_water[t] > 0 else 0
            drought_level = (demands_et_mm[t] / max_demand) ** 2
            if eta >= 1.0:
                yields[t] = 1
            else:
                yields[t] = 1.0 - drought_level * Ky_vals[t] * (1 - eta) ** 2
        return yields

    effective_water_pi = final_water_pi * effective_ratio * irrigation_efficiencies
    effective_water_admm = final_water_admm * effective_ratio * irrigation_efficiencies
    effective_water_area = water_area * effective_ratio * irrigation_efficiencies
    effective_water_demand = water_demand * effective_ratio * irrigation_efficiencies

    water_best = np.zeros(N)
    for t in range(N):
        denom = effective_ratio[t] * irrigation_efficiencies[t]
        water_best[t] = demand[t] / denom if denom > 0 else demand[t]
    effective_best = water_best * effective_ratio * irrigation_efficiencies

    best_yield = calculate_yield(effective_best, demand, Ky, max_yield_per_acre, areas)
    pi_yield = calculate_yield(effective_water_pi, demand, Ky, max_yield_per_acre, areas)
    admm_yield = calculate_yield(effective_water_admm, demand, Ky, max_yield_per_acre, areas)
    area_yield = calculate_yield(effective_water_area, demand, Ky, max_yield_per_acre, areas)
    demand_yield = calculate_yield(effective_water_demand, demand, Ky, max_yield_per_acre, areas)

    best_absolute_yield = np.sum(best_yield * max_yield_per_acre * areas)
    pi_absolute_yield = np.sum(pi_yield * max_yield_per_acre * areas)
    admm_absolute_yield = np.sum(admm_yield * max_yield_per_acre * areas)
    area_absolute_yield = np.sum(area_yield * max_yield_per_acre * areas)
    demand_absolute_yield = np.sum(demand_yield * max_yield_per_acre * areas)

    best_yield_vals = best_yield * max_yield_per_acre * areas
    pi_loss = best_yield_vals - (pi_yield * max_yield_per_acre * areas)
    admm_loss = best_yield_vals - (admm_yield * max_yield_per_acre * areas)
    area_loss = best_yield_vals - (area_yield * max_yield_per_acre * areas)
    demand_loss = best_yield_vals - (demand_yield * max_yield_per_acre * areas)

    def cv_calc(loss_array):
        if np.mean(loss_array) == 0:
            return 0
        return np.std(loss_array) / np.mean(loss_array)

    pi_cv = cv_calc(pi_loss)
    admm_cv = cv_calc(admm_loss)
    area_cv = cv_calc(area_loss)
    demand_cv = cv_calc(demand_loss)

    pi_total_effective = np.sum(effective_water_pi)
    admm_total_effective = np.sum(effective_water_admm)
    area_total_effective = np.sum(effective_water_area)
    demand_total_effective = np.sum(effective_water_demand)

    pi_loss_t = np.sum(pi_loss)
    admm_loss_t = np.sum(admm_loss)
    area_loss_t = np.sum(area_loss)
    demand_loss_t = np.sum(demand_loss)

    
    def centralized_objective_old(xv):
        obj = 0
        mean_x = np.mean(xv)
        for t in range(N):
            dro_level = (demands_et_mm[t] / max_demand) ** 2
            effective_x = xv[t] * effective_ratio[t] * irrigation_efficiencies[t]
            shortage = gamma * dro_level * Ky[t] * (effective_x - r[t]) ** 2
            transport = gamma_trans * theta * distances[t] * xv[t]
            irr_cost = gamma_irricost * cost_per_unit_waters[t] * xv[t]
            obj += shortage + transport + irr_cost
        obj += beta * (mean_x - target_avg) ** 2
        variance = np.sum((xv - mean_x) ** 2) / N
        obj += fairness_weight * variance
        return obj

    _bounds_old = [(min_allocation_ratio * r[t], r[t]) for t in range(N)]
    _res_old = minimize(centralized_objective_old, x_demand, method='SLSQP',
                        bounds=_bounds_old,
                        constraints=[{'type': 'eq',
                                      'fun': lambda xv: np.sum(xv) - N * target_avg}],
                        options={'maxiter': 2000, 'disp': False})
    _xc_old = _res_old.x if _res_old.success else final_x_pi
    _wc_old = _xc_old * (total_supply_est / np.sum(_xc_old))
    _ew_old = _wc_old * effective_ratio * irrigation_efficiencies
    _yc_old = calculate_yield(_ew_old, demand, Ky, max_yield_per_acre, areas)
    central_old_absolute_yield = np.sum(_yc_old * max_yield_per_acre * areas)
    central_old_loss = best_yield_vals - (_yc_old * max_yield_per_acre * areas)
    central_old_cv = cv_calc(central_old_loss)
    central_old_total_effective = np.sum(_ew_old)
    central_old_loss_t = np.sum(central_old_loss)

  
    e_all = effective_ratio * irrigation_efficiencies
    dro_all = (demands_et_mm / max_demand) ** 2
    cost_per_unit_waters_pi = np.array(
        [np.average(irr_costs[t], weights=irr_costs[t]) for t in range(N)]) 

    def pi_potential(xv):
        m = np.mean(xv)
        return (np.sum(gamma * dro_all * Ky * (xv * e_all - r) ** 2)
                + fairness_weight * np.sum((xv - m) ** 2)
                + beta * N * (m - target_avg) ** 2)

    def pi_potential_grad(xv):
        m = np.mean(xv)
        return (2 * gamma * dro_all * Ky * (xv * e_all - r) * e_all
                + fairness_weight * 2 * (xv - m)
                + 2 * beta * (m - target_avg))

    def cost_report(xv, sv=None):
    
        s_vec = np.full(N, np.mean(xv)) if sv is None else np.asarray(sv)
        return (np.sum(gamma * dro_all * Ky * (xv * e_all - r) ** 2)
                + fairness_weight * np.sum((xv - s_vec) ** 2)
                + beta * np.sum((s_vec - target_avg) ** 2)
                + np.sum(gamma_trans * theta * distances * xv)
                + np.sum(gamma_irricost * cost_per_unit_waters_pi * xv))

    bounds = [(min_allocation_ratio * r[t], r[t]) for t in range(N)]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    starts = [x_demand, final_x_pi, x_area,
              np.clip(np.full(N, target_avg), lo, hi),
              np.clip(lo, lo, hi)]
    best_x, best_val = None, np.inf
    for x0 in starts:
        res = minimize(pi_potential, np.clip(x0, lo, hi), jac=pi_potential_grad,
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x
    final_x_central = np.clip(best_x, lo, hi)
    final_water_central = final_x_central * (total_supply_est / np.sum(final_x_central))
    effective_water_central = final_water_central * effective_ratio * irrigation_efficiencies
    central_yield = calculate_yield(effective_water_central, demand, Ky, max_yield_per_acre, areas)
    central_absolute_yield = np.sum(central_yield * max_yield_per_acre * areas)
    central_loss = best_yield_vals - (central_yield * max_yield_per_acre * areas)
    central_cv = cv_calc(central_loss)
    central_total_effective = np.sum(effective_water_central)
    central_loss_t = np.sum(central_loss)


    import numpy.fft as _fft
    _k_nbr = 2
    _ker = np.zeros(N)
    for _d in range(-_k_nbr, _k_nbr + 1):
        _ker[_d % N] += 1.0 / (2 * _k_nbr + 1)
    NC_T = 512          
    NC_K = 2000       
    _fk = _fft.fft(_ker) ** NC_T
    x_nc = np.clip(r * target_avg, lo, hi)
    e_nc = effective_ratio * irrigation_efficiencies
    dro_nc = (demands_et_mm / max_demand) ** 2
    w_nc = gamma * dro_nc * Ky
    for _kk in range(NC_K):
        _v = np.real(_fft.ifft(_fft.fft(x_nc) * _fk))
        _g = (2 * w_nc * (x_nc * e_nc - r) * e_nc
              + fairness_weight * 2 * (x_nc - _v)
              + 2 * beta * (_v - target_avg))
        x_nc = np.clip(x_nc - alpha * _g, lo, hi)
    final_x_npga = np.clip(x_nc, lo, hi)
    final_water_npga = final_x_npga * (total_supply_est / np.sum(final_x_npga))
    effective_water_npga = final_water_npga * effective_ratio * irrigation_efficiencies
    npga_yield = calculate_yield(effective_water_npga, demand, Ky, max_yield_per_acre, areas)
    npga_absolute_yield = np.sum(npga_yield * max_yield_per_acre * areas)
    npga_loss = best_yield_vals - (npga_yield * max_yield_per_acre * areas)
    npga_cv = cv_calc(npga_loss)
    npga_total_effective = np.sum(effective_water_npga)
    npga_loss_t = np.sum(npga_loss)

    cost_all = {
        'area': cost_report(x_area),
        'demand': cost_report(x_demand),
        'admm': cost_report(final_x_admm),
        'pi': cost_report(final_x_pi, final_s_pi),
        'centralized_old': cost_report(_xc_old),
        'centralized': cost_report(final_x_central),
        'npga': cost_report(final_x_npga),
    }
  
    obj_all = {m: pi_potential(v) for m, v in
               [('area', x_area), ('demand', x_demand), ('admm', final_x_admm),
                ('pi', final_x_pi), ('centralized_old', _xc_old),
                ('centralized', final_x_central), ('npga', final_x_npga)]}

    return {
        'cost': cost_all,
        'obj': obj_all,
        'yield_loss': {'area': area_loss_t, 'demand': demand_loss_t, 'admm': admm_loss_t,
                       'pi': pi_loss_t, 'centralized_old': central_old_loss_t,
                       'centralized': central_loss_t, 'npga': npga_loss_t},
        'yield': {'area': area_absolute_yield, 'demand': demand_absolute_yield,
                  'admm': admm_absolute_yield, 'pi': pi_absolute_yield,
                  'centralized_old': central_old_absolute_yield,
                  'centralized': central_absolute_yield, 'npga': npga_absolute_yield},
        'cv': {'area': area_cv, 'demand': demand_cv, 'admm': admm_cv, 'pi': pi_cv,
               'centralized_old': central_old_cv, 'centralized': central_cv, 'npga': npga_cv},
        'effective_water': {'area': area_total_effective, 'demand': demand_total_effective,
                            'admm': admm_total_effective, 'pi': pi_total_effective,
                            'centralized_old': central_old_total_effective,
                            'centralized': central_total_effective, 'npga': npga_total_effective},
        'ideal_yield': best_absolute_yield,
    }


if __name__ == '__main__':
    import sys
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    kstep, alpha, gamma, theta = 600, 0.01, 5, 1
    beta, gamma_trans, gamma_irricost = 10, 5, 10
    ratio, max_de = 0.8, 100

    methods = ['area', 'demand', 'admm', 'pi', 'centralized_old', 'centralized', 'npga']
    keys = ['yield_loss', 'yield', 'cv', 'effective_water', 'cost', 'obj']
    results = {k: {m: [] for m in methods} for k in keys}
    ideal = []

    np.random.seed(0)
    for seed in range(n_seeds):
        N_random = np.random.randint(10, 51)
        if (seed + 1) % 10 == 0:
            print(' progress', seed + 1, '/', n_seeds, flush=True)
        sim = run_simulation(seed, kstep, alpha, gamma, theta, beta,
                             gamma_trans, gamma_irricost, N_random, ratio, max_de)
        ideal.append(sim['ideal_yield'])
        for k in keys:
            for m in methods:
                results[k][m].append(sim[k][m])

    rows = []
    for i in range(n_seeds):
        for m in methods:
            rows.append({'Method': m,
                         'Yield_loss': results['yield_loss'][m][i],
                         'Yield': results['yield'][m][i],
                         'CV': results['cv'][m][i],
                         'Effective_Water': results['effective_water'][m][i],
                         'Cost': results['cost'][m][i],
                         'Obj': results['obj'][m][i],
                         'Seed': i})
    df = pd.DataFrame(rows)
    out = 'four_methods_results.csv'
    try:
        df.to_csv(out, index=False, encoding='utf-8-sig')
    except PermissionError:
        out = 'four_methods_results_new.csv'
        df.to_csv(out, index=False, encoding='utf-8-sig')



    focus = ['pi', 'npga', 'admm', 'centralized']
    fdf = df[df.Method.isin(focus)].copy()
    fmean = fdf.groupby('Method').agg(
        Yield_loss=('Yield_loss', 'mean'), CV=('CV', 'mean'),
        Effective_Water=('Effective_Water', 'mean'), Obj=('Obj', 'mean')).reindex(focus)

    fmean.to_csv('four_methods_summary.csv', encoding='utf-8-sig')


