import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_simulation(seed, a, b, c, d, e, f, g, h, i, max_demand, min_allocation_ratio=0.1, fairness_weight=1):

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
    i=np.random.uniform(0.5, 0.9)

    total_demand_true = np.sum(demand)
    total_supply = total_demand_true * i
   

    base_crops = ['wheat', 'maize', 'sunflower', 'potato']
    base_Ky = [1, 0.9, 1, 1.2]
    crop_max_yield = {
        'wheat': 1.05, 'maize': 1.80, 'sunflower': 0.45, 'potato': 9.01
    }
   

    crop_lists, crop_props, crop_kys, crop_max_yields = [], [], [], []
   
    for _ in range(N):
        num_crops = np.random.randint(1, 4)
        selected_crops = np.random.choice(base_crops, size=num_crops, replace=False)
        props = np.random.dirichlet(np.ones(num_crops))
        kys = [base_Ky[base_crops.index(c)] for c in selected_crops]
        max_ys = [crop_max_yield[c] for c in selected_crops]
        crop_lists.append(selected_crops)
        crop_props.append(props)
        crop_kys.append(kys)
        crop_max_yields.append(max_ys)
   
    Ky = np.array([np.average(crop_kys[i], weights=crop_props[i]) for i in range(N)])
    max_yield_per_acre = np.array([np.average(crop_max_yields[i], weights=crop_props[i]) for i in range(N)])
   

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

    a = np.zeros((N, N))
    for i in range(N):
        neighbors = [(i + j) % N for j in [-2, -1, 0, 1, 2]]
        for j in neighbors:
            a[i, j] = 1 / 5
   

    consensus_steps = 300
    z = np.zeros(N)
    for i in range(N):
        z[i] = demand[i]
   
    ts = np.zeros(N)
    ts[0] = total_supply
   
    for k in range(consensus_steps):
        z_new = np.zeros(N)
        ts_new = np.zeros(N)
        for i in range(N):
            z_new[i] = sum(a[i, j] * z[j] for j in range(N))
            ts_new[i] = sum(a[i, j] * ts[j] for j in range(N))
        z = z_new
        ts = ts_new
   
    total_demand_est = z[0] * N
    total_supply_est = ts[0] * N
    target_avg = total_supply_est / total_demand_est
   

    r = np.zeros(N)
    for i in range(N):
        r[i] = demand[i] / total_demand_est * N
   

    x = np.zeros((N, kstep + 1)) 
    s = np.zeros((N, kstep + 1))
    y = np.zeros((N, kstep + 1))
    f = np.zeros(kstep + 1)
    irrigation_costs = np.zeros(kstep + 1)
    transportation_costs = np.zeros(kstep + 1)
    shortage_costs = np.zeros(kstep + 1)
    fairness_costs = np.zeros(kstep + 1)
    global_constraint_costs = np.zeros(kstep + 1)
   

    def phi(x_val):
        return x_val
   
    def nabla_phi():
        return 1 
   
    def nabla_1(x_val, sigma, r_val, gamma_val, ky, eff_ratio, eff_irr, dro_level):
        effective_x = x_val * eff_ratio * eff_irr
        return 2 * ky * gamma_val * dro_level * (effective_x - r_val) * (eff_ratio * eff_irr) + fairness_weight * 2 * (x_val - sigma)
   
    def nabla_2(x_val, sigma, beta_val, target_avg_val):
        return fairness_weight * 2 * (sigma - x_val) + 2 * beta_val * (sigma - target_avg_val)
   
    def transportation_cost(x_val, i_val):
        return theta * distances[i_val] * x_val
   

    for i in range(N):
        cc = target_avg
        min_alloc = min_allocation_ratio * r[i]
        x[i, 0] = max(r[i] * cc, min_alloc)
        s[i, 0] = phi(x[i, 0])
        y[i, 0] = nabla_2(x[i, 0], s[i, 0], beta, target_avg)
       

        irrigation_efficiency = np.average(irr_effs[i], weights=irr_props[i])
        cost_per_unit_water = np.average(irr_costs[i], weights=irr_costs[i])
       
        effective_x = x[i, 0] * effective_ratio[i] * irrigation_efficiency
        dro_level = (demands_et[i] * 1000 / max_demand) ** 2
       
        shortage_cost = gamma * (dro_level) * Ky[i] * (effective_x - r[i]) ** 2 
        fairness_cost = fairness_weight * (x[i, 0] - s[i, 0]) ** 2
        global_constraint_cost = beta * (s[i, 0] - target_avg) ** 2
        transport_cost = gamma_trans * transportation_cost(x[i, 0], i)
        irrigation_cost_val = gamma_irricost * cost_per_unit_water * x[i, 0]
       
        f[0] += shortage_cost + fairness_cost + global_constraint_cost + transport_cost + irrigation_cost_val
        shortage_costs[0] += shortage_cost
        fairness_costs[0] += fairness_cost
        global_constraint_costs[0] += global_constraint_cost
        transportation_costs[0] += transport_cost
        irrigation_costs[0] += irrigation_cost_val
   

    pi_convergence = 0
    for k in range(kstep):
        x_new = np.zeros(N)
        s_new = np.zeros(N)
        y_new = np.zeros(N)
       
        for i in range(N):

            irrigation_efficiency = np.average(irr_effs[i], weights=irr_props[i])
            dro_level = (demands_et[i] * 1000 / max_demand) ** 2
            grad1 = nabla_1(x[i, k], s[i, k], r[i], gamma, Ky[i], effective_ratio[i], irrigation_efficiency,dro_level)
            y_term = nabla_phi() * y[i, k] 
            x_new[i] = x[i, k] - alpha * (grad1 + y_term)
            min_alloc = min_allocation_ratio * r[i]
            x_new[i] = np.clip(x_new[i], min_alloc, r[i])
           
            s_new[i] = phi(x_new[i]) - phi(x[i, k])
            for j in range(N):
                s_new[i] += a[i, j] * s[j, k]
           
            y_new[i] = nabla_2(x_new[i], s_new[i], beta, target_avg) - nabla_2(x[i, k], s[i, k], beta, target_avg)
            for j in range(N):
                y_new[i] += a[i, j] * y[j, k]
       
        x[:, k + 1] = x_new
        s[:, k + 1] = s_new
        y[:, k + 1] = y_new
       

        f_current = 0
        for i in range(N):
            irrigation_efficiency = np.average(irr_effs[i], weights=irr_props[i])
            cost_per_unit_water = np.average(irr_costs[i], weights=irr_props[i])
           
            effective_x = x_new[i] * effective_ratio[i] * irrigation_efficiency
            dro_level = (demands_et[i] * 1000 / max_demand) ** 2
           
            shortage_cost = gamma * (dro_level) * Ky[i] * (effective_x - r[i]) ** 2
            fairness_cost = fairness_weight * (x_new[i] - s_new[i]) ** 2
            global_constraint_cost = beta * (s_new[i] - target_avg) ** 2
            transport_cost = gamma_trans * transportation_cost(x_new[i], i)
            irrigation_cost_val = gamma_irricost * cost_per_unit_water * x_new[i]
           
            f_current += shortage_cost + fairness_cost + global_constraint_cost + transport_cost + irrigation_cost_val
       
        f[k + 1] = f_current

   
    final_x_pi = x[:, -1] if pi_convergence == 0 else x[:, pi_convergence]
    final_water_pi = final_x_pi * (total_supply_est / np.sum(final_x_pi))

    irrigation_efficiencies = np.array([np.average(irr_effs[i], weights=irr_props[i]) for i in range(N)])
    cost_per_unit_waters = np.array([np.average(irr_costs[i], weights=irr_props[i]) for i in range(N)])
   

    rho_admm = 0.5
    admm_steps = 600
    x_admm = np.zeros(N)
    z_admm = np.zeros(N)
    u_admm = np.zeros(N)
   
    for i in range(N):
        cc = target_avg
        min_alloc = min_allocation_ratio * r[i]
        x_admm[i] = max(r[i] * cc, min_alloc)
        z_admm[i] = x_admm[i]
   
    admm_convergence = 0
    for k in range(admm_steps):
        x_new_admm = x_admm.copy()
       
        for i in range(N):
            dro_level = (demands_et_mm[i] / max_demand) ** 2
            grad_fi = 2 * gamma * dro_level * Ky[i] * (x_admm[i] * effective_ratio[i] * irrigation_efficiencies[i] - r[i]) * (effective_ratio[i] * irrigation_efficiencies[i]) \
        + 2 * beta * (np.sum(x_admm) - N * target_avg) \
        + gamma_trans * theta * distances[i]
            grad_admm = grad_fi + rho_admm * (x_admm[i] - z_admm[i] + u_admm[i])
            x_new_admm[i] -= alpha * grad_admm
            min_alloc = min_allocation_ratio * r[i]
            x_new_admm[i] = np.clip(x_new_admm[i], min_alloc, r[i])
       
        avg_constraint = (np.sum(x_new_admm + u_admm) - N * target_avg) / N
        z_admm_new = x_new_admm + u_admm - avg_constraint
       
        for i in range(N):
            min_alloc = min_allocation_ratio * r[i]
            z_admm_new[i] = np.clip(z_admm_new[i], min_alloc, r[i])
       
        x_admm = x_new_admm
        z_admm = z_admm_new
        u_admm += x_admm - z_admm
       

   
    final_x_admm = z_admm.copy()
    final_water_admm = final_x_admm * (total_supply_est / np.sum(final_x_admm))
   

    x_area = (areas / np.sum(areas)) * target_avg * N
    water_area = x_area * (total_supply_est / np.sum(x_area)) if np.sum(x_area) > 0 else np.zeros(N)
   
    x_demand = r.copy() * target_avg
    water_demand = x_demand * (total_supply_est / np.sum(x_demand))
   


    def calculate_yield(effective_water, demand_water, Ky_vals, max_yield_per_acre_vals, area_vals, max_demand_val):
        yields = np.zeros(len(effective_water))
        for i in range(len(effective_water)):
            eta = effective_water[i] / demand_water[i] if demand_water[i] > 0 else 0
            drought_level = (demands_et_mm[i] / max_demand_val) ** 2
            if eta >= 1.0:
                yields[i] = 1
            else:
                yields[i] = 1.0 - drought_level * Ky_vals[i] * (1 - eta) ** 2
        return yields
   

    effective_water_pi = final_water_pi * effective_ratio * irrigation_efficiencies
    effective_water_admm = final_water_admm * effective_ratio * irrigation_efficiencies
    effective_water_area = water_area * effective_ratio * irrigation_efficiencies
    effective_water_demand = water_demand * effective_ratio * irrigation_efficiencies
   

    water_best = np.zeros(N)
    for i in range(N):
        denom = effective_ratio[i] * irrigation_efficiencies[i]
        water_best[i] = demand[i] / denom if denom > 0 else demand[i]
    effective_best = water_best * effective_ratio * irrigation_efficiencies
   

    best_yield = calculate_yield(effective_best, demand, Ky, max_yield_per_acre, areas, max_demand)
    pi_yield = calculate_yield(effective_water_pi, demand, Ky, max_yield_per_acre, areas, max_demand)
    admm_yield = calculate_yield(effective_water_admm, demand, Ky, max_yield_per_acre, areas, max_demand)
    area_yield = calculate_yield(effective_water_area, demand, Ky, max_yield_per_acre, areas, max_demand)
    demand_yield = calculate_yield(effective_water_demand, demand, Ky, max_yield_per_acre, areas, max_demand)
   

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
    area_loss_t= np.sum(area_loss)
    demand_loss_t = np.sum(demand_loss)
   

    pi_steps = pi_convergence if pi_convergence > 0 else kstep
    admm_steps_converged = admm_convergence if admm_convergence > 0 else admm_steps

    def centralized_objective(x):
        """集中式目标函数：min sum shortage + transport + irr_cost + beta * (mean(x) - target_avg)^2 + fairness_variance"""
        obj = 0
        mean_x = np.mean(x)
        for i in range(N):
            dro_level = (demands_et_mm[i] / max_demand) ** 2
            effective_x = x[i] * effective_ratio[i] * irrigation_efficiencies[i]
            shortage = gamma * dro_level * Ky[i] * (effective_x - r[i]) ** 2
            transport = gamma_trans * theta * distances[i] * x[i]
            irr_cost = gamma_irricost * cost_per_unit_waters[i] * x[i]
            obj += shortage + transport + irr_cost
        obj += beta * (mean_x - target_avg) ** 2
        variance = np.sum((x - mean_x)**2) / N 
        obj += fairness_weight * variance
        return obj
   

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N * target_avg}]
   

    bounds = [(min_allocation_ratio * r[i], r[i]) for i in range(N)]
   

    x0 = x_demand
   

    res = minimize(centralized_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 2000, 'disp': False})
   
    if res.success:
        final_x_central = res.x
        final_water_central = final_x_central * (total_supply_est / np.sum(final_x_central))  # 缩放确保sum water = total_supply_est
        effective_water_central = final_water_central * effective_ratio * irrigation_efficiencies
        central_yield = calculate_yield(effective_water_central, demand, Ky, max_yield_per_acre, areas, max_demand)
        central_absolute_yield = np.sum(central_yield * max_yield_per_acre * areas)
        central_loss = best_yield_vals - (central_yield * max_yield_per_acre * areas)
        central_cv = cv_calc(central_loss)
        central_total_effective = np.sum(effective_water_central)
        central_loss_t = np.sum(central_loss)
        central_steps = 0  # 无迭代
    else:
        print(f"警告: 集中式优化未收敛，使用PI作为fallback")
        final_x_central = final_x_pi
        final_water_central = final_water_pi
        effective_water_central = effective_water_pi
        central_yield = pi_yield
        central_absolute_yield = pi_absolute_yield
        central_loss = pi_loss
        central_cv = pi_cv
        central_total_effective = pi_total_effective
        central_loss_t = pi_loss_t
        central_steps = pi_steps
   
    return {
        'yield_loss': {
            'area': area_loss_t,
            'demand': demand_loss_t,
            'admm': admm_loss_t,
            'pi': pi_loss_t,
            'centralized': central_loss_t
        },
        'yield': {
            'area': area_absolute_yield,
            'demand': demand_absolute_yield,
            'admm': admm_absolute_yield,
            'pi': pi_absolute_yield,
            'centralized': central_absolute_yield
        },
        'cv': {
            'area': area_cv,
            'demand': demand_cv,
            'admm': admm_cv,
            'pi': pi_cv,
            'centralized': central_cv
        },
        'effective_water': {
            'area': area_total_effective,
            'demand': demand_total_effective,
            'admm': admm_total_effective,
            'pi': pi_total_effective,
            'centralized': central_total_effective
        },
        'convergence_steps': {
            'area': 0, # 基准方法无迭代
            'demand': 0, # 基准方法无迭代
            'admm': admm_steps_converged,
            'pi': pi_steps,
            'centralized': central_steps
        }
    }

N_seeds = 100
kstep = 600
alpha = 0.01
gamma = 5
theta = 1
beta = 10
gamma_trans = 5
gamma_irricost = 10
ratio = 0.8
max_de = 100

methods = ['area', 'demand', 'admm', 'pi', 'centralized']
results = {
    'yield_loss': {method: [] for method in methods},
    'yield': {method: [] for method in methods},
    'cv': {method: [] for method in methods},
    'effective_water': {method: [] for method in methods},
    'convergence_steps': {method: [] for method in methods}
}

for seed in range(N_seeds):
    
   

    N_random = np.random.randint(10, 51)
   
    sim_result = run_simulation(
        seed, kstep, alpha, gamma, theta, beta,
        gamma_trans, gamma_irricost, N_random, ratio, max_de
    )
   
    for metric in results.keys():
        for method in methods:
            results[metric][method].append(sim_result[metric][method])

data_list = []
for i in range(N_seeds):
    for method in methods:
        data_list.append({
            'Method': method,
            'Yield_loss': results['yield_loss'][method][i],
            'Yield': results['yield'][method][i],
            'CV': results['cv'][method][i],
            'Effective_Water': results['effective_water'][method][i],
            'Convergence_Steps': results['convergence_steps'][method][i],
            'Seed': i
        })
df = pd.DataFrame(data_list)
df.to_csv('simulation_results1.csv', index=False, encoding='utf-8-sig')