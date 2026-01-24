import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors



def hanshu(a, b, c, d, e, f, g, h, i, max_demand, min_allocation_ratio=0.1, fairness_weight=1):

    N = h
    n = 1

    area_data = [
        (14, 40.1120919,),
        (1, 46.92357921),
        (2, 385.9842806),
        (3, 208.1287788),
        (19, 824.9467958),
        (4, 223.2654172),
        (21, 438.2056833),
        (16, 22.70495768),
        (22, 214.9402661),
        (13, 613.0338574),
        (15, 361.7656591),
        (5, 311.0579202),
        (18, 679.6350666),
        (6, 359.4951633),
        (7, 372.361306),
        (8, 402.6345829),
        (20, 332.2492141),
        (10, 14.37980653),
        (9, 12.10931076),
        (12, 12.86614269),
        (11, 231.5905684),
        (17, 62.06021766),
        (23, 108.226965)
    ]
   

    areas_df = pd.DataFrame(area_data, columns=['Region Name', 'Croplands_Area'])
   
    # Sort by region number
    areas_df = areas_df.sort_values('Region Name').reset_index(drop=True)
   
    # Create dictionary mapping region number to area
    area_dict = dict(zip(areas_df['Region Name'], areas_df['Croplands_Area']))
   
    # Extract area array
    areas = areas_df['Croplands_Area'].values*1000000
    demand_data = [
        (1, 75.85152),
        (2, 80.40801),
        (3, 79.65585),
        (4, 85.19045),
        (5, 83.12192),
        (6, 76.31368),
        (7, 86.85813),
        (8, 81.19632),
        (9, 53.47567),
        (10, 52.72133),
        (11, 66.70147),
        (12, 52.32854),
        (13, 83.75077),
        (14, 76.52269),
        (15, 81.75228),
        (16, 69.42101),
        (17, 93.77501),
        (18, 77.28229),
        (19, 80.41541),
        (20, 81.57736),
        (21, 79.19909),
        (22, 83.08717),
        (23, 84.56689)
    ]

    demand_df = pd.DataFrame(demand_data, columns=['Region Name', 'Water Demand'])
   

    demand_df = demand_df.sort_values('Region Name').reset_index(drop=True)
   

    demand_dict = dict(zip(demand_df['Region Name'], demand_df['Water Demand']))
   

    demands_et = demand_df['Water Demand'].values/1000

    demand = areas[:N] *  demands_et[:N] 

   

    total_demand_true = np.sum(demand) 
    total_supply = total_demand_true * i 


    kstep = a
    alpha = b
    gamma = c
    theta = d
    beta = e 
    gamma_trans = f 
    gamma_irricost = g 
   

    base_crops = ['wheat', 'maize', 'sunflower', 'potato']
    base_Ky = [1.05,1.25, 0.95, 1.1]

    crop_lists = [] 
    crop_props = [] 
    crop_kys = [] 
    crop_max_yields = []
    crop_max_yield = {
        'wheat': 1.05, 
        'maize': 1.80,
        'sunflower': 0.45,
        'potato': 9.01
    }
    for _ in range(N):
        num_crops = np.random.randint(1, 4)  # 1-3 crops
        selected_crops = np.random.choice(base_crops, size=num_crops, replace=False)
        props = np.random.dirichlet(np.ones(num_crops))  # proportions sum to 1
        kys = [base_Ky[base_crops.index(c)] for c in selected_crops]
        max_ys = [crop_max_yield[c] for c in selected_crops]
        crop_lists.append(selected_crops)
        crop_props.append(props)
        crop_kys.append(kys)
        crop_max_yields.append(max_ys)

    Ky = np.array([np.average(crop_kys[i], weights=crop_props[i]) for i in range(N)])

    max_yield_per_acre = np.array([np.average(crop_max_yields[i], weights=crop_props[i]) for i in range(N)])

   
    # Irrigation method (randomly assigned for each node)
    irrigation_methods = ['traditional', 'sprinkler', 'drip']
    irrigation_params = {
        'traditional': {'name': 'Traditional Flood', 'cost_per_m3': 0.5, 'efficiency': 0.30},
        'sprinkler': {'name': 'Sprinkler', 'cost_per_m3': 0.2, 'efficiency': 0.70},
        'drip': {'name': 'Drip', 'cost_per_m3': 0.1, 'efficiency': 0.90}
    }

    irr_lists = [] 
    irr_props = [] 
    irr_effs = []  
    irr_costs = []  
    for _ in range(N):
        num_irr = np.random.randint(1, 3)  # 1-2 methods
        selected_irr = np.random.choice(irrigation_methods, size=num_irr, replace=False)
        props = np.random.dirichlet(np.ones(num_irr))  # proportions sum to 1
        effs = [irrigation_params[m]['efficiency'] for m in selected_irr]
        costs = [irrigation_params[m]['cost_per_m3'] for m in selected_irr]
        irr_lists.append(selected_irr)
        irr_props.append(props)
        irr_effs.append(effs)
        irr_costs.append(costs)

    for i in range(N):
        crop_str = ", ".join([f"{crop_lists[i][j]} (proportion:{crop_props[i][j]:.2f}, Ky:{crop_kys[i][j]:.2f})" for j in range(len(crop_lists[i]))])
        irr_str = ", ".join([f"{irrigation_params[irr_lists[i][k]]['name']} (proportion:{irr_props[i][k]:.2f}, cost:{irr_costs[i][k]:.2f}CNY/m³, efficiency:{irr_effs[i][k]:.2f})" for k in range(len(irr_lists[i]))])
        print(f"Region {i+1}: Crops={crop_str}; Irrigation methods={irr_str}")
   

    a = np.zeros((N, N))
    for i in range(N):
        neighbors = [(i+j) % N for j in [-2, -1, 0, 1, 2]] # 5 neighbors (including self)
        for j in neighbors:
            a[i, j] = 1/5

    
    distance_data = [
        (14, 105.8,),
        (1, 112.55),
        (2, 119.5),
        (3, 98.21),
        (19, 142.76),
        (4, 112.97),
        (21, 131.85),
        (16, 148.73),
        (22, 142.48),
        (13, 80.62),
        (15, 76.89),
        (5, 61.4),
        (18, 97.66),
        (6, 50.12),
        (7, 43.7),
        (8, 63.26),
        (20, 193.96),
        (10, 45.8),
        (9, 30.73),
        (12, 42.33),
        (11, 23.56),
        (17, 170.15),
        (23, 158.55)
    ]
   

    distance_df = pd.DataFrame(distance_data, columns=['Region Name', 'Distance'])
   

    distance_df = distance_df.sort_values('Region Name').reset_index(drop=True)
   

    distance_dict = dict(zip(distance_df['Region Name'], distance_df['Distance']))

    distances = distance_df['Distance'].values
    max_distance = np.max(distances)
    max_loss_rate = 0.1
    loss_rate = (distances / max_distance) * max_loss_rate
    effective_ratio = 1 - loss_rate

    consensus_steps = 300 # increase to accommodate larger N

    z_trajectory = np.zeros((N, consensus_steps + 1))
    ts_trajectory = np.zeros((N, consensus_steps + 1))
   

    z = np.zeros(N)
    for i in range(N):
        z[i] = demand[i]
    z_trajectory[:, 0] = z.copy() 
   

    ts = np.zeros(N)
    ts[0] = total_supply
    ts_trajectory[:, 0] = ts.copy() 
   
    for k in range(consensus_steps):
        z_new = np.zeros(N)
        ts_new = np.zeros(N)
        for i in range(N):
            z_new[i] = sum(a[i, j] * z[j] for j in range(N))
            ts_new[i] = sum(a[i, j] * ts[j] for j in range(N))
        z = z_new
        ts = ts_new
        z_trajectory[:, k + 1] = z.copy()
        ts_trajectory[:, k + 1] = ts.copy() 

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
   

    def phi(x):
        return x
   
    def nabla_phi():
        return 1

    def nabla_1(x_val, sigma, r_val, gamma_val, ky, eff_ratio, eff_irr, dro_level):
        effective_x = x_val * eff_ratio * eff_irr 
        return 2 * ky * gamma_val * dro_level * (effective_x - r_val) * (eff_ratio * eff_irr) + fairness_weight * 2 * (x_val - sigma)
   
    def nabla_2(x, sigma, beta, target_avg):
        return fairness_weight * 2 * (sigma - x) + 2 * beta * (sigma - target_avg)
   
    def transportation_cost(x, i):
        return theta*distances[i] * x

    for i in range(N):
        cc = target_avg
        min_alloc = min_allocation_ratio * r[i]
        x[i, 0] = max(r[i] * cc, min_alloc) 
        s[i, 0] = phi(x[i, 0])
        y[i, 0] = nabla_2(x[i, 0], s[i, 0], beta, target_avg)
       
        irrigation_efficiency = np.average(irr_effs[i], weights=irr_props[i])
        cost_per_unit_water = np.average(irr_costs[i], weights=irr_props[i])
       
        effective_x = x[i, 0] * effective_ratio[i] * irrigation_efficiency
        dro_level=(demands_et[i]*1000/max_demand)**2
       
        shortage_cost = gamma *(dro_level)* Ky[i] * (effective_x - r[i])**2 # simplified norm to square (n=1)
        fairness_cost = fairness_weight * (x[i, 0] - s[i, 0])**2
        global_constraint_cost = beta * (s[i, 0] - target_avg)**2
        transport_cost = gamma_trans * transportation_cost(x[i, 0], i)
        irrigation_cost_val = gamma_irricost * cost_per_unit_water * x[i, 0]
       
        f[0] += shortage_cost + fairness_cost + global_constraint_cost + transport_cost + irrigation_cost_val
        shortage_costs[0] += shortage_cost
        fairness_costs[0] += fairness_cost
        global_constraint_costs[0] += global_constraint_cost
        transportation_costs[0] += transport_cost
        irrigation_costs[0] += irrigation_cost_val

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
       
        f[k + 1] = 0
        irrigation_costs[k + 1] = 0
        transportation_costs[k + 1] = 0
        shortage_costs[k + 1] = 0
        fairness_costs[k + 1] = 0
        global_constraint_costs[k + 1] = 0
       
        for i in range(N):

            irrigation_efficiency = np.average(irr_effs[i], weights=irr_props[i])
            cost_per_unit_water = np.average(irr_costs[i], weights=irr_props[i])
           
            effective_x = x[i, k + 1] * effective_ratio[i] * irrigation_efficiency
            dro_level=(demands_et[i]*1000/max_demand)**2
           
            shortage_cost = gamma * Ky[i]*(dro_level) * (effective_x - r[i])**2
            fairness_cost = fairness_weight * (x[i, k + 1] - s[i, k + 1])**2
            global_constraint_cost = beta * (s[i, k + 1] - target_avg)**2
            transport_cost = gamma_trans * transportation_cost(x[i, k + 1], i)
            irrigation_cost_val = gamma_irricost * cost_per_unit_water * x[i, k + 1]
           
            f[k + 1] += shortage_cost + fairness_cost + global_constraint_cost + transport_cost*0.1 + irrigation_cost_val
            shortage_costs[k + 1] += shortage_cost
            fairness_costs[k + 1] += fairness_cost
            global_constraint_costs[k + 1] += global_constraint_cost
            transportation_costs[k + 1] += transport_cost
            irrigation_costs[k + 1] += irrigation_cost_val
       
   
    
   

    initial_x = x[:, 0]
    final_x = x[:, -1]
    sum_initial_x = np.sum(initial_x)
    sum_final_x = np.sum(final_x)
    initial_water = initial_x * (total_supply_est / sum_initial_x) if sum_initial_x > 0 else np.zeros(N)
    final_water = final_x * (total_supply_est / sum_final_x) if sum_final_x > 0 else np.zeros(N)

   
    initial_effective_water = np.zeros(N)
    final_effective_water = np.zeros(N)
    for i in range(N):
        irrigation_efficiency = np.average(irr_effs[i], weights=irr_props[i])  # weighted efficiency
        initial_effective_water[i] = initial_water[i] * irrigation_efficiency# * effective_ratio[i]
        final_effective_water[i] = final_water[i] * irrigation_efficiency#* effective_ratio[i]

    x_best = r
    water_best = np.zeros(N)
    for i in range(N):
        irrigation_efficiency = np.average(irr_effs[i], weights=irr_props[i])
        denom = irrigation_efficiency#effective_ratio[i] *
        water_best[i] = demand[i] / denom if denom > 0 else demand[i]
    effective_best = water_best * np.array([np.average(irr_effs[i], weights=irr_props[i]) for i in range(N)])# * effective_ratio

    x_area = (areas[:N] / np.sum(areas[:N])) * target_avg * N
    sum_x_area = np.sum(x_area)
    water_area = x_area * (total_supply_est / sum_x_area) if sum_x_area > 0 else np.zeros(N)
    effective_area = np.zeros(N)
    for i in range(N):
        irrigation_efficiency = np.average(irr_effs[i], weights=irr_props[i])
        effective_area[i] = water_area[i] * irrigation_efficiency# * effective_ratio[i]
   
    # ==============================
    # FAO-56 
    # ==============================
    def calculate_yield(effective, demand,ky):#, crop_type, stage

        yields = np.zeros(len(effective))

        for i in range(len(effective)):
            eta = effective[i] / demand[i] if demand[i] > 0 else 0
            drought_level=(demands_et[i]*1000/max_demand)**2
            if eta >= 1.0:
                yields[i] = 1 # sufficient irrigation
            else:
                yields[i] =1.0 - drought_level*ky[i] * (1 - eta)**2 # severe deficit quadratic deduction
        return yields
   
   

    best_yield = calculate_yield(effective_best, demand, Ky)
    area_yield = calculate_yield(effective_area, demand, Ky)
    initial_yield = calculate_yield(initial_effective_water, demand, Ky)
    final_yield = calculate_yield(final_effective_water, demand, Ky)

    weighted_best_yield = np.sum(best_yield * max_yield_per_acre * areas[:N])
    weighted_area_yield = np.sum(area_yield * max_yield_per_acre * areas[:N])
    weighted_initial_yield = np.sum(initial_yield * max_yield_per_acre * areas[:N])
    weighted_final_yield = np.sum(final_yield * max_yield_per_acre * areas[:N])
   

    best_absolute_yield = best_yield * max_yield_per_acre * areas[:N]
    area_loss = best_absolute_yield - (area_yield * max_yield_per_acre * areas[:N])
    initial_loss = best_absolute_yield - (initial_yield * max_yield_per_acre * areas[:N]) 
    final_loss = best_absolute_yield - (final_yield * max_yield_per_acre * areas[:N])

    loss_data = [area_loss, initial_loss, final_loss]
    labels = ['Uniform by Area', 'Initial Allocation', 'Optimized Allocation']

    print('\n=== Yield Loss Statistics Summary ===')
    for i, label in enumerate(labels):
        loss = loss_data[i]
        print(f"{label}:")
        print(f"  Median: {np.median(loss):.2f} kg")
        print(f"  Mean: {np.mean(loss):.2f} kg")
        print(f"  Standard Deviation: {np.std(loss):.2f} kg")
        print(f"  Coefficient of Variation (CV): {np.std(loss)/np.mean(loss) if np.mean(loss) != 0 else float('inf'):.2f}")
        print(f"  Minimum: {np.min(loss):.2f} kg, Maximum: {np.max(loss):.2f} kg")
        print()
    
    # Calculate coefficient of variation (CV) = standard deviation / mean
    def calculate_cv(losses):
        """Calculate coefficient of variation for yield loss array"""
        nonzero_losses = losses[losses > 0]  # Only consider regions with loss
        if len(nonzero_losses) == 0:
            return 0
        return np.std(nonzero_losses) / np.mean(nonzero_losses) if np.mean(nonzero_losses) > 0 else 0
    
    area_cv = calculate_cv(area_loss)
    initial_cv = calculate_cv(initial_loss)
    final_cv = calculate_cv(final_loss)
    
    
    # Output CV results and analysis
    print('\n=== Yield Loss Fairness Analysis (CV) ===')
    print(f'Uniform by area allocation strategy CV: {area_cv:.4f}')
    print(f'Initial allocation strategy CV: {initial_cv:.4f}') 
    print(f'Optimized allocation strategy CV: {final_cv:.4f}')
    
    # Calculate fairness improvement
    if initial_cv > 0:
        cv_improvement_vs_initial = (initial_cv - final_cv) / initial_cv * 100
    else:
        cv_improvement_vs_initial = 0
        
    if area_cv > 0:
        cv_improvement_vs_area = (area_cv - final_cv) / area_cv * 100
    else:
        cv_improvement_vs_area = 0
    
    print(f'Fairness improvement (optimized vs initial): {cv_improvement_vs_initial:.1f}%')
    print(f'Fairness improvement (optimized vs uniform by area): {cv_improvement_vs_area:.1f}%')
    
    if final_cv < initial_cv and final_cv < area_cv:
        print("Optimized allocation strategy significantly improves fairness (lowest CV)")
    elif final_cv < initial_cv:
        print("Optimized allocation strategy improves fairness compared to initial allocation")
    else:
        print("Fairness of optimized allocation strategy needs further improvement")

    
       
    # Final result output (expanded 4 categories, including absolute yield)
    print('\n=== Optimal Infinite Water Supply Status ===')
    print(f'Total water volume: {np.sum(water_best):.2f} cubic meters')
    print(f'Total effective water: {np.sum(effective_best):.2f} cubic meters')
    print(f'Total yield index (weighted proportion): {np.sum(best_yield * areas[:N]):.4f}')
    print(f'Absolute total yield (kg): {weighted_best_yield:.2f}')
   
    print('\n=== Water Scarcity Allocation by Area Status ===')
    print(f'Total water volume: {np.sum(water_area):.2f} cubic meters')
    print(f'Total effective water: {np.sum(effective_area):.2f} cubic meters')
    print(f'Total yield index (weighted proportion): {np.sum(area_yield * areas[:N]):.4f}')
    print(f'Absolute total yield (kg): {weighted_area_yield:.2f}')
   
    print('\n=== Initial Status ===')
    print(f'Initial water allocation ratio: {initial_x}')
    print(f'Initial total ratio sum: {np.sum(initial_x):.4f}')
    print(f'Initial average ratio: {np.mean(initial_x):.4f}')
    print(f'Initial actual total water: {np.sum(initial_water):.2f} cubic meters')
    print(f'Initial effective total water: {np.sum(initial_effective_water):.2f} cubic meters (considering conveyance loss and irrigation efficiency)')
    print(f'Initial total yield index (weighted proportion): {np.sum(initial_yield * areas[:N]):.4f}')
    print(f'Initial absolute total yield (kg): {weighted_initial_yield:.2f}')
   
    print('\n=== Optimized Status ===')
    print(f'Final water allocation ratio: {final_x}')
    print(f'Final total ratio sum: {np.sum(final_x):.4f}')
    print(f'Final average ratio: {np.mean(final_x):.4f}')
    print(f'Final actual total water: {np.sum(final_water):.2f} cubic meters')
    print(f'Final effective total water: {np.sum(final_effective_water):.2f} cubic meters (considering conveyance loss and irrigation efficiency)')
    print(f'Final total yield index (weighted proportion): {np.sum(final_yield * areas[:N]):.4f}')
    print(f'Final absolute total yield (kg): {weighted_final_yield:.2f}')
   
    print('\n=== Optimization Effect (vs allo by demand) ===')
    yield_improve_index = np.sum(final_yield * areas[:N]) - np.sum(initial_yield * areas[:N])
    yield_improve_abs = weighted_final_yield - weighted_initial_yield
    print(f'Yield index improvement: {yield_improve_index:.4f} ({(yield_improve_index / np.sum(initial_yield * areas[:N]) * 100):.2f}%)')
    print(f'Absolute yield improvement (kg): {yield_improve_abs:.2f} ({(yield_improve_abs / weighted_initial_yield * 100):.2f}%)')
    print(f'Water balance difference: {np.sum(final_water) - total_supply:.2f} cubic meters')
    print(f'Effective water improvement: {np.sum(final_effective_water) - np.sum(initial_effective_water):.2f} cubic meters ({((np.sum(final_effective_water) - np.sum(initial_effective_water)) / np.sum(initial_effective_water) * 100):.2f}%)')
   
    print('\n=== Optimization Effect (vs allo by area) ===')
    yield_improve_index = np.sum(final_yield * areas[:N]) - np.sum(area_yield * areas[:N])
    yield_improve_abs = weighted_final_yield - weighted_area_yield
    print(f'Yield index improvement: {yield_improve_index:.4f} ({(yield_improve_index / np.sum(area_yield * areas[:N]) * 100):.2f}%)')
    print(f'Absolute yield improvement (kg): {yield_improve_abs:.2f} ({(yield_improve_abs / weighted_initial_yield * 100):.2f}%)')
    print(f'Effective water improvement: {np.sum(final_effective_water) - np.sum(effective_area):.2f} cubic meters ({((np.sum(final_effective_water) - np.sum(effective_area)) / np.sum(effective_area) * 100):.2f}%)')

np.random.seed(18) # Set NumPy random seed
# Example call
N = 23 # Number of sub-regions
kstep = 600 # Optimization iteration steps
alpha = 0.01 # Learning rate
gamma = 5# Local penalty weight
theta = 1 # Transportation cost weight (user-set)
beta = 10 # Total water penalty weight
gamma_trans = 1 # User-set
gamma_irricost = 5# User-set
ratio = 0.8
max_de=93.78
hanshu(kstep, alpha, gamma, theta, beta, gamma_trans, gamma_irricost, N, ratio, max_de)