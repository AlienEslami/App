from flask import Flask, request, jsonify
import pyomo.environ as pyo
import pandas as pd
import os
import traceback
import threading
import uuid
import json

app = Flask(__name__)

JOBS_FILE = '/tmp/jobs.json'
_jobs_lock = threading.Lock()


def load_jobs():
    try:
        with open(JOBS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}


def save_job(job_id, data):
    with _jobs_lock:
        jobs = load_jobs()
        jobs[job_id] = data
        with open(JOBS_FILE, 'w') as f:
            json.dump(jobs, f)


def get_job(job_id):
    return load_jobs().get(job_id)


MOCK_RESULT = {
    "status":                "complete",
    "is_mock":               True,
    "mock_reason":           "Optimization infeasible or failed",
    "price_guidance_used":   {},
    "disturbances_applied":  [],
    "buy_multipliers":       [1.10, 1.10, 1.10],
    "sell_multipliers":      [0.80, 0.80, 0.80],
    "period_boundaries":     [16, 32, 48],
    "avg_grid_price":        None,
    "avg_buy_price":         None,
    "avg_sell_price":        None,
    "pto_daily_cost":        None,
    "aggregator_revenue":    None,
    "aggregator_buy_margin": None,
    "aggregator_sell_margin":None,
    "total_buy_cost":        None,
    "total_sell_revenue":    None,
    "total_degradation":     None,
    "peak_power_cost":       None,
    "total_kwh_sold":        None,
    "total_kwh_bought":      None,
    "w_buy":                 [],
    "w_sell":                [],
    "energy":                [],
}


# ==============================================================================
# DATA BUILDER
# ==============================================================================
def build_dataframes(input_data):
    buses = pd.DataFrame(input_data['buses']).rename(
        columns={'bus_kwh': 'Bus (kWh)'})
    chargers = pd.DataFrame(input_data['chargers']).rename(columns={
        'charger_kwhmin': 'Charger (kWh/min)',
        'charger_kw':     'Charger (kW)',
        'max_power_kw':   'Max Power (kW)'})
    trip_time = pd.DataFrame(input_data['trip_time']).rename(columns={
        'time_begin_min':  'Time begin (min)',
        'time_finish_min': 'Time finish (min)'})
    energy_consumption = pd.DataFrame(input_data['energy_consumption']).rename(
        columns={'uncertain_energy_kwhkmmin': 'Uncertain energy (kWh/km*min)'})
    prices = pd.DataFrame(input_data['prices']).rename(
        columns={'spot_market': 'Spot Market'})
    power_price = pd.DataFrame(input_data['power_price']).rename(
        columns={'power': 'Power', 'price': 'Price'})
    return {
        'Buses':              buses,
        'Chargers':           chargers,
        'Trip time':          trip_time,
        'Energy consumption': energy_consumption,
        'Prices':             prices,
        'Power price':        power_price,
    }


# ==============================================================================
# PARAMETER EXTRACTION
# ==============================================================================
def extract_scalars(data, price_guidance={}):
    # ── Fleet size ──────────────────────────────────────────────────────────
    k_count = 2   # buses
    n_count = 2   # chargers
    i_count = 2   # trips
    l_count = 2   # power levels

    # Charging continuity (15-min steps)
    # d_on=4  → must charge ≥ 1 hour consecutively
    # d_off=2 → must rest  ≥ 30 min between charges
    d_on  = 3
    d_off = 2

    # ── Grid buying price P — from data, not LLM-controlled ────────────────
    P_raw   = data['Prices']['Spot Market'].values.flatten().tolist()
    T_steps = len(P_raw)
    P       = P_raw
    avg_P   = sum(P) / len(P)

    mode = price_guidance.get('mode', 'altruistic')

    # ── Period boundaries — divide day into 3 equal parts ──────────────────
    default_boundaries = [T_steps // 3, (T_steps * 2) // 3, T_steps]
    boundaries = price_guidance.get('period_boundaries', default_boundaries)
    # Ensure last boundary covers full T_steps
    boundaries[-1] = T_steps

    # ── Read raw multipliers from price guidance ────────────────────────────
    buy_mults_raw  = price_guidance.get('buy_multipliers',  [1.05, 1.10, 1.05])
    sell_mults_raw = price_guidance.get('sell_multipliers', [0.80, 0.85, 0.80])
    # ── Step 1: clip individual multipliers ─────────────────────────────────
    buy_multipliers  = [max(1.01, min(1.50, float(m))) for m in buy_mults_raw]
    sell_multipliers = [max(0.40, min(0.99, float(m))) for m in sell_mults_raw]

    # ── Step 2: enforce average buy multiplier (eq 7 analog from paper) ─────
    # eq 7 analog: average buy price EBA charges PTO ≈ average grid price
    # Multiplier average should stay close to 1.0
    # Allow small tolerance: selfish can go up to 1.05, altruistic up to 1.02
    TARGET_AVG_BUY = 1.05   # selfish: small average premium allowed
    TARGET_AVG_ALT = 1.02   # altruistic: essentially pass-through pricing

    target = TARGET_AVG_BUY if mode == 'selfish' else TARGET_AVG_ALT

    actual_avg_buy = sum(buy_multipliers) / len(buy_multipliers)
    if actual_avg_buy > target:
        scale = target / actual_avg_buy
        buy_multipliers = [max(1.01, m * scale) for m in buy_multipliers]
        print(f'Buy multipliers rescaled: avg was {actual_avg_buy:.4f}, '
              f'now {sum(buy_multipliers)/len(buy_multipliers):.4f}')

    # ── Step 3: enforce sell multipliers stay below buy multipliers ──────────
    # S_sell[t] must always be less than S_buy[t] (EBA can't pay more than it charges)
    # and less than P[t] (EBA must profit on V2G resale)
    # δ relationship from paper: ρ^- = δ × ρ^+ (sell is fraction of buy)
    # We enforce this per-period
    sell_multipliers = [
        min(sell_multipliers[i], buy_multipliers[i] - 0.01)  # sell < buy always
        for i in range(len(sell_multipliers))
    ]

    # ── Step 4: enforce average sell multiplier ──────────────────────────────
    # Average sell price EBA pays PTO should not be below a floor
    # (prevents EBA from paying almost nothing for V2G — unrealistic)
    # Floor: sell average must be at least 0.60 × buy average
    avg_buy  = sum(buy_multipliers)  / len(buy_multipliers)
    avg_sell = sum(sell_multipliers) / len(sell_multipliers)
    SELL_FLOOR_RATIO = 0.60
    if avg_sell < avg_buy * SELL_FLOOR_RATIO:
        scale_up = (avg_buy * SELL_FLOOR_RATIO) / avg_sell
        sell_multipliers = [min(m * scale_up, buy_multipliers[i] - 0.01)
                            for i, m in enumerate(sell_multipliers)]
        print(f'Sell multipliers rescaled up: avg was {avg_sell:.4f}, '
              f'now {sum(sell_multipliers)/len(sell_multipliers):.4f}')

    print(f'Final buy_multipliers={[round(m,4) for m in buy_multipliers]}, '
          f'avg={sum(buy_multipliers)/len(buy_multipliers):.4f}')
    print(f'Final sell_multipliers={[round(m,4) for m in sell_multipliers]}, '
          f'avg={sum(sell_multipliers)/len(sell_multipliers):.4f}')

    # ── Build per-timestep price arrays ────────────────────────────────────
    S_buy  = []
    S_sell = []
    for t_idx in range(T_steps):
        if t_idx < boundaries[0]:
            p = 0
        elif t_idx < boundaries[1]:
            p = 1
        else:
            p = 2
        S_buy.append(buy_multipliers[p]  * P[t_idx])
        S_sell.append(sell_multipliers[p] * P[t_idx])

    avg_S_buy  = sum(S_buy)  / len(S_buy)
    avg_S_sell = sum(S_sell) / len(S_sell)

    # ── Bus / charger parameters ────────────────────────────────────────────
    C_bat = data['Buses']['Bus (kWh)'].tolist()[:k_count]

    # alpha/beta: already in kWh/step (verified from data — raw=50 kWh/step)
    alpha_raw = data['Chargers']['Charger (kWh/min)'].tolist()[:n_count]
    beta_raw  = data['Chargers']['Charger (kWh/min)'].tolist()[:n_count]
    alpha = alpha_raw
    beta  = beta_raw

    # gama: kWh/km × speed(km/min) = kWh/step
    # raw value in sheet is kWh/km (e.g. 3.6), speed = 12 km/h = 0.2 km/min
    avg_speed_km_min = 12.0 / 60.0
    gama_raw = data['Energy consumption'][
                   'Uncertain energy (kWh/km*min)'].tolist()[:i_count]
    gama = [v * avg_speed_km_min for v in gama_raw]

    U_pow   = data['Power price']['Power'].tolist()[:l_count]
    U_price = data['Power price']['Price'].tolist()[:l_count]
    U_max   = data['Chargers']['Max Power (kW)'].tolist()[0]

    # ── Trip times ──────────────────────────────────────────────────────────
    # Raw values are already in timestep indices (same resolution as T_steps)
    T_start_raw = [int(x) for x in
                   data['Trip time']['Time begin (min)'].tolist()[:i_count]]
    T_end_raw   = [int(x) for x in
                   data['Trip time']['Time finish (min)'].tolist()[:i_count]]
    T_start = [max(1, min(T_steps-1, t)) for t in T_start_raw]
    T_end   = [max(T_start[i]+1, min(T_steps, t))
               for i, t in enumerate(T_end_raw)]

    # ── Sanity check ────────────────────────────────────────────────────────
    charge_per_step = 0.90 * alpha[0]
    drain_per_step  = gama[0]
    usable_kWh      = (1.0 - 0.2) * C_bat[0]
    print(f'T={T_steps}, k={k_count}, n={n_count}, i={i_count}')
    print(f'avg_P={avg_P:.6f}')
    print(f'avg_S_buy={avg_S_buy:.6f} ({avg_S_buy/avg_P*100:.1f}% of grid)')
    print(f'avg_S_sell={avg_S_sell:.6f} ({avg_S_sell/avg_P*100:.1f}% of grid)')
    print(f'buy_multipliers={buy_multipliers}')
    print(f'sell_multipliers={sell_multipliers}')
    print(f'alpha={alpha[0]:.2f} kWh/step, gama={gama[0]:.4f} kWh/step')
    print(f'Charge/step={charge_per_step:.2f}, Drain/step={drain_per_step:.4f}')
    print(f'T_start={T_start}, T_end={T_end}')
    print(f'Trip durations={[T_end[i]-T_start[i] for i in range(i_count)]} steps')
    print(f'Max trip on full charge: {usable_kWh/drain_per_step:.0f} steps — '
          f'{"OK" if max(T_end[i]-T_start[i] for i in range(i_count)) < usable_kWh/drain_per_step else "INFEASIBLE"}')

    return {
        'T_steps': T_steps,
        'k_count': k_count, 'n_count': n_count,
        'i_count': i_count, 'l_count': l_count,
        'd_on': d_on, 'd_off': d_off,
        'P': P, 'S_buy': S_buy, 'S_sell': S_sell,
        'avg_P': avg_P, 'avg_S_buy': avg_S_buy, 'avg_S_sell': avg_S_sell,
        'buy_multipliers': buy_multipliers,
        'sell_multipliers': sell_multipliers,
        'boundaries': boundaries,
        'C_bat': C_bat, 'alpha': alpha, 'beta': beta, 'gama': gama,
        'U_pow': U_pow, 'U_price': U_price, 'U_max': U_max,
        'T_start': T_start, 'T_end': T_end,
        'ch_eff': 0.90, 'dch_eff': 1.0 / 0.90,
        'E_0': 0.2, 'E_min': 0.2, 'E_max': 1.0, 'E_end': 0.2,
        'R': 130, 'Ah': 905452, 'V': 512,
    }
import numpy as np



def apply_disturbances(sc, disturbances):
    if not disturbances:
        return sc
    T_start = list(sc['T_start'])
    T_end   = list(sc['T_end'])
    result = [a / b for a, b in zip(T_start, T_end)]
    print(f'T_start/T_end={result}')
    for d in disturbances:
        try:
            bus_id = int(d.get('bus_id', 0)) - 1
            delay  = int(d.get('delay_minutes', 0))
            d_type = d.get('disturbance_type', '')
            if bus_id < 0 or bus_id >= len(T_start):
                continue
            delay_steps = int(round(delay / 15.0))
            if d_type in ['late', 'breakdown']:
                T_start[bus_id] = min(sc['T_steps'], T_start[bus_id] + delay_steps)
                T_end[bus_id]   = min(sc['T_steps'], T_end[bus_id]   + delay_steps)
            elif d_type == 'early_return':
                T_end[bus_id] = max(T_start[bus_id] + 1,
                                    T_end[bus_id] - delay_steps)
        except Exception as e:
            print(f'Disturbance error: {e}')
    sc['T_start'] = T_start
    sc['T_end']   = T_end
    return sc


# ==============================================================================
# PTO SOLVER
# Objective: min DOC = S_buy*w_buy - S_sell*w_sell + degradation + peak_power
#
# Economic interpretation:
#   S_buy[t]  = price EBA charges PTO per kWh (> P[t]) → PTO buys from EBA
#   S_sell[t] = price EBA pays PTO per kWh  (< P[t]) → PTO sells to EBA (V2G)
#
# EBA profit (computed in run_optimization):
#   = sum((S_buy[t]  - P[t]) × w_buy[t])   ← margin on energy sold to PTO
#   + sum((P[t] - S_sell[t]) × w_sell[t])  ← margin on V2G resold to grid
# ==============================================================================
def solvePTO(sc):
    T       = sc['T_steps']
    P       = sc['P']
    S_buy   = sc['S_buy']
    S_sell  = sc['S_sell']
    alpha   = sc['alpha']
    beta    = sc['beta']
    gama    = sc['gama']
    C_bat   = sc['C_bat']
    d_on    = sc['d_on']
    d_off   = sc['d_off']
    ch_eff  = sc['ch_eff']
    dch_eff = sc['dch_eff']
    E_0     = sc['E_0']
    E_min   = sc['E_min']
    E_max   = sc['E_max']
    E_end   = sc['E_end']
    R       = sc['R']
    Ah      = sc['Ah']
    V       = sc['V']
    U_pow   = sc['U_pow']
    U_price = sc['U_price']
    U_max   = sc['U_max']
    T_start = sc['T_start']
    T_end_  = sc['T_end']

    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(sc['i_count'])
    model.T = pyo.RangeSet(T)
    model.K = pyo.RangeSet(sc['k_count'])
    model.N = pyo.RangeSet(sc['n_count'])
    model.L = pyo.RangeSet(sc['l_count'])

    # Parameters
    model.T_start  = pyo.Param(model.I, initialize=lambda m,i: T_start[i-1])
    model.T_end    = pyo.Param(model.I, initialize=lambda m,i: T_end_[i-1])
    model.alpha    = pyo.Param(model.N, initialize=lambda m,n: alpha[n-1])
    model.beta     = pyo.Param(model.N, initialize=lambda m,n: beta[n-1])
    model.gama     = pyo.Param(model.I, initialize=lambda m,i: gama[i-1])
    model.P        = pyo.Param(model.T, initialize=lambda m,t: P[t-1])
    model.S_buy    = pyo.Param(model.T, initialize=lambda m,t: S_buy[t-1])
    model.S_sell   = pyo.Param(model.T, initialize=lambda m,t: S_sell[t-1])
    model.C_bat    = pyo.Param(model.K, initialize=lambda m,k: C_bat[k-1])
    model.U_pow    = pyo.Param(model.L, initialize=lambda m,l: U_pow[l-1])
    model.U_price  = pyo.Param(model.L, initialize=lambda m,l: U_price[l-1])

    # Variables
    model.b      = pyo.Var(model.K, model.I, model.T, domain=pyo.Binary)
    model.x      = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.y      = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.c      = pyo.Var(model.K, model.T, domain=pyo.Binary)
    model.u      = pyo.Var(model.L, domain=pyo.Binary)
    model.e      = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.w_buy  = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.w_sell = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.d      = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)

    # 1:1 assignment: bus k can only serve trip i=k
    for k in model.K:
        for i in model.I:
            if i != k:
                for t in model.T:
                    model.b[k, i, t].fix(0)

    model.constraints = pyo.ConstraintList()

    # ── Constraint 2: bus on trip XOR at charger ──────────────────────────
    for k in model.K:
        for t in model.T:
            model.constraints.add(
                sum(model.b[k,i,t] for i in model.I) + model.c[k,t] <= 1)

    # ── Constraint 3: one bus per trip-timeslot ───────────────────────────
    for i in model.I:
        for t in range(model.T_start[i], model.T_end[i]):
            model.constraints.add(
                sum(model.b[k,i,t] for k in model.K) == 1)

    # ── Constraint 4: trip continuity ────────────────────────────────────
    for i in model.I:
        for k in model.K:
            for t in range(model.T_start[i], model.T_end[i]-1):
                model.constraints.add(model.b[k,i,t+1] >= model.b[k,i,t])

    # ── Constraint 5: one operation per charger per slot ─────────────────
    for n in model.N:
        for t in model.T:
            model.constraints.add(
                sum(model.x[k,n,t] for k in model.K)
              + sum(model.y[k,n,t] for k in model.K) <= 1)

    # ── Constraint 6: charger ops linked to c[k,t] ───────────────────────
    for k in model.K:
        for t in model.T:
            model.constraints.add(
                sum(model.x[k,n,t] for n in model.N)
              + sum(model.y[k,n,t] for n in model.N) <= model.c[k,t])

    # ── Constraint 7: energy balance ─────────────────────────────────────
    for k in model.K:
        model.constraints.add(model.e[k,1] == E_0 * C_bat[k-1])
        for t in range(2, T+1):
            model.constraints.add(
                model.e[k,t] == model.e[k,t-1]
                + sum(ch_eff  * alpha[n-1] * model.x[k,n,t] for n in model.N)
                - sum(gama[i-1] * model.b[k,i,t] for i in model.I)
                - sum(dch_eff * beta[n-1]  * model.y[k,n,t] for n in model.N))

    # ── Constraints 8.1-8.4: charging continuity ─────────────────────────
    for k in model.K:
        for n in model.N:
            for t in range(2, T - d_off):
                model.constraints.add(
                    1 - model.x[k,n,t] + model.x[k,n,t-1]
                    + (1/d_off)*sum(model.x[k,n,j]
                                    for j in range(t, t+d_off)) <= 2)
            for t in range(T - d_off + 1, T):
                if T - t + 1 > 0:
                    model.constraints.add(
                        1 - model.x[k,n,t] + model.x[k,n,t-1]
                        + (1/(T-t+1))*sum(model.x[k,n,j]
                                          for j in range(t, T)) <= 2)
            for t in range(2, T - d_on):
                model.constraints.add(
                    1 - model.x[k,n,t] + model.x[k,n,t-1]
                    + (1/d_on)*sum(model.x[k,n,j]
                                   for j in range(t, t+d_on)) >= 1)
            for t in range(T - d_on + 1, T):
                if T - t + 1 > 0:
                    model.constraints.add(
                        1 - model.x[k,n,t] + model.x[k,n,t-1]
                        + (1/(T-t+1))*sum(model.x[k,n,j]
                                          for j in range(t, T)) >= 1)

    # ── Constraints 9.1-9.2: w_buy and w_sell definitions ────────────────
    for t in model.T:
        model.constraints.add(
            sum(ch_eff * alpha[n-1] * model.x[k,n,t]
                for n in model.N for k in model.K) == model.w_buy[t])
        model.constraints.add(
            sum(dch_eff * beta[n-1] * model.y[k,n,t]
                for n in model.N for k in model.K) == model.w_sell[t])

    # No discharging at t=1
    model.constraints.add(
        sum(dch_eff * beta[n-1] * model.y[k,n,1]
            for n in model.N for k in model.K) == 0)

    # ── Constraint 10: one peak power level ──────────────────────────────
    model.constraints.add(sum(model.u[l] for l in model.L) == 1)

    # ── Constraints 11-12: peak power limits ─────────────────────────────
    for t in model.T:
        model.constraints.add(
            sum(alpha[n-1]*model.x[k,n,t] for k in model.K for n in model.N)
            <= sum(U_pow[l-1]*model.u[l] for l in model.L))
        model.constraints.add(
            sum(alpha[n-1]*model.x[k,n,t] for k in model.K for n in model.N)
            <= U_max)

    # ── Constraints 13-14: SOC bounds ────────────────────────────────────
    for k in model.K:
        for t in model.T:
            model.constraints.add(model.e[k,t] >= C_bat[k-1] * E_min)
            model.constraints.add(
                E_max * C_bat[k-1] >= model.e[k,t]
                + sum(ch_eff * alpha[n-1] * model.x[k,n,t] for n in model.N))

    # ── Constraint 15.2: end-of-day minimum SOC ──────────────────────────
    for k in model.K:
        model.constraints.add(
            model.e[k, T-1]
            + sum(ch_eff * alpha[n-1] * model.x[k,n,T] for n in model.N)
            >= E_end * C_bat[k-1])

    # ── Constraint 16: battery degradation cost ───────────────────────────
    for k in model.K:
        for t in model.T:
            model.constraints.add(
                model.d[k,t] == ((R * C_bat[k-1] * 1000) / (Ah * V))
                * sum(beta[n-1] * model.y[k,n,t] for n in model.N))

    # ── Objective: min PTO daily cost ────────────────────────────────────
    # PTO pays S_buy[t] per kWh to EBA for charging
    # PTO receives S_sell[t] per kWh from EBA for V2G
    def rule_obj(mod):
        return (sum(S_buy[t-1]  * mod.w_buy[t]  for t in mod.T)
              - sum(S_sell[t-1] * mod.w_sell[t] for t in mod.T)
              + sum(mod.d[k,t] for k in mod.K for t in mod.T)
              + sum(U_price[l-1] * mod.u[l] for l in mod.L))

    model.obj = pyo.Objective(rule=rule_obj, sense=pyo.minimize)

    print('Solving PTO')
    opt = pyo.SolverFactory('highs')
    results = opt.solve(model, load_solutions=False, tee=False)
    tc = results.solver.termination_condition
    print(f'PTO termination: {tc}')
    if tc in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible):
        model.solutions.load_from(results)
        print('PTO done')
        return model
    print(f'PTO infeasible: {tc}')
    return None


# ==============================================================================
# OPTIMIZATION RUNNER
# ==============================================================================
def run_optimization(job_id, input_data, price_guidance={},
                     disturbances=[], max_iter=1):
    try:
        print(f'Job {job_id} started')
        print(f'Price guidance: {price_guidance}')
        print(f'Disturbances:   {disturbances}')

        data = build_dataframes(input_data)
        sc   = extract_scalars(data, price_guidance)

        if disturbances:
            sc = apply_disturbances(sc, disturbances)
            print(f'Applied {len(disturbances)} disturbances')
            print(f'After disturbances: T_start={sc["T_start"]}, T_end={sc["T_end"]}')

        T = sc['T_steps']

        model = solvePTO(sc)
        if model is None:
            mock = dict(MOCK_RESULT)
            mock.update({
                'price_guidance_used':  price_guidance,
                'disturbances_applied': disturbances,
                'mock_reason':          'PTO optimization infeasible',
                'buy_multipliers':      sc['buy_multipliers'],
                'sell_multipliers':     sc['sell_multipliers'],
                'period_boundaries':    sc['boundaries'],
                'avg_grid_price':       sc['avg_P'],
            })
            save_job(job_id, mock)
            return

        # ── Economic metrics ──────────────────────────────────────────────
        # PTO costs and revenues
        total_pto_buy_cost = sum(
            sc['S_buy'][t-1] * pyo.value(model.w_buy[t])
            for t in range(1, T+1))
        total_pto_sell_rev = sum(
            sc['S_sell'][t-1] * pyo.value(model.w_sell[t])
            for t in range(1, T+1))
        total_degradation = sum(
            pyo.value(model.d[k,t])
            for k in range(1, sc['k_count']+1)
            for t in range(1, T+1))
        peak_power_cost = sum(
            sc['U_price'][l-1] * pyo.value(model.u[l])
            for l in range(1, sc['l_count']+1))

        pto_daily_cost = (total_pto_buy_cost - total_pto_sell_rev
                          + total_degradation + peak_power_cost)

        # EBA margins:
        # Charging margin: EBA buys from grid at P[t], sells to PTO at S_buy[t]
        # V2G margin:      EBA buys from PTO at S_sell[t], resells to grid at P[t]
        agg_buy_margin = sum(
            (sc['S_buy'][t-1] - sc['P'][t-1]) * pyo.value(model.w_buy[t])
            for t in range(1, T+1))
        agg_sell_margin = sum(
            (sc['P'][t-1] - sc['S_sell'][t-1]) * pyo.value(model.w_sell[t])
            for t in range(1, T+1))
        aggregator_revenue = agg_buy_margin + agg_sell_margin

        total_kwh_sold   = sum(pyo.value(model.w_sell[t]) for t in range(1, T+1))
        total_kwh_bought = sum(pyo.value(model.w_buy[t])  for t in range(1, T+1))

        print(f'PTO daily cost:        {pto_daily_cost:.4f}')
        print(f'Aggregator revenue:    {aggregator_revenue:.4f}')
        print(f'  Buy margin:          {agg_buy_margin:.4f}')
        print(f'  Sell (V2G) margin:   {agg_sell_margin:.4f}')
        print(f'Total kWh bought:      {total_kwh_bought:.4f}')
        print(f'Total kWh sold (V2G):  {total_kwh_sold:.4f}')

        result = {
            "status":                 "complete",
            "is_mock":                False,
            "price_guidance_used":    price_guidance,
            "disturbances_applied":   disturbances,
            # Price settings
            "buy_multipliers":        sc['buy_multipliers'],
            "sell_multipliers":       sc['sell_multipliers'],
            "period_boundaries":      sc['boundaries'],
            "avg_grid_price":         sc['avg_P'],
            "avg_buy_price":          sc['avg_S_buy'],
            "avg_sell_price":         sc['avg_S_sell'],
            # Economic outcomes
            "pto_daily_cost":         pto_daily_cost,
            "aggregator_revenue":     aggregator_revenue,
            "aggregator_buy_margin":  agg_buy_margin,
            "aggregator_sell_margin": agg_sell_margin,
            "total_buy_cost":         total_pto_buy_cost,
            "total_sell_revenue":     total_pto_sell_rev,
            "total_degradation":      total_degradation,
            "peak_power_cost":        peak_power_cost,
            "total_kwh_sold":         total_kwh_sold,
            "total_kwh_bought":       total_kwh_bought,
            # Time series
            "w_buy":  [float(pyo.value(model.w_buy[t]))
                       for t in range(1, T+1)],
            "w_sell": [float(pyo.value(model.w_sell[t]))
                       for t in range(1, T+1)],
            "energy": [[float(pyo.value(model.e[k,t]))
                        for t in range(1, T+1)]
                       for k in range(1, sc['k_count']+1)],
        }
        save_job(job_id, result)
        print(f'Job {job_id} complete')

    except Exception as e:
        traceback.print_exc()
        mock = dict(MOCK_RESULT)
        mock.update({
            'price_guidance_used':  price_guidance,
            'disturbances_applied': disturbances,
            'mock_reason':          f'Unexpected error: {str(e)}',
        })
        save_job(job_id, mock)


# ==============================================================================
# FLASK ENDPOINTS
# ==============================================================================
@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        input_data     = request.json['input']
        price_guidance = request.json.get('price_guidance', {})
        disturbances   = request.json.get('disturbances', [])
        max_iter       = request.json.get('max_iter', 1)
        job_id = str(uuid.uuid4())
        save_job(job_id, {"status": "running"})
        threading.Thread(
            target=run_optimization,
            args=(job_id, input_data, price_guidance, disturbances, max_iter)
        ).start()
        return jsonify({"job_id": job_id, "status": "running"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


@app.route('/health', methods=['GET'])
def health():
    jobs = load_jobs()
    return jsonify({
        "status":      "ok",
        "active_jobs": len([j for j in jobs.values()
                            if j.get("status") == "running"]),
        "total_jobs":  len(jobs)
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
