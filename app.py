from flask import Flask, request, jsonify
import pyomo.environ as pyo
import pandas as pd
import os
import traceback
import threading
import uuid

app = Flask(__name__)

# In-memory job store
jobs = {}

# ==============================================================================
# LLM HOOK - Replace this function body with LLM API call in the future
# ==============================================================================
def get_high_level_prices(data):
    """
    FUTURE: Replace with LLM API call that returns price bounds.
    Input:  market data, system state, historical patterns
    Output: price bounds dict
    """
    X_up   = data['Average prices']['Max price'].values.flatten()
    X_low  = data['Average prices']['Min price'].values.flatten()
    Mi_up  = data['Average prices']['Max cap'].values.flatten()
    Mi_low = data['Average prices']['Min cap'].values.flatten()
    return {
        'X_up':   X_up,
        'X_low':  X_low,
        'Mi_up':  Mi_up,
        'Mi_low': Mi_low,
        'X_avg':  0.12,
        'Mi_avg': 0.015,
        'delta':  0.8,
    }


# ==============================================================================
# DATA BUILDER
# ==============================================================================
def build_dataframes(input_data):
    buses = pd.DataFrame(input_data['buses']).rename(columns={
        'bus_kwh': 'Bus (kWh)'})
    chargers = pd.DataFrame(input_data['chargers']).rename(columns={
        'charger_kwhmin': 'Charger (kWh/min)',
        'charger_kw':     'Charger (kW)',
        'max_power_kw':   'Max Power (kW)'})
    trip_time = pd.DataFrame(input_data['trip_time']).rename(columns={
        'time_begin_min':  'Time begin (min)',
        'time_finish_min': 'Time finish (min)'})
    energy_consumption = pd.DataFrame(input_data['energy_consumption']).rename(columns={
        'uncertain_energy_kwhkmmin': 'Uncertain energy (kWh/km*min)'})
    prices = pd.DataFrame(input_data['prices']).rename(columns={
        'spot_market':    'Spot Market',
        'capacity_price': 'Capacity price'})
    power_price = pd.DataFrame(input_data['power_price']).rename(columns={
        'power': 'Power',
        'price': 'Price'})
    average_prices = pd.DataFrame(input_data['average_prices']).rename(columns={
        'max_price': 'Max price',
        'min_price': 'Min price',
        'max_cap':   'Max cap',
        'min_cap':   'Min cap'})
    periods = pd.DataFrame(input_data['periods']).rename(columns={
        'period': 'Period',
        'begin':  'Begin',
        'end':    'End',
        'len':    'Len'})
    return {
        'Buses':              buses,
        'Chargers':           chargers,
        'Trip time':          trip_time,
        'Energy consumption': energy_consumption,
        'Prices':             prices,
        'Power price':        power_price,
        'Average prices':     average_prices,
        'Periods':            periods,
    }


# ==============================================================================
# HELPERS
# ==============================================================================
def extract_scalars(data):
    T_steps = 24
    delta_t = 1
    M       = 10000

    PI     = data['Prices']['Spot Market'].values.flatten()[:T_steps]
    PI_cap = data['Prices']['Capacity price'].values.flatten()[:T_steps]

    p_count = len(data['Periods']['Period'])
    Q_begin_raw = [int(x) for x in data['Periods']['Begin'].tolist()]
    Q_end_raw   = [int(x) for x in data['Periods']['End'].tolist()]
    Q_len_raw   = [int(x) for x in data['Periods']['Len'].tolist()]

    scale   = T_steps / 96.0
    Q_begin = [max(1, int(round(b * scale))) for b in Q_begin_raw]
    Q_end   = [min(T_steps, int(round(e * scale))) for e in Q_end_raw]
    Q_len   = [max(1, Q_end[i] - Q_begin[i] + 1) for i in range(p_count)]

    k_count = min(3, len(data['Buses']['Bus (kWh)']))
    n_count = min(3, len(data['Chargers']['Charger (kWh/min)']))
    i_count = min(3, len(data['Trip time']['Time begin (min)']))
    l_count = len(data['Power price']['Power'])

    C_bat   = data['Buses']['Bus (kWh)'].tolist()[:k_count]
    alpha   = data['Chargers']['Charger (kWh/min)'].tolist()[:n_count]
    beta    = data['Chargers']['Charger (kWh/min)'].tolist()[:n_count]
    gama    = data['Energy consumption']['Uncertain energy (kWh/km*min)'].tolist()[:i_count]
    U_pow   = data['Power price']['Power'].tolist()
    U_price = data['Power price']['Price'].tolist()
    U_max   = data['Chargers']['Max Power (kW)'].tolist()[0]
    U_cap   = 200

    T_start_raw = [int(x) for x in data['Trip time']['Time begin (min)'].tolist()[:i_count]]
    T_end_raw   = [int(x) for x in data['Trip time']['Time finish (min)'].tolist()[:i_count]]
    T_start = [max(1, min(T_steps, int(round(t * scale)))) for t in T_start_raw]
    T_end   = [max(1, min(T_steps, int(round(t * scale)))) for t in T_end_raw]
    T_end   = [max(T_start[i]+1, T_end[i]) for i in range(i_count)]

    return {
        'T_steps': T_steps, 'delta_t': delta_t, 'M': M,
        'PI': PI, 'PI_cap': PI_cap,
        'p_count': p_count, 'Q_begin': Q_begin, 'Q_end': Q_end, 'Q_len': Q_len,
        'k_count': k_count, 'n_count': n_count, 'i_count': i_count, 'l_count': l_count,
        'C_bat': C_bat, 'alpha': alpha, 'beta': beta, 'gama': gama,
        'U_pow': U_pow, 'U_price': U_price, 'U_max': U_max, 'U_cap': U_cap,
        'T_start': T_start, 'T_end': T_end,
        'ch_eff': 0.90, 'dch_eff': 1.0,
        'E_0': 0.2, 'E_min': 0.2, 'E_max': 1.0, 'E_end': 0.2,
        'R': 130, 'Ah': 905452, 'V': 512,
        'd_on': 1, 'd_off': 1,
    }


def get_period_timesteps(p_idx, Q_begin, Q_end, valid_T):
    return [t for t in range(Q_begin[p_idx], Q_end[p_idx]+1) if t in valid_T]


# ==============================================================================
# HPR SOLVER
# ==============================================================================
def solveHRP(sc, price_bounds, y_buy, y_sell, d_l, u_l, count):
    T_steps  = sc['T_steps']
    p_count  = sc['p_count']
    i_count  = sc['i_count']
    k_count  = sc['k_count']
    n_count  = sc['n_count']
    l_count  = sc['l_count']
    Q_begin  = sc['Q_begin']
    Q_end    = sc['Q_end']
    Q_len    = sc['Q_len']
    T_start  = sc['T_start']
    T_end    = sc['T_end']
    alpha    = sc['alpha']
    beta     = sc['beta']
    gama     = sc['gama']
    C_bat    = sc['C_bat']
    U_pow    = sc['U_pow']
    U_price  = sc['U_price']
    U_max    = sc['U_max']
    ch_eff   = sc['ch_eff']
    dch_eff  = sc['dch_eff']
    E_0      = sc['E_0']
    E_min    = sc['E_min']
    E_max    = sc['E_max']
    E_end    = sc['E_end']
    R        = sc['R']
    Ah       = sc['Ah']
    V        = sc['V']
    d_on     = sc['d_on']
    d_off    = sc['d_off']
    M        = sc['M']
    PI       = sc['PI']
    PI_cap   = sc['PI_cap']

    X_up     = price_bounds['X_up']
    X_low    = price_bounds['X_low']
    X_avg    = price_bounds['X_avg']
    Mi_up    = price_bounds['Mi_up']
    Mi_low   = price_bounds['Mi_low']
    Mi_avg   = price_bounds['Mi_avg']
    delta    = price_bounds['delta']

    model = pyo.ConcreteModel()

    model.P = pyo.RangeSet(p_count)
    model.I = pyo.RangeSet(i_count)
    model.T = pyo.RangeSet(T_steps)
    model.K = pyo.RangeSet(k_count)
    model.N = pyo.RangeSet(n_count)
    model.L = pyo.RangeSet(l_count)

    valid_T = set(range(1, T_steps+1))

    model.PI      = pyo.Param(model.T, initialize=lambda m,t: PI[t-1])
    model.PI_cap  = pyo.Param(model.T, initialize=lambda m,t: PI_cap[t-1])
    model.alpha   = pyo.Param(model.N, initialize=lambda m,n: alpha[n-1])
    model.beta    = pyo.Param(model.N, initialize=lambda m,n: beta[n-1])
    model.C_bat   = pyo.Param(model.K, initialize=lambda m,k: C_bat[k-1])
    model.U_pow   = pyo.Param(model.L, initialize=lambda m,l: U_pow[l-1])
    model.U_price = pyo.Param(model.L, initialize=lambda m,l: U_price[l-1])
    model.gama    = pyo.Param(model.I, initialize=lambda m,i: gama[i-1])
    model.X_low   = pyo.Param(model.P, initialize=lambda m,p: float(X_low[p-1]))
    model.X_up    = pyo.Param(model.P, initialize=lambda m,p: float(X_up[p-1]))
    model.Mi_low  = pyo.Param(model.P, initialize=lambda m,p: float(Mi_low[p-1]))
    model.Mi_up   = pyo.Param(model.P, initialize=lambda m,p: float(Mi_up[p-1]))
    model.y_buy   = pyo.Param(model.T, initialize=y_buy)
    model.y_sell  = pyo.Param(model.T, initialize=y_sell)
    model.d_l     = pyo.Param(model.K, model.T, initialize=d_l)
    model.u_l     = pyo.Param(model.L, initialize=u_l)

    model.pho_plus  = pyo.Var(model.P, domain=pyo.NonNegativeReals)
    model.pho_minus = pyo.Var(model.P, domain=pyo.NonNegativeReals)
    model.mi        = pyo.Var(model.P, domain=pyo.NonNegativeReals)
    model.e         = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.w_buy     = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.w_sell    = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.d         = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.b         = pyo.Var(model.K, model.I, model.T, domain=pyo.Binary)
    model.x         = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.y         = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.c         = pyo.Var(model.K, model.T, domain=pyo.Binary)
    model.u         = pyo.Var(model.L, domain=pyo.Binary)
    model.q1        = pyo.Var(model.P, model.K, model.N, model.T, within=pyo.NonNegativeReals)
    model.q2        = pyo.Var(model.P, model.K, model.N, model.T, within=pyo.NonNegativeReals)

    model.constraints = pyo.ConstraintList()

    for p in model.P:
        rho_up = float(X_up[p-1])
        for k in model.K:
            for n in model.N:
                for t in model.T:
                    model.constraints.add(model.q1[p,k,n,t] <= rho_up * model.x[k,n,t])
                    model.constraints.add(model.q1[p,k,n,t] <= model.pho_plus[p])
                    model.constraints.add(model.q1[p,k,n,t] >= model.pho_plus[p] - (1 - model.x[k,n,t]) * rho_up)
                    model.constraints.add(model.q1[p,k,n,t] >= 0)
                    model.constraints.add(model.q2[p,k,n,t] <= delta * rho_up * model.y[k,n,t])
                    model.constraints.add(model.q2[p,k,n,t] <= model.pho_minus[p])
                    model.constraints.add(model.q2[p,k,n,t] >= model.pho_minus[p] - (1 - model.y[k,n,t]) * delta * rho_up)
                    model.constraints.add(model.q2[p,k,n,t] >= 0)

    def rule_obj(mod):
        f1_buy  = sum(ch_eff * mod.alpha[n] * mod.q1[p,k,n,t]
                      for p in mod.P
                      for t in get_period_timesteps(p-1, Q_begin, Q_end, valid_T)
                      for k in mod.K for n in mod.N)
        f1_sell = sum(dch_eff * mod.beta[n] * mod.q2[p,k,n,t]
                      for p in mod.P
                      for t in get_period_timesteps(p-1, Q_begin, Q_end, valid_T)
                      for k in mod.K for n in mod.N)
        f3 = sum(mod.PI[t] * (mod.w_sell[t] - mod.w_buy[t]) for t in mod.T)
        return f1_buy - f1_sell + f3

    model.obj = pyo.Objective(rule=rule_obj, sense=pyo.maximize)

    for p in model.P:
        model.constraints.add(model.pho_plus[p] >= model.X_low[p])
        model.constraints.add(model.pho_plus[p] <= model.X_up[p])
        model.constraints.add(model.pho_minus[p] == model.pho_plus[p] * delta)
        model.constraints.add(model.mi[p] >= model.Mi_low[p])
        model.constraints.add(model.mi[p] <= model.Mi_up[p])

    model.constraints.add(
        (1/T_steps) * sum(Q_len[p-1] * model.pho_plus[p] for p in model.P) <= X_avg)
    model.constraints.add(
        (1/T_steps) * sum(Q_len[p-1] * model.mi[p] for p in model.P) <= Mi_avg)

    for k in model.K:
        for t in model.T:
            model.constraints.add(
                sum(model.b[k,i,t] for i in model.I) + model.c[k,t] <= 1)

    for i in model.I:
        ts = [t for t in range(T_start[i-1], T_end[i-1]) if t in valid_T]
        for t in ts:
            model.constraints.add(sum(model.b[k,i,t] for k in model.K) == 1)

    for i in model.I:
        for k in model.K:
            ts = [t for t in range(T_start[i-1], T_end[i-1]-1)
                  if t in valid_T and t+1 in valid_T]
            for t in ts:
                model.constraints.add(model.b[k,i,t+1] >= model.b[k,i,t])

    for n in model.N:
        for t in model.T:
            model.constraints.add(
                sum(model.x[k,n,t] for k in model.K)
              + sum(model.y[k,n,t] for k in model.K) <= 1)

    for k in model.K:
        for t in model.T:
            model.constraints.add(
                sum(model.x[k,n,t] for n in model.N)
              + sum(model.y[k,n,t] for n in model.N) <= model.c[k,t])

    for k in model.K:
        model.constraints.add(model.e[k,1] == E_0 * C_bat[k-1])
        for t in range(2, T_steps+1):
            model.constraints.add(
                model.e[k,t] == model.e[k,t-1]
                + sum(ch_eff * alpha[n-1] * model.x[k,n,t] for n in model.N)
                - sum(gama[i-1] * model.b[k,i,t] for i in model.I)
                - sum(dch_eff * beta[n-1] * model.y[k,n,t] for n in model.N))

    for t in model.T:
        model.constraints.add(
            sum(ch_eff * alpha[n-1] * model.x[k,n,t]
                for n in model.N for k in model.K) == model.w_buy[t])
        model.constraints.add(
            sum(dch_eff * beta[n-1] * model.y[k,n,t]
                for n in model.N for k in model.K) == model.w_sell[t])

    for k in model.K:
        for t in model.T:
            model.constraints.add(model.e[k,t] >= C_bat[k-1] * E_min)
            model.constraints.add(model.e[k,t] <= C_bat[k-1] * E_max)
        model.constraints.add(model.e[k, T_steps] >= E_end * C_bat[k-1])

    for k in model.K:
        for t in model.T:
            model.constraints.add(
                model.d[k,t] == ((R * C_bat[0] * 1000) / (4 * Ah * V))
                * sum(dch_eff * beta[n-1] * model.y[k,n,t] for n in model.N))

    model.constraints.add(sum(model.u[l] for l in model.L) == 1)
    for t in model.T:
        model.constraints.add(
            sum(alpha[n-1] * model.x[k,n,t] for k in model.K for n in model.N)
            <= sum(U_pow[l-1] * model.u[l] for l in model.L))
        model.constraints.add(
            sum(alpha[n-1] * model.x[k,n,t] for k in model.K for n in model.N)
            <= U_max)

    if count > 1:
        model.constraints.add(
            sum(ch_eff * alpha[n-1] * model.q1[p,k,n,t]
                for p in model.P
                for t in get_period_timesteps(p-1, Q_begin, Q_end, valid_T)
                for k in model.K for n in model.N)
          - sum(dch_eff * beta[n-1] * model.q2[p,k,n,t]
                for p in model.P
                for t in get_period_timesteps(p-1, Q_begin, Q_end, valid_T)
                for k in model.K for n in model.N)
          + sum(model.d[k,t] for k in model.K for t in model.T)
          + sum(model.U_price[l] * model.u[l] for l in model.L)
          <=
            sum(ch_eff * alpha[n-1] * model.y_buy[t]
                for n in model.N for t in model.T)
          + sum(model.d_l[k,t] for k in model.K for t in model.T)
          + sum(model.U_price[l] * model.u_l[l] for l in model.L)
        )

    print('Solving HRP')
    opt = pyo.SolverFactory('appsi_highs')
    opt.config.load_solution = False
    results = opt.solve(model)
    print(f'HRP status: {results.termination_condition}')
    if results.termination_condition in [
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible
    ]:
        model.solutions.load_from(results)
    return model


# ==============================================================================
# LL SOLVER
# ==============================================================================
def solveLL(sc, pho_plus_vals, pho_minus_vals, mi_vals):
    T_steps  = sc['T_steps']
    p_count  = sc['p_count']
    i_count  = sc['i_count']
    k_count  = sc['k_count']
    n_count  = sc['n_count']
    l_count  = sc['l_count']
    Q_begin  = sc['Q_begin']
    Q_end    = sc['Q_end']
    T_start  = sc['T_start']
    T_end    = sc['T_end']
    alpha    = sc['alpha']
    beta     = sc['beta']
    gama     = sc['gama']
    C_bat    = sc['C_bat']
    U_pow    = sc['U_pow']
    U_price  = sc['U_price']
    U_max    = sc['U_max']
    ch_eff   = sc['ch_eff']
    dch_eff  = sc['dch_eff']
    E_0      = sc['E_0']
    E_min    = sc['E_min']
    E_max    = sc['E_max']
    E_end    = sc['E_end']
    R        = sc['R']
    Ah       = sc['Ah']
    V        = sc['V']
    M        = sc['M']

    valid_T = set(range(1, T_steps+1))

    model = pyo.ConcreteModel()
    model.P = pyo.RangeSet(p_count)
    model.I = pyo.RangeSet(i_count)
    model.T = pyo.RangeSet(T_steps)
    model.K = pyo.RangeSet(k_count)
    model.N = pyo.RangeSet(n_count)
    model.L = pyo.RangeSet(l_count)

    model.alpha    = pyo.Param(model.N, initialize=lambda m,n: alpha[n-1])
    model.beta     = pyo.Param(model.N, initialize=lambda m,n: beta[n-1])
    model.C_bat    = pyo.Param(model.K, initialize=lambda m,k: C_bat[k-1])
    model.U_pow    = pyo.Param(model.L, initialize=lambda m,l: U_pow[l-1])
    model.U_price  = pyo.Param(model.L, initialize=lambda m,l: U_price[l-1])
    model.gama     = pyo.Param(model.I, initialize=lambda m,i: gama[i-1])
    model.pho_plus  = pyo.Param(model.P, initialize=pho_plus_vals)
    model.pho_minus = pyo.Param(model.P, initialize=pho_minus_vals)
    model.mi        = pyo.Param(model.P, initialize=mi_vals)

    model.b      = pyo.Var(model.K, model.I, model.T, domain=pyo.Binary)
    model.x      = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.y      = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.c      = pyo.Var(model.K, model.T, domain=pyo.Binary)
    model.u      = pyo.Var(model.L, domain=pyo.Binary)
    model.e      = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.w_buy  = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.w_sell = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.w_cap  = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.d      = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)

    def rule_obj(mod):
        f4 = (sum(mod.pho_plus[p] * mod.w_buy[t]
                  for p in mod.P
                  for t in get_period_timesteps(p-1, Q_begin, Q_end, valid_T))
            - sum(mod.pho_minus[p] * mod.w_sell[t]
                  for p in mod.P
                  for t in get_period_timesteps(p-1, Q_begin, Q_end, valid_T))
            + sum(mod.U_price[l] * mod.u[l] for l in mod.L))
        f5 = sum(mod.mi[p] * mod.w_cap[t]
                 for p in mod.P
                 for t in get_period_timesteps(p-1, Q_begin, Q_end, valid_T))
        f6 = sum(mod.d[k,t] for k in mod.K for t in mod.T)
        return f4 - f5 + f6

    model.obj = pyo.Objective(rule=rule_obj, sense=pyo.minimize)
    model.constraints = pyo.ConstraintList()

    for k in model.K:
        for t in model.T:
            model.constraints.add(
                sum(model.b[k,i,t] for i in model.I) + model.c[k,t] <= 1)

    for i in model.I:
        ts = [t for t in range(T_start[i-1], T_end[i-1]) if t in valid_T]
        for t in ts:
            model.constraints.add(sum(model.b[k,i,t] for k in model.K) == 1)

    for i in model.I:
        for k in model.K:
            ts = [t for t in range(T_start[i-1], T_end[i-1]-1)
                  if t in valid_T and t+1 in valid_T]
            for t in ts:
                model.constraints.add(model.b[k,i,t+1] >= model.b[k,i,t])

    for n in model.N:
        for t in model.T:
            model.constraints.add(
                sum(model.x[k,n,t] for k in model.K)
              + sum(model.y[k,n,t] for k in model.K) <= 1)

    for k in model.K:
        for t in model.T:
            model.constraints.add(
                sum(model.x[k,n,t] for n in model.N)
              + sum(model.y[k,n,t] for n in model.N) <= model.c[k,t])

    for k in model.K:
        model.constraints.add(model.e[k,1] == E_0 * C_bat[k-1])
        for t in range(2, T_steps+1):
            model.constraints.add(
                model.e[k,t] == model.e[k,t-1]
                + sum(ch_eff * alpha[n-1] * model.x[k,n,t] for n in model.N)
                - sum(gama[i-1] * model.b[k,i,t] for i in model.I)
                - sum(dch_eff * beta[n-1] * model.y[k,n,t] for n in model.N))

    for t in model.T:
        model.constraints.add(
            sum(ch_eff * alpha[n-1] * model.x[k,n,t]
                for n in model.N for k in model.K) == model.w_buy[t])
        model.constraints.add(
            sum(dch_eff * beta[n-1] * model.y[k,n,t]
                for n in model.N for k in model.K) == model.w_sell[t])

    for k in model.K:
        for t in model.T:
            model.constraints.add(model.e[k,t] >= C_bat[k-1] * E_min)
            model.constraints.add(model.e[k,t] <= C_bat[k-1] * E_max)
        model.constraints.add(model.e[k, T_steps] >= E_end * C_bat[k-1])

    for k in model.K:
        for t in model.T:
            model.constraints.add(
                model.d[k,t] == ((R * C_bat[0] * 1000) / (4 * Ah * V))
                * sum(dch_eff * beta[n-1] * model.y[k,n,t] for n in model.N))

    for t in model.T:
        model.constraints.add(model.w_cap[t] == 0)

    model.constraints.add(sum(model.u[l] for l in model.L) == 1)
    for t in model.T:
        model.constraints.add(
            sum(alpha[n-1] * model.x[k,n,t] for k in model.K for n in model.N)
            <= sum(U_pow[l-1] * model.u[l] for l in model.L))
        model.constraints.add(
            sum(alpha[n-1] * model.x[k,n,t] for k in model.K for n in model.N)
            <= U_max)

    print('Solving LL')
    opt = pyo.SolverFactory('appsi_highs')
    opt.config.load_solution = False
    results = opt.solve(model)
    print(f'LL status: {results.termination_condition}')
    if results.termination_condition in [
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible
    ]:
        model.solutions.load_from(results)
    return model


# ==============================================================================
# OPTIMIZATION RUNNER (runs in background thread)
# ==============================================================================
def run_optimization(job_id, input_data):
    try:
        data = build_dataframes(input_data)
        price_bounds = get_high_level_prices(data)
        sc = extract_scalars(data)

        UB      = float('inf')
        LB      = float('-inf')
        count   = 1
        epsilon = 0.0001

        y_buy_l  = {t: 0 for t in range(1, sc['T_steps']+1)}
        y_sell_l = {t: 0 for t in range(1, sc['T_steps']+1)}
        d_l      = {(k,t): 0
                    for k in range(1, sc['k_count']+1)
                    for t in range(1, sc['T_steps']+1)}
        u_l      = {l: 0 for l in range(1, sc['l_count']+1)}

        model_LL = None

        while True:
            print(f'--- Iteration {count} ---')

            model_HRP = solveHRP(sc, price_bounds,
                                 y_buy_l, y_sell_l, d_l, u_l, count)

            pho_plus_vals  = {p: pyo.value(model_HRP.pho_plus[p])
                              for p in range(1, sc['p_count']+1)}
            pho_minus_vals = {p: pyo.value(model_HRP.pho_minus[p])
                              for p in range(1, sc['p_count']+1)}
            mi_vals        = {p: pyo.value(model_HRP.mi[p])
                              for p in range(1, sc['p_count']+1)}

            UB = pyo.value(model_HRP.obj)

            model_LL = solveLL(sc, pho_plus_vals, pho_minus_vals, mi_vals)

            y_buy_l  = {t: pyo.value(model_LL.w_buy[t])
                        for t in range(1, sc['T_steps']+1)}
            y_sell_l = {t: pyo.value(model_LL.w_sell[t])
                        for t in range(1, sc['T_steps']+1)}
            d_l      = {(k,t): pyo.value(model_LL.d[k,t])
                        for k in range(1, sc['k_count']+1)
                        for t in range(1, sc['T_steps']+1)}
            u_l      = {l: pyo.value(model_LL.u[l])
                        for l in range(1, sc['l_count']+1)}

            LL_obj = pyo.value(model_LL.obj)
            if LL_obj > LB:
                LB = LL_obj

            print(f'UB={UB:.4f}, LB={LB:.4f}')

            if abs(UB - LB) <= epsilon:
                break

            count += 1
            if count == 3:
                break

        result = {
            "status": "complete",
            "iterations": count,
            "upper_bound": UB,
            "lower_bound": LB,
            "pho_plus":  list(pho_plus_vals.values()),
            "pho_minus": list(pho_minus_vals.values()),
            "mi":        list(mi_vals.values()),
            "w_buy":  [pyo.value(model_LL.w_buy[t])
                       for t in range(1, sc['T_steps']+1)],
            "w_sell": [pyo.value(model_LL.w_sell[t])
                       for t in range(1, sc['T_steps']+1)],
            "energy": [[pyo.value(model_LL.e[k,t])
                        for t in range(1, sc['T_steps']+1)]
                       for k in range(1, sc['k_count']+1)],
        }
        jobs[job_id] = result

    except Exception as e:
        traceback.print_exc()
        jobs[job_id] = {"status": "error", "message": str(e)}


# ==============================================================================
# FLASK ENDPOINTS
# ==============================================================================
@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        input_data = request.json['input']
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "running"}

        thread = threading.Thread(
            target=run_optimization,
            args=(job_id, input_data)
        )
        thread.start()

        return jsonify({"job_id": job_id, "status": "running"})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "active_jobs": len([j for j in jobs.values() if j.get("status") == "running"]),
        "total_jobs": len(jobs)
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
