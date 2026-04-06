from flask import Flask, request, jsonify
import pyomo.environ as pyo
import pandas as pd
import os
import traceback

app = Flask(__name__)


# ==============================================================================
# LLM HOOK - Replace this function with LLM call in the future
# Currently returns default price bounds from the data
# ==============================================================================
def get_high_level_prices(data):
    """
    This function provides the upper-level price bounds to the optimization.
    
    FUTURE: Replace the body of this function with an LLM API call that:
    - Takes market conditions, historical data, and system state as input
    - Returns price bounds decided by the LLM agent
    
    Returns a dict with:
        X_up, X_low: energy price upper/lower bounds per period
        Mi_up, Mi_low: capacity price upper/lower bounds per period
        X_avg: average energy price target
        Mi_avg: average capacity price target
        delta: sell/buy price ratio
    """
    X_up  = data['Average prices']['Max price'].values.flatten()
    X_low = data['Average prices']['Min price'].values.flatten()
    Mi_up = data['Average prices']['Max cap'].values.flatten()
    Mi_low = data['Average prices']['Min cap'].values.flatten()

    return {
        'X_up':   X_up,
        'X_low':  X_low,
        'Mi_up':  Mi_up,
        'Mi_low': Mi_low,
        'X_avg':  0.12,   # average electricity price target (€/kWh)
        'Mi_avg': 0.015,  # average capacity price target (€/kW)
        'delta':  0.8,    # sell price = delta * buy price
    }


# ==============================================================================
# DATA BUILDER
# ==============================================================================
def build_dataframes(input_data):
    buses = pd.DataFrame(input_data['buses']).rename(columns={
        'bus_kwh': 'Bus (kWh)'
    })
    chargers = pd.DataFrame(input_data['chargers']).rename(columns={
        'charger_kwhmin': 'Charger (kWh/min)',
        'charger_kw':     'Charger (kW)',
        'max_power_kw':   'Max Power (kW)'
    })
    trip_time = pd.DataFrame(input_data['trip_time']).rename(columns={
        'time_begin_min':  'Time begin (min)',
        'time_finish_min': 'Time finish (min)'
    })
    energy_consumption = pd.DataFrame(input_data['energy_consumption']).rename(columns={
        'uncertain_energy_kwhkmmin': 'Uncertain energy (kWh/km*min)'
    })
    prices = pd.DataFrame(input_data['prices']).rename(columns={
        'spot_market':    'Spot Market',
        'capacity_price': 'Capacity price'
    })
    power_price = pd.DataFrame(input_data['power_price']).rename(columns={
        'power': 'Power',
        'price': 'Price'
    })
    average_prices = pd.DataFrame(input_data['average_prices']).rename(columns={
        'max_price': 'Max price',
        'min_price': 'Min price',
        'max_cap':   'Max cap',
        'min_cap':   'Min cap'
    })
    periods = pd.DataFrame(input_data['periods']).rename(columns={
        'period': 'Period',
        'begin':  'Begin',
        'end':    'End',
        'len':    'Len'
    })
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
# HPR SOLVER (Upper Level - Aggregator)
# ==============================================================================
def solveHRP(data, price_bounds, y_buy, y_sell, y_cap, d_l, u_l, count):

    model = pyo.ConcreteModel()

    # --- Raw data ---
    T_steps = 24
    delta_t = 4
    M = 10000

    PI     = data['Prices']['Spot Market'].values.flatten()
    PI_cap = data['Prices']['Capacity price'].values.flatten()

    X_up   = price_bounds['X_up']
    X_low  = price_bounds['X_low']
    X_avg  = price_bounds['X_avg']
    Mi_up  = price_bounds['Mi_up']
    Mi_low = price_bounds['Mi_low']
    Mi_avg = price_bounds['Mi_avg']
    delta  = price_bounds['delta']

    p_count = len(data['Periods']['Period'])
    Q_begin = [int(x) for x in data['Periods']['Begin'].tolist()]
    Q_end   = [int(x) for x in data['Periods']['End'].tolist()]
    Q_len   = [int(x) for x in data['Periods']['Len'].tolist()]

    i_count = len(data['Trip time']['Time begin (min)'])
    k_count = len(data['Buses']['Bus (kWh)'])
    n_count = len(data['Chargers']['Charger (kWh/min)'])
    l_count = len(data['Power price']['Power'])

    T_start = [int(x) for x in data['Trip time']['Time begin (min)'].tolist()]
    T_end   = [int(x) for x in data['Trip time']['Time finish (min)'].tolist()]
    alpha   = data['Chargers']['Charger (kWh/min)'].tolist()
    beta    = data['Chargers']['Charger (kWh/min)'].tolist()
    gama    = data['Energy consumption']['Uncertain energy (kWh/km*min)'].tolist()
    C_bat   = data['Buses']['Bus (kWh)'].tolist()
    U_pow   = data['Power price']['Power'].tolist()
    U_price = data['Power price']['Price'].tolist()
    U_max   = data['Chargers']['Max Power (kW)'].tolist()[0]
    U_cap   = 200

    ch_eff  = 0.90
    dch_eff = 1.0
    E_0     = 0.2
    E_min   = 0.2
    E_max   = 1.0
    E_end   = 0.2
    R       = 130
    Ah      = 905452
    V       = 512
    d_on    = 1
    d_off   = 1
    d_cap   = 1

    # --- Sets ---
    model.P = pyo.RangeSet(p_count)
    model.I = pyo.RangeSet(i_count)
    model.T = pyo.RangeSet(T_steps)
    model.K = pyo.RangeSet(k_count)
    model.N = pyo.RangeSet(n_count)
    model.L = pyo.RangeSet(l_count)

    # --- Parameters ---
    model.PI     = pyo.Param(model.T, initialize=lambda m, t: PI[t-1])
    model.PI_cap = pyo.Param(model.T, initialize=lambda m, t: PI_cap[t-1])
    model.Q_begin = pyo.Param(model.P, initialize=lambda m, p: Q_begin[p-1])
    model.Q_end   = pyo.Param(model.P, initialize=lambda m, p: Q_end[p-1])
    model.Q_len   = pyo.Param(model.P, initialize=lambda m, p: Q_len[p-1])
    model.X_low  = pyo.Param(model.P, initialize=lambda m, p: X_low[p-1])
    model.X_up   = pyo.Param(model.P, initialize=lambda m, p: X_up[p-1])
    model.Mi_low = pyo.Param(model.P, initialize=lambda m, p: Mi_low[p-1])
    model.Mi_up  = pyo.Param(model.P, initialize=lambda m, p: Mi_up[p-1])
    model.T_start = pyo.Param(model.I, initialize=lambda m, i: T_start[i-1])
    model.T_end   = pyo.Param(model.I, initialize=lambda m, i: T_end[i-1])
    model.alpha  = pyo.Param(model.N, initialize=lambda m, n: alpha[n-1])
    model.beta   = pyo.Param(model.N, initialize=lambda m, n: beta[n-1])
    model.gama   = pyo.Param(model.I, initialize=lambda m, i: gama[i-1], mutable=True)
    model.C_bat  = pyo.Param(model.K, initialize=lambda m, k: C_bat[k-1])
    model.U_pow  = pyo.Param(model.L, initialize=lambda m, l: U_pow[l-1])
    model.U_price = pyo.Param(model.L, initialize=lambda m, l: U_price[l-1])
    model.U_max  = pyo.Param(initialize=U_max)
    model.U_cap  = pyo.Param(initialize=U_cap)
    model.ch_eff  = pyo.Param(initialize=ch_eff)
    model.dch_eff = pyo.Param(initialize=dch_eff)
    model.E_0    = pyo.Param(initialize=E_0)
    model.E_min  = pyo.Param(initialize=E_min)
    model.E_max  = pyo.Param(initialize=E_max)
    model.E_end  = pyo.Param(initialize=E_end)
    model.R      = pyo.Param(initialize=R)
    model.Ah     = pyo.Param(initialize=Ah)
    model.V      = pyo.Param(initialize=V)
    model.delta  = pyo.Param(initialize=delta)

    model.y_buy  = pyo.Param(model.T, initialize=y_buy)
    model.y_sell = pyo.Param(model.T, initialize=y_sell)
    model.y_cap  = pyo.Param(model.T, initialize=y_cap)
    model.d_l    = pyo.Param(model.K, model.T, initialize=d_l)
    model.u_l    = pyo.Param(model.L, initialize=u_l)

    # --- Decision variables ---
    model.pho_plus  = pyo.Var(model.P, domain=pyo.NonNegativeReals)
    model.pho_minus = pyo.Var(model.P, domain=pyo.NonNegativeReals)
    model.mi        = pyo.Var(model.P, domain=pyo.NonNegativeReals)

    model.e      = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.w_buy  = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.w_sell = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.w_cap  = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.d      = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.a      = pyo.Var(model.T, domain=pyo.Binary)
    model.b      = pyo.Var(model.K, model.I, model.T, domain=pyo.Binary)
    model.x      = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.y      = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.z      = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.c      = pyo.Var(model.K, model.T, domain=pyo.Binary)
    model.u      = pyo.Var(model.L, domain=pyo.Binary)
    model.z_up   = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.z_down = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)

    # --- Linearization auxiliary variables (McCormick envelopes) ---
    # q1[p,k,n,t] = pho_plus[p] * x[k,n,t]
    # q2[p,k,n,t] = pho_minus[p] * y[k,n,t]
    # q3[p,k,n,t] = mi[p] * z_up[k,n,t]
    # q4[p,k,n,t] = mi[p] * z_down[k,n,t]
    model.q1 = pyo.Var(model.P, model.K, model.N, model.T, within=pyo.NonNegativeReals)
    model.q2 = pyo.Var(model.P, model.K, model.N, model.T, within=pyo.NonNegativeReals)
    model.q3 = pyo.Var(model.P, model.K, model.N, model.T, within=pyo.NonNegativeReals)
    model.q4 = pyo.Var(model.P, model.K, model.N, model.T, within=pyo.NonNegativeReals)

    model.constraints = pyo.ConstraintList()

    # --- McCormick envelope constraints ---
    for p in model.P:
        rho_up = X_up[p-1]
        rho_lo = X_low[p-1]
        mu_up  = Mi_up[p-1]
        mu_lo  = Mi_low[p-1]
        for k in model.K:
            for n in model.N:
                for t in model.T:
                    # q1 = pho_plus[p] * x[k,n,t]
                    model.constraints.add(model.q1[p,k,n,t] <= rho_up * model.x[k,n,t])
                    model.constraints.add(model.q1[p,k,n,t] <= model.pho_plus[p])
                    model.constraints.add(model.q1[p,k,n,t] >= model.pho_plus[p] - (1 - model.x[k,n,t]) * rho_up)

                    # q2 = pho_minus[p] * y[k,n,t]
                    model.constraints.add(model.q2[p,k,n,t] <= delta * rho_up * model.y[k,n,t])
                    model.constraints.add(model.q2[p,k,n,t] <= model.pho_minus[p])
                    model.constraints.add(model.q2[p,k,n,t] >= model.pho_minus[p] - (1 - model.y[k,n,t]) * delta * rho_up)

                    # q3 = mi[p] * z_up[k,n,t]
                    model.constraints.add(model.q3[p,k,n,t] <= mu_up * model.z_up[k,n,t])
                    model.constraints.add(model.q3[p,k,n,t] <= model.mi[p])
                    model.constraints.add(model.q3[p,k,n,t] >= model.mi[p] - (1 - model.z_up[k,n,t]) * mu_up)

                    # q4 = mi[p] * z_down[k,n,t]
                    model.constraints.add(model.q4[p,k,n,t] <= mu_up * model.z_down[k,n,t])
                    model.constraints.add(model.q4[p,k,n,t] <= model.mi[p])
                    model.constraints.add(model.q4[p,k,n,t] >= model.mi[p] - (1 - model.z_down[k,n,t]) * mu_up)

    # --- Linearized objective (A7-A10 from paper) ---
    def rule_obj(mod):
        # f1: energy trading with PTO (linearized)
        f1 = (sum(ch_eff * mod.alpha[n] * mod.q1[p,k,n,t]
                  for p in mod.P for t in range(Q_begin[p-1], Q_end[p-1]+1)
                  for k in mod.K for n in mod.N)
            - sum((1/dch_eff) * mod.beta[n] * mod.q2[p,k,n,t]
                  for p in mod.P for t in range(Q_begin[p-1], Q_end[p-1]+1)
                  for k in mod.K for n in mod.N))

        # f2: reserve market (linearized)
        f2 = (sum(mod.PI_cap[t] * mod.w_cap[t] for t in mod.T)
            - sum((1/dch_eff) * mod.beta[n] * mod.q3[p,k,n,t]
                  for p in mod.P for t in range(Q_begin[p-1], Q_end[p-1]+1)
                  for k in mod.K for n in mod.N)
            - sum(ch_eff * mod.alpha[n] * mod.q4[p,k,n,t]
                  for p in mod.P for t in range(Q_begin[p-1], Q_end[p-1]+1)
                  for k in mod.K for n in mod.N))

        # f3: spot market trading
        f3 = sum(mod.PI[t] * (mod.w_sell[t] - mod.w_buy[t]) for t in mod.T)

        return f1 + f2 + f3

    model.obj = pyo.Objective(rule=rule_obj, sense=pyo.maximize)

    # --- Price constraints ---
    for p in model.P:
        model.constraints.add(model.pho_plus[p] >= model.X_low[p])
        model.constraints.add(model.pho_plus[p] <= model.X_up[p])
        model.constraints.add(model.pho_minus[p] == model.pho_plus[p] * delta)
        model.constraints.add(model.mi[p] >= model.Mi_low[p])
        model.constraints.add(model.mi[p] <= model.Mi_up[p])

    model.constraints.add(
        (1/T_steps) * sum(Q_len[p-1] * model.mi[p] for p in model.P) <= Mi_avg)
    model.constraints.add(
        (1/T_steps) * sum(Q_len[p-1] * model.pho_plus[p] for p in model.P) <= X_avg)

    # --- Operational constraints ---
    for k in model.K:
        for t in model.T:
            model.constraints.add(
                sum(model.b[k,i,t] for i in model.I) + model.c[k,t] <= 1)

    for i in model.I:
        for t in range(T_start[i-1], T_end[i-1]):
            model.constraints.add(sum(model.b[k,i,t] for k in model.K) == 1)

    for i in model.I:
        for k in model.K:
            for t in range(T_start[i-1], T_end[i-1]-1):
                model.constraints.add(model.b[k,i,t+1] >= model.b[k,i,t])

    for n in model.N:
        for t in model.T:
            model.constraints.add(
                sum(model.x[k,n,t] for k in model.K)
              + sum(model.y[k,n,t] for k in model.K)
              + sum(model.z[k,n,t] for k in model.K) <= 1)

    for k in model.K:
        for t in model.T:
            model.constraints.add(
                sum(model.x[k,n,t] for n in model.N)
              + sum(model.y[k,n,t] for n in model.N)
              + sum(model.z[k,n,t] for n in model.N) <= model.c[k,t])

    for k in model.K:
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

    model.constraints.add(
        sum(dch_eff * beta[n-1] * model.y[k,n,1]
            for n in model.N for k in model.K) == 0)

    for k in model.K:
        for n in model.N:
            for t in range(2, T_steps - d_on):
                model.constraints.add(
                    1 - model.x[k,n,t] + model.x[k,n,t-1]
                    + (1/d_on) * sum(model.x[k,n,j] for j in range(t, t+d_on)) >= 1)
            for t in range(T_steps - d_on + 1, T_steps):
                model.constraints.add(
                    1 - model.x[k,n,t] + model.x[k,n,t-1]
                    + (1/(T_steps-t+1)) * sum(model.x[k,n,j] for j in range(t, T_steps)) >= 1)
            for t in range(2, T_steps - d_off):
                model.constraints.add(
                    1 - model.x[k,n,t] + model.x[k,n,t-1]
                    + (1/d_off) * sum(model.x[k,n,j] for j in range(t, t+d_off)) <= 2)
            for t in range(T_steps - d_off + 1, T_steps):
                model.constraints.add(
                    1 - model.x[k,n,t] + model.x[k,n,t-1]
                    + (1/(T_steps-t+1)) * sum(model.x[k,n,j] for j in range(t, T_steps)) <= 2)

    model.constraints.add(sum(model.u[l] for l in model.L) == 1)

    for t in model.T:
        model.constraints.add(
            sum(alpha[n-1] * model.x[k,n,t] for k in model.K for n in model.N)
            <= sum(U_pow[l-1] * model.u[l] for l in model.L))
        model.constraints.add(
            sum(alpha[n-1] * model.x[k,n,t] for k in model.K for n in model.N)
            <= U_max)

    for k in model.K:
        for t in model.T:
            model.constraints.add(model.e[k,t] >= C_bat[k-1] * E_min)
            model.constraints.add(
                E_max * C_bat[k-1] >= model.e[k,t]
                + sum(ch_eff * alpha[n-1] * model.x[k,n,t] for n in model.N))
        model.constraints.add(model.e[k,1] == E_0 * C_bat[k-1])
        model.constraints.add(
            model.e[k, T_steps-1]
            + sum(ch_eff * alpha[n-1] * model.x[k,n,T_steps] for n in model.N)
            >= E_end * C_bat[k-1])
        for t in model.T:
            model.constraints.add(
                model.d[k,t] == ((R * C_bat[0] * 1000) / (4 * Ah * V))
                * sum(dch_eff * beta[n-1] * model.y[k,n,t] for n in model.N))

    # --- Reserve constraints ---
    for n in model.N:
        for t in model.T:
            model.constraints.add(
                sum(model.z_up[k,n,t] for k in model.K)
              + sum(model.z_down[k,n,t] for k in model.K)
              == sum(model.z[k,n,t] for k in model.K))

    for t in model.T:
        model.constraints.add(
            sum(ch_eff * alpha[n-1] * model.z_down[k,n,t]
                for k in model.K for n in model.N) * delta_t
          + sum(dch_eff * beta[n-1] * model.z_up[k,n,t]
                for k in model.K for n in model.N) * delta_t
          == model.w_cap[t])
        model.constraints.add(model.w_cap[t] >= U_cap - M*(1 - model.a[t]))
        model.constraints.add(model.w_cap[t] <= M * model.a[t])

    for k in model.K:
        for t in model.T:
            model.constraints.add(
                E_max * C_bat[k-1] >= model.e[k,t]
                + sum(ch_eff * alpha[n-1] * model.z_down[k,n,t] for n in model.N))
            model.constraints.add(
                E_min * C_bat[k-1] <= model.e[k,t]
                - sum(dch_eff * beta[n-1] * model.z_up[k,n,t]
                      for k in model.K for n in model.N))

    model.constraints.add(
        sum(model.z_up[k,n,t] for k in model.K for n in model.N for t in model.T)
     == sum(model.z_down[k,n,t] for k in model.K for n in model.N for t in model.T))

    for k in model.K:
        for n in model.N:
            for t in range(2, T_steps - d_cap):
                model.constraints.add(
                    1 - model.z[k,n,t] + model.z[k,n,t-1]
                    + (1/d_cap) * sum(model.z[k,n,j] for j in range(t, t+d_cap)) >= 1)
            for t in range(T_steps - d_cap + 1, T_steps):
                model.constraints.add(
                    1 - model.z[k,n,t] + model.z[k,n,t-1]
                    + (1/(T_steps-t+1)) * sum(model.z[k,n,j] for j in range(t, T_steps)) >= 1)

    # --- HPR constraint (added from iteration 2 onwards) ---
    if count > 1:
        model.constraints.add(
            sum(ch_eff * alpha[n-1] * model.q1[p,k,n,t]
                for p in model.P for t in range(Q_begin[p-1], Q_end[p-1]+1)
                for k in model.K for n in model.N)
          - sum((1/dch_eff) * beta[n-1] * model.q2[p,k,n,t]
                for p in model.P for t in range(Q_begin[p-1], Q_end[p-1]+1)
                for k in model.K for n in model.N)
          - sum((1/dch_eff) * beta[n-1] * model.q3[p,k,n,t]
                for p in model.P for t in range(Q_begin[p-1], Q_end[p-1]+1)
                for k in model.K for n in model.N)
          - sum(ch_eff * alpha[n-1] * model.q4[p,k,n,t]
                for p in model.P for t in range(Q_begin[p-1], Q_end[p-1]+1)
                for k in model.K for n in model.N)
          + sum(model.d[k,t] for k in model.K for t in model.T)
          + sum(model.U_price[l] * model.u[l] for l in model.L)
          <=
            sum(ch_eff * alpha[n-1] * model.y_buy[t]
                for n in model.N for t in model.T)  # simplified LL bound
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
# LL SOLVER (Lower Level - Public Transportation Operator)
# ==============================================================================
def solveLL(data, pho_plus, pho_minus, mi):

    model = pyo.ConcreteModel()

    T_steps = 24
    delta_t = 4
    M = 10000

    p_count = len(data['Periods']['Period'])
    Q_begin = [int(x) for x in data['Periods']['Begin'].tolist()]
    Q_end   = [int(x) for x in data['Periods']['End'].tolist()]

    i_count = len(data['Trip time']['Time begin (min)'])
    k_count = len(data['Buses']['Bus (kWh)'])
    n_count = len(data['Chargers']['Charger (kWh/min)'])
    l_count = len(data['Power price']['Power'])

    T_start = [int(x) for x in data['Trip time']['Time begin (min)'].tolist()]
    T_end   = [int(x) for x in data['Trip time']['Time finish (min)'].tolist()]
    alpha   = data['Chargers']['Charger (kWh/min)'].tolist()
    beta    = data['Chargers']['Charger (kWh/min)'].tolist()
    gama    = data['Energy consumption']['Uncertain energy (kWh/km*min)'].tolist()
    C_bat   = data['Buses']['Bus (kWh)'].tolist()
    U_pow   = data['Power price']['Power'].tolist()
    U_price = data['Power price']['Price'].tolist()
    U_max   = data['Chargers']['Max Power (kW)'].tolist()[0]
    U_cap   = 200

    ch_eff  = 0.90
    dch_eff = 1.0
    E_0     = 0.2
    E_min   = 0.2
    E_max   = 1.0
    E_end   = 0.2
    R       = 130
    Ah      = 905452
    V       = 512
    d_on    = 1
    d_off   = 1
    d_cap   = 1

    # Extract pho_plus, pho_minus, mi values from HRP model
    pho_plus_vals  = {p: pyo.value(pho_plus[p])  for p in range(1, p_count+1)}
    pho_minus_vals = {p: pyo.value(pho_minus[p]) for p in range(1, p_count+1)}
    mi_vals        = {p: pyo.value(mi[p])         for p in range(1, p_count+1)}

    model.P = pyo.RangeSet(p_count)
    model.I = pyo.RangeSet(i_count)
    model.T = pyo.RangeSet(T_steps)
    model.K = pyo.RangeSet(k_count)
    model.N = pyo.RangeSet(n_count)
    model.L = pyo.RangeSet(l_count)

    model.T_start  = pyo.Param(model.I, initialize=lambda m, i: T_start[i-1])
    model.T_end    = pyo.Param(model.I, initialize=lambda m, i: T_end[i-1])
    model.alpha    = pyo.Param(model.N, initialize=lambda m, n: alpha[n-1])
    model.beta     = pyo.Param(model.N, initialize=lambda m, n: beta[n-1])
    model.gama     = pyo.Param(model.I, initialize=lambda m, i: gama[i-1])
    model.C_bat    = pyo.Param(model.K, initialize=lambda m, k: C_bat[k-1])
    model.U_pow    = pyo.Param(model.L, initialize=lambda m, l: U_pow[l-1])
    model.U_price  = pyo.Param(model.L, initialize=lambda m, l: U_price[l-1])
    model.U_max    = pyo.Param(initialize=U_max)
    model.U_cap    = pyo.Param(initialize=U_cap)
    model.Q_begin  = pyo.Param(model.P, initialize=lambda m, p: Q_begin[p-1])
    model.Q_end    = pyo.Param(model.P, initialize=lambda m, p: Q_end[p-1])
    model.pho_plus  = pyo.Param(model.P, initialize=pho_plus_vals)
    model.pho_minus = pyo.Param(model.P, initialize=pho_minus_vals)
    model.mi        = pyo.Param(model.P, initialize=mi_vals)

    model.b      = pyo.Var(model.K, model.I, model.T, domain=pyo.Binary)
    model.x      = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.y      = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.z      = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.u      = pyo.Var(model.L, domain=pyo.Binary)
    model.c      = pyo.Var(model.K, model.T, domain=pyo.Binary)
    model.a      = pyo.Var(model.T, domain=pyo.Binary)
    model.z_up   = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.z_down = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.e      = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.w_buy  = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.w_sell = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.w_cap  = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.d      = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)

    # LL objective: minimize DOC = f4 - f5 + f6
    def rule_obj(mod):
        f4 = (sum(mod.pho_plus[p] * mod.w_buy[t]
                  for p in mod.P for t in range(Q_begin[p-1], Q_end[p-1]+1))
            - sum(mod.pho_minus[p] * mod.w_sell[t]
                  for p in mod.P for t in range(Q_begin[p-1], Q_end[p-1]+1))
            + sum(mod.U_price[l] * mod.u[l] for l in mod.L))
        f5 = sum(mod.mi[p] * mod.w_cap[t]
                 for p in mod.P for t in range(Q_begin[p-1], Q_end[p-1]+1))
        f6 = sum(mod.d[k,t] for k in mod.K for t in mod.T)
        return f4 - f5 + f6

    model.obj = pyo.Objective(rule=rule_obj, sense=pyo.minimize)

    model.constraints = pyo.ConstraintList()

    for k in model.K:
        for t in model.T:
            model.constraints.add(
                sum(model.b[k,i,t] for i in model.I) + model.c[k,t] <= 1)

    for i in model.I:
        for t in range(T_start[i-1], T_end[i-1]):
            model.constraints.add(sum(model.b[k,i,t] for k in model.K) == 1)

    for i in model.I:
        for k in model.K:
            for t in range(T_start[i-1], T_end[i-1]-1):
                model.constraints.add(model.b[k,i,t+1] >= model.b[k,i,t])

    for n in model.N:
        for t in model.T:
            model.constraints.add(
                sum(model.x[k,n,t] for k in model.K)
              + sum(model.y[k,n,t] for k in model.K)
              + sum(model.z[k,n,t] for k in model.K) <= 1)

    for k in model.K:
        for t in model.T:
            model.constraints.add(
                sum(model.x[k,n,t] for n in model.N)
              + sum(model.y[k,n,t] for n in model.N)
              + sum(model.z[k,n,t] for n in model.N) <= model.c[k,t])

    for k in model.K:
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

    model.constraints.add(
        sum(dch_eff * beta[n-1] * model.y[k,n,1]
            for n in model.N for k in model.K) == 0)

    for k in model.K:
        for n in model.N:
            for t in range(2, T_steps - d_on):
                model.constraints.add(
                    1 - model.x[k,n,t] + model.x[k,n,t-1]
                    + (1/d_on) * sum(model.x[k,n,j] for j in range(t, t+d_on)) >= 1)
            for t in range(T_steps - d_on + 1, T_steps):
                model.constraints.add(
                    1 - model.x[k,n,t] + model.x[k,n,t-1]
                    + (1/(T_steps-t+1)) * sum(model.x[k,n,j] for j in range(t, T_steps)) >= 1)
            for t in range(2, T_steps - d_off):
                model.constraints.add(
                    1 - model.x[k,n,t] + model.x[k,n,t-1]
                    + (1/d_off) * sum(model.x[k,n,j] for j in range(t, t+d_off)) <= 2)
            for t in range(T_steps - d_off + 1, T_steps):
                model.constraints.add(
                    1 - model.x[k,n,t] + model.x[k,n,t-1]
                    + (1/(T_steps-t+1)) * sum(model.x[k,n,j] for j in range(t, T_steps)) <= 2)

    model.constraints.add(sum(model.u[l] for l in model.L) == 1)

    for t in model.T:
        model.constraints.add(
            sum(alpha[n-1] * model.x[k,n,t] for k in model.K for n in model.N)
            <= sum(U_pow[l-1] * model.u[l] for l in model.L))
        model.constraints.add(
            sum(alpha[n-1] * model.x[k,n,t] for k in model.K for n in model.N)
            <= U_max)

    for k in model.K:
        for t in model.T:
            model.constraints.add(model.e[k,t] >= C_bat[k-1] * E_min)
            model.constraints.add(
                E_max * C_bat[k-1] >= model.e[k,t]
                + sum(ch_eff * alpha[n-1] * model.x[k,n,t] for n in model.N))
        model.constraints.add(model.e[k,1] == E_0 * C_bat[k-1])
        model.constraints.add(
            model.e[k, T_steps-1]
            + sum(ch_eff * alpha[n-1] * model.x[k,n,T_steps] for n in model.N)
            >= E_end * C_bat[k-1])
        for t in model.T:
            model.constraints.add(
                model.d[k,t] == ((R * C_bat[0] * 1000) / (4 * Ah * V))
                * sum(dch_eff * beta[n-1] * model.y[k,n,t] for n in model.N))

    for n in model.N:
        for t in model.T:
            model.constraints.add(
                sum(model.z_up[k,n,t] for k in model.K)
              + sum(model.z_down[k,n,t] for k in model.K)
              == sum(model.z[k,n,t] for k in model.K))

    for t in model.T:
        model.constraints.add(
            sum(ch_eff * alpha[n-1] * model.z_down[k,n,t]
                for k in model.K for n in model.N) * delta_t
          + sum(dch_eff * beta[n-1] * model.z_up[k,n,t]
                for k in model.K for n in model.N) * delta_t
          == model.w_cap[t])
        model.constraints.add(model.w_cap[t] >= U_cap - M*(1 - model.a[t]))
        model.constraints.add(model.w_cap[t] <= M * model.a[t])

    for k in model.K:
        for t in model.T:
            model.constraints.add(
                E_max * C_bat[k-1] >= model.e[k,t]
                + sum(ch_eff * alpha[n-1] * model.z_down[k,n,t] for n in model.N))
            model.constraints.add(
                E_min * C_bat[k-1] <= model.e[k,t]
                - sum(dch_eff * beta[n-1] * model.z_up[k,n,t]
                      for k in model.K for n in model.N))

    model.constraints.add(
        sum(model.z_up[k,n,t] for k in model.K for n in model.N for t in model.T)
     == sum(model.z_down[k,n,t] for k in model.K for n in model.N for t in model.T))

    for k in model.K:
        for n in model.N:
            for t in range(2, T_steps - d_cap):
                model.constraints.add(
                    1 - model.z[k,n,t] + model.z[k,n,t-1]
                    + (1/d_cap) * sum(model.z[k,n,j] for j in range(t, t+d_cap)) >= 1)
            for t in range(T_steps - d_cap + 1, T_steps):
                model.constraints.add(
                    1 - model.z[k,n,t] + model.z[k,n,t-1]
                    + (1/(T_steps-t+1)) * sum(model.z[k,n,j] for j in range(t, T_steps)) >= 1)

    print('Solving LL')
    opt = pyo.SolverFactory('appsi_highs')
    opt.config.load_solution = False
    results = opt.solve(model, tee=True)
    print(f'LL status: {results.termination_condition}')

    if results.termination_condition in [
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible
    ]:
        model.solutions.load_from(results)

    return model


# ==============================================================================
# FLASK ENDPOINT
# ==============================================================================
@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        input_data = request.json['input']
        data = build_dataframes(input_data)

        # --- Get high-level price bounds (swap this for LLM call in future) ---
        price_bounds = get_high_level_prices(data)

        # --- Bilevel optimization loop ---
        UB = float('inf')
        LB = float('-inf')
        k = 1
        y_buy_l = y_sell_l = y_cap_l = 0
        d_l = 0
        u_l = 0
        epsilon = 0.0001

        while UB != LB:
            print(f'Starting iteration: {k}')

            model_HRP = solveHRP(data, price_bounds,
                                 y_buy_l, y_sell_l, y_cap_l, d_l, u_l, k)

            pho_plus  = model_HRP.pho_plus
            pho_minus = model_HRP.pho_minus
            mi        = model_HRP.mi
            y_buy_u   = model_HRP.w_buy
            y_sell_u  = model_HRP.w_sell
            y_cap_u   = model_HRP.w_cap
            UB        = pyo.value(model_HRP.obj)

            model_LL = solveLL(data, pho_plus, pho_minus, mi)

            y_buy_l  = model_LL.w_buy
            y_sell_l = model_LL.w_sell
            y_cap_l  = model_LL.w_cap
            d_l      = model_LL.d
            u_l      = model_LL.u

            f_y_u = (
                sum(pyo.value(model_LL.pho_plus[p]) * pyo.value(y_buy_u[t])
                    for p in model_LL.P
                    for t in range(int(pyo.value(model_LL.Q_begin[p])),
                                   int(pyo.value(model_LL.Q_end[p]))+1))
              - sum(pyo.value(model_LL.pho_minus[p]) * pyo.value(y_sell_u[t])
                    for p in model_LL.P
                    for t in range(int(pyo.value(model_LL.Q_begin[p])),
                                   int(pyo.value(model_LL.Q_end[p]))+1))
              + sum(pyo.value(model_LL.d[kk, t])
                    for kk in model_LL.K for t in model_LL.T)
              - sum(pyo.value(model_LL.mi[p]) * pyo.value(y_cap_u[t])
                    for p in model_LL.P
                    for t in range(int(pyo.value(model_LL.Q_begin[p])),
                                   int(pyo.value(model_LL.Q_end[p]))+1))
              + sum(pyo.value(model_LL.U_price[l]) * pyo.value(model_LL.u[l])
                    for l in model_LL.L)
            )

            LL_obj = pyo.value(model_LL.obj)
            if f_y_u - LL_obj <= epsilon:
                LB = UB

            print(f'UB={UB}, LB={LB}')
            k += 1
            if k == 2:
                break

        result = {
            "status": "optimal",
            "iterations": k - 1,
            "upper_bound": UB,
            "lower_bound": LB,
            "price_bounds_used": {
                "X_up":  price_bounds['X_up'].tolist(),
                "X_low": price_bounds['X_low'].tolist(),
            },
            "w_buy":  [pyo.value(model_LL.w_buy[t])  for t in model_LL.T],
            "w_sell": [pyo.value(model_LL.w_sell[t]) for t in model_LL.T],
            "w_cap":  [pyo.value(model_LL.w_cap[t])  for t in model_LL.T],
            "energy": [[pyo.value(model_LL.e[kk, t])
                        for t in model_LL.T] for kk in model_LL.K],
            "pho_plus":  [pyo.value(pho_plus[p])  for p in model_HRP.P],
            "pho_minus": [pyo.value(pho_minus[p]) for p in model_HRP.P],
            "mi":        [pyo.value(mi[p])         for p in model_HRP.P],
        }
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
