from flask import Flask, request, jsonify
import pyomo.environ as pyo
import pandas as pd
import os

app = Flask(__name__)


def build_dataframes(input_data):
    return {
        'Buses':              pd.DataFrame(input_data['buses']),
        'Chargers':           pd.DataFrame(input_data['chargers']),
        'Trip time':          pd.DataFrame(input_data['trip_time']),
        'Energy consumption': pd.DataFrame(input_data['energy_consumption']),
        'Prices':             pd.DataFrame(input_data['prices']),
        'Power price':        pd.DataFrame(input_data['power_price']),
        'Average prices':     pd.DataFrame(input_data['average_prices']),
        'Periods':            pd.DataFrame(input_data['periods']),
    }


def solveHRP(data, y_buy, y_sell, y_cap, d_l, u_l, count):

    model = pyo.ConcreteModel()

    t = len(data['Prices']['Spot Market'])
    PI = data['Prices']['Spot Market'].values.flatten()
    PI_cap = data['Prices']['Capacity price'].values.flatten()
    l = len(data['Power price']['Power'])

    X_up = data['Average prices']['Max price'].values.flatten()
    X_low = data['Average prices']['Min price'].values.flatten()
    X_avg = 0.1
    Mi_up = data['Average prices']['Max cap'].values.flatten()
    Mi_low = data['Average prices']['Min cap'].values.flatten()
    Mi_avg = 0.01

    p = len(data['Periods']['Period'])
    Q_begin = data['Periods']['Begin'].tolist()
    Q_end = data['Periods']['End'].tolist()
    Q_len = data['Periods']['Len'].tolist()

    i = len(data['Trip time']['Time begin (min)'])
    k = len(data['Buses']['Bus (kWh)'])
    n = len(data['Chargers']['Charger (kWh/min)'])
    T_start = data['Trip time']['Time begin (min)'].tolist()
    T_start = [int(x) for x in T_start]
    T_end = data['Trip time']['Time finish (min)'].tolist()
    T_end = [int(x) for x in T_end]
    alpha = data['Chargers']['Charger (kWh/min)'].tolist()
    beta = data['Chargers']['Charger (kWh/min)'].tolist()
    gama = data['Energy consumption']['Uncertain energy (kWh/km*min)'].tolist()
    ch_eff = 0.90
    dch_eff = 1
    E_0 = 0.2
    E_min = 0.2
    E_max = 1
    E_end = 0.2
    C_bat = data['Buses']['Bus (kWh)'].tolist()

    U_pow = data['Power price']['Power'].tolist()
    U_price = data['Power price']['Price'].tolist()
    U_max = data['Chargers']['Max Power (kW)'].tolist()
    U_max = U_max[0]
    U_cap = 200

    d_on = 1
    d_off = 1
    d_cap = 1

    R = 130
    Ah = 905452
    V = 512
    T = 96
    delta_t = 4
    delta = 0.8
    M = 10000

    model.P = pyo.RangeSet(p)
    model.I = pyo.RangeSet(i)
    model.T = pyo.RangeSet(t)
    model.K = pyo.RangeSet(k)
    model.N = pyo.RangeSet(n)
    model.L = pyo.RangeSet(l)

    model.PI = pyo.Param(model.T, initialize=lambda model, t: PI[t-1])
    model.PI_cap = pyo.Param(model.T, initialize=lambda model, t: PI_cap[t-1])
    model.Q_begin = pyo.Param(model.P, initialize=lambda model, p: Q_begin[p-1])
    model.Q_end = pyo.Param(model.P, initialize=lambda model, p: Q_end[p-1])
    model.Q_len = pyo.Param(model.P, initialize=lambda model, p: Q_len[p-1])
    model.X_low = pyo.Param(model.P, initialize=lambda model, p: X_low[p-1])
    model.X_up = pyo.Param(model.P, initialize=lambda model, p: X_up[p-1])
    model.X_avg = pyo.Param(initialize=X_avg)
    model.Mi_low = pyo.Param(model.P, initialize=lambda model, p: Mi_low[p-1])
    model.Mi_up = pyo.Param(model.P, initialize=lambda model, p: Mi_up[p-1])
    model.Mi_avg = pyo.Param(initialize=Mi_avg)
    model.T_start = pyo.Param(model.I, initialize=lambda model, i: T_start[i-1])
    model.T_end = pyo.Param(model.I, initialize=lambda model, i: T_end[i-1])
    model.alpha = pyo.Param(model.N, initialize=lambda model, n: alpha[n-1])
    model.beta = pyo.Param(model.N, initialize=lambda model, n: beta[n-1])
    model.ch_eff = pyo.Param(initialize=ch_eff)
    model.dch_eff = pyo.Param(initialize=dch_eff)
    model.gama = pyo.Param(model.I, initialize=lambda model, i: gama[i-1], mutable=True)
    model.E_0 = pyo.Param(initialize=E_0)
    model.E_min = pyo.Param(initialize=E_min)
    model.E_max = pyo.Param(initialize=E_max)
    model.E_end = pyo.Param(initialize=E_end)
    model.C_bat = pyo.Param(model.K, initialize=lambda model, k: C_bat[k-1])
    model.U_pow = pyo.Param(model.L, initialize=lambda model, l: U_pow[l-1])
    model.U_price = pyo.Param(model.L, initialize=lambda model, l: U_price[l-1])
    model.U_max = pyo.Param(initialize=U_max)
    model.U_cap = pyo.Param(initialize=U_cap)
    model.delta = pyo.Param(initialize=delta)
    model.R = pyo.Param(initialize=R)
    model.Ah = pyo.Param(initialize=Ah)
    model.V = pyo.Param(initialize=V)

    model.y_buy = pyo.Param(model.T, initialize=y_buy)
    model.y_sell = pyo.Param(model.T, initialize=y_sell)
    model.y_cap = pyo.Param(model.T, initialize=y_cap)
    model.d_l = pyo.Param(model.K, model.T, initialize=d_l)
    model.u_l = pyo.Param(model.L, initialize=u_l)

    model.e = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.w_buy = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.w_sell = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.w_cap = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.d = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.a = pyo.Var(model.T, domain=pyo.Binary)
    model.b = pyo.Var(model.K, model.I, model.T, within=pyo.Binary)
    model.x = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.y = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.z = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.c = pyo.Var(model.K, model.T, domain=pyo.Binary)
    model.u = pyo.Var(model.L, domain=pyo.Binary)
    model.pho_plus = pyo.Var(model.P, domain=pyo.NonNegativeIntegers)
    model.pho_minus = pyo.Var(model.P, domain=pyo.NonNegativeIntegers)
    model.mi = pyo.Var(model.P, domain=pyo.NonNegativeIntegers)
    model.z_up = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.z_down = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)

    model.constraints = pyo.ConstraintList()

    for p in model.P:
        model.constraints.add(model.pho_plus[p] >= model.X_low[p])
    for p in model.P:
        model.constraints.add(model.pho_plus[p] <= model.X_up[p])
    for p in model.P:
        model.constraints.add(model.pho_minus[p] == model.pho_plus[p] * model.delta)
    for p in model.P:
        model.constraints.add(model.mi[p] >= model.Mi_low[p])
    for p in model.P:
        model.constraints.add(model.mi[p] <= model.Mi_up[p])
    model.constraints.add(((1/T) * sum(model.Q_len[p] * model.mi[p] for p in model.P)) <= model.Mi_avg)
    model.constraints.add(((1/T) * sum(model.Q_len[p] * model.mi[p] for p in model.P)) <= model.X_avg)

    for k in model.K:
        for t in model.T:
            model.constraints.add(sum(model.b[k, i, t] for i in model.I) + model.c[k, t] <= 1)
    for i in model.I:
        for t in range(model.T_start[i], model.T_end[i]):
            model.constraints.add(sum(model.b[k, i, t] for k in model.K) == 1)
    for i in model.I:
        for k in model.K:
            for t in range(model.T_start[i], model.T_end[i]-1):
                model.constraints.add(model.b[k, i, t+1] >= model.b[k, i, t])
    for n in model.N:
        for t in model.T:
            model.constraints.add(sum(model.x[k, n, t] for k in model.K) + sum(model.y[k, n, t] for k in model.K) + sum(model.z[k, n, t] for k in model.K) <= 1)
    for k in model.K:
        for t in model.T:
            model.constraints.add(sum(model.x[k, n, t] for n in model.N) + sum(model.y[k, n, t] for n in model.N) + sum(model.z[k, n, t] for n in model.N) <= model.c[k, t])
    for k in model.K:
        for t in range(2, T+1):
            model.constraints.add(model.e[k, t] == model.e[k, t-1] + sum(model.ch_eff*model.alpha[n]*model.x[k, n, t] for n in model.N) - sum(model.gama[i]*model.b[k, i, t] for i in model.I) - sum(model.dch_eff*model.beta[n]*model.y[k, n, t] for n in model.N))
    for t in model.T:
        model.constraints.add(sum(model.ch_eff*model.alpha[n]*model.x[k, n, t] for n in model.N for k in model.K) == model.w_buy[t])
    for t in model.T:
        model.constraints.add(sum(model.dch_eff*model.beta[n]*model.y[k, n, t] for n in model.N for k in model.K) == model.w_sell[t])
    model.constraints.add(sum(model.dch_eff*model.beta[n]*model.y[k, n, 1] for n in model.N for k in model.K) == 0)
    for k in model.K:
        for n in model.N:
            for t in range(2, T-d_on):
                model.constraints.add(1 - model.x[k, n, t] + model.x[k, n, t-1] + ((1/d_on)*sum(model.x[k, n, j] for j in range(t, t+d_on))) >= 1)
    for k in model.K:
        for n in model.N:
            for t in range(T-d_on+1, T):
                model.constraints.add(1 - model.x[k, n, t] + model.x[k, n, t-1] + ((1/(T-t+1))*sum(model.x[k, n, j] for j in range(t, T))) >= 1)
    for k in model.K:
        for n in model.N:
            for t in range(2, T-d_off):
                model.constraints.add(1 - model.x[k, n, t] + model.x[k, n, t-1] + ((1/d_off)*sum(model.x[k, n, j] for j in range(t, t+d_off))) <= 2)
    for k in model.K:
        for n in model.N:
            for t in range(T-d_off+1, T):
                model.constraints.add(1 - model.x[k, n, t] + model.x[k, n, t-1] + ((1/(T-t+1))*sum(model.x[k, n, j] for j in range(t, T))) <= 2)
    model.constraints.add(sum(model.u[l] for l in model.L) == 1)
    for t in model.T:
        model.constraints.add(sum(model.alpha[n]*model.x[k, n, t] for k in model.K for n in model.N) <= sum(model.U_pow[l]*model.u[l] for l in model.L))
    for t in model.T:
        model.constraints.add(sum(model.alpha[n]*model.x[k, n, t] for k in model.K for n in model.N) <= model.U_max)
    for k in model.K:
        for t in model.T:
            model.constraints.add(model.e[k, t] >= model.C_bat[k] * model.E_min)
    for k in model.K:
        for t in model.T:
            model.constraints.add(E_max * model.C_bat[k] >= model.e[k, t] + sum(model.ch_eff*model.alpha[n]*model.x[k, n, t] for n in model.N))
    for k in model.K:
        model.constraints.add(model.e[k, 1] == model.E_0*model.C_bat[k])
    for k in model.K:
        model.constraints.add(model.e[k, T-1] + sum(model.ch_eff*model.alpha[n]*model.x[k, n, T] for n in model.N) >= model.E_end*model.C_bat[k])
    for k in model.K:
        for t in model.T:
            model.constraints.add(model.d[k, t] == ((model.R*model.C_bat[1]*1000)/(4*model.Ah*model.V)) * (sum(model.dch_eff*model.beta[n]*model.y[k, n, t] for n in model.N)))
    for n in model.N:
        for t in model.T:
            model.constraints.add(sum(model.z_up[k, n, t] for k in model.K)+sum(model.z_down[k, n, t] for k in model.K) == sum(model.z[k, n, t] for k in model.K))
    for t in model.T:
        model.constraints.add(sum(model.ch_eff*model.alpha[n]*model.z_down[k, n, t] for k in model.K for n in model.N)*delta_t + sum(model.dch_eff*model.beta[n]*model.z_up[k, n, t] for k in model.K for n in model.N)*delta_t == model.w_cap[t])
    for k in model.K:
        for t in model.T:
            model.constraints.add(E_max * model.C_bat[k] >= model.e[k, t] + sum(model.ch_eff*model.alpha[n]*model.z_down[k, n, t] for n in model.N))
    for k in model.K:
        for t in model.T:
            model.constraints.add(E_min * model.C_bat[k] <= model.e[k, t] - sum(model.dch_eff*model.beta[n]*model.z_up[k, n, t] for k in model.K for n in model.N))
    for t in model.T:
        model.constraints.add(model.w_cap[t] >= model.U_cap - M*(1-model.a[t]))
    for t in model.T:
        model.constraints.add(model.w_cap[t] <= M*model.a[t])
    model.constraints.add(sum(model.z_up[k, n, t] for k in model.K for n in model.N for t in model.T) == sum(model.z_down[k, n, t] for k in model.K for n in model.N for t in model.T))
    for k in model.K:
        for n in model.N:
            for t in range(2, T-d_cap):
                model.constraints.add(1 - model.z[k, n, t] + model.z[k, n, t-1] + ((1/d_cap)*sum(model.z[k, n, j] for j in range(t, t+d_cap))) >= 1)
    for k in model.K:
        for n in model.N:
            for t in range(T-d_cap+1, T):
                model.constraints.add(1 - model.z[k, n, t] + model.z[k, n, t-1] + ((1/(T-t+1))*sum(model.z[k, n, j] for j in range(t, T))) >= 1)

    if count > 1:
        model.constraints.add(
            sum(model.pho_plus[p]*model.w_buy[t] for p in model.P for t in range(model.Q_begin[p], model.Q_end[p])) -
            sum(model.pho_minus[p]*model.w_sell[t] for p in model.P for t in range(model.Q_begin[p], model.Q_end[p])) -
            sum(model.mi[p]*model.w_cap[t] for p in model.P for t in range(model.Q_begin[p], model.Q_end[p])) +
            sum(model.d[k, t] for k in model.K for t in model.T) +
            sum(model.U_price[l]*model.u[l] for l in model.L)
            <=
            sum(model.pho_plus[p]*model.y_buy[t] for p in model.P for t in range(model.Q_begin[p], model.Q_end[p])) -
            sum(model.pho_minus[p]*model.y_sell[t] for p in model.P for t in range(model.Q_begin[p], model.Q_end[p])) -
            sum(model.mi[p]*model.y_cap[t] for p in model.P for t in range(model.Q_begin[p], model.Q_end[p])) +
            sum(model.d_l[k, t] for k in model.K for t in model.T) +
            sum(model.U_price[l]*model.u_l[l] for l in model.L)
        )

    def rule_obj(mod):
        return (sum(mod.pho_plus[p]*mod.w_buy[t] for p in mod.P for t in range(mod.Q_begin[p], mod.Q_end[p]))
              - sum(mod.pho_minus[p]*mod.w_sell[t] for p in mod.P for t in range(mod.Q_begin[p], mod.Q_end[p]))
              + sum(mod.PI_cap[t]*mod.w_cap[t] for p in mod.P for t in range(mod.Q_begin[p], mod.Q_end[p]))
              - sum(mod.mi[p]*mod.w_cap[t] for p in mod.P for t in range(mod.Q_begin[p], mod.Q_end[p]))
              + sum(mod.PI[t]*mod.w_sell[t] for t in mod.T)
              - sum(mod.PI[t]*mod.w_buy[t] for t in mod.T))

    model.obj = pyo.Objective(rule=rule_obj, sense=pyo.maximize)

    print('Solving HRP')
    opt = pyo.SolverFactory('appsi_highs')
    results = opt.solve(model)

    return model


def solveLL(data, pho_plus, pho_minus, mi):

    model = pyo.ConcreteModel()

    i = len(data['Trip time']['Time begin (min)'])
    t = 96
    k = len(data['Buses']['Bus (kWh)'])
    n = len(data['Chargers']['Charger (kWh/min)'])
    l = len(data['Power price']['Power'])
    p = len(data['Periods']['Period'])

    T_start = data['Trip time']['Time begin (min)'].tolist()
    T_start = [int(x) for x in T_start]
    T_end = data['Trip time']['Time finish (min)'].tolist()
    T_end = [int(x) for x in T_end]

    Q_begin = data['Periods']['Begin'].tolist()
    Q_begin = [int(x) for x in Q_begin]
    Q_end = data['Periods']['End'].tolist()
    Q_end = [int(x) for x in Q_end]

    alpha = data['Chargers']['Charger (kWh/min)'].tolist()
    beta = data['Chargers']['Charger (kWh/min)'].tolist()
    ch_eff = 0.90
    dch_eff = 1
    gama = data['Energy consumption']['Uncertain energy (kWh/km*min)'].tolist()

    E_0 = 0.2
    E_min = 0.2
    E_max = 1
    E_end = 0.2
    C_bat = data['Buses']['Bus (kWh)'].tolist()

    d_off = 1
    d_on = 1
    d_cap = 1

    U_pow = data['Power price']['Power'].tolist()
    U_price = data['Power price']['Price'].tolist()
    U_max = data['Chargers']['Max Power (kW)'].tolist()
    U_max = U_max[0]
    U_cap = 200

    R = 130
    Ah = 905452
    V = 512
    T = 96
    delta_t = 4
    M = 10000

    model.I = pyo.RangeSet(i)
    model.T = pyo.RangeSet(t)
    model.K = pyo.RangeSet(k)
    model.N = pyo.RangeSet(n)
    model.L = pyo.RangeSet(l)
    model.P = pyo.RangeSet(p)

    model.T_start = pyo.Param(model.I, initialize=lambda model, i: T_start[i-1])
    model.T_end = pyo.Param(model.I, initialize=lambda model, i: T_end[i-1])
    model.alpha = pyo.Param(model.N, initialize=lambda model, n: alpha[n-1])
    model.beta = pyo.Param(model.N, initialize=lambda model, n: beta[n-1])
    model.ch_eff = pyo.Param(initialize=ch_eff)
    model.dch_eff = pyo.Param(initialize=dch_eff)
    model.gama = pyo.Param(model.I, initialize=lambda model, i: gama[i-1])
    model.E_0 = pyo.Param(initialize=E_0)
    model.E_min = pyo.Param(initialize=E_min)
    model.E_max = pyo.Param(initialize=E_max)
    model.E_end = pyo.Param(initialize=E_end)
    model.C_bat = pyo.Param(model.K, initialize=lambda model, k: C_bat[k-1])
    model.U_pow = pyo.Param(model.L, initialize=lambda model, l: U_pow[l-1])
    model.U_price = pyo.Param(model.L, initialize=lambda model, l: U_price[l-1])
    model.U_max = pyo.Param(initialize=U_max)
    model.U_cap = pyo.Param(initialize=U_cap)
    model.R = pyo.Param(initialize=R)
    model.Ah = pyo.Param(initialize=Ah)
    model.V = pyo.Param(initialize=V)
    model.Q_begin = pyo.Param(model.P, initialize=lambda model, p: Q_begin[p-1])
    model.Q_end = pyo.Param(model.P, initialize=lambda model, p: Q_end[p-1])
    model.pho_plus = pyo.Param(model.P, initialize=pho_plus)
    model.pho_minus = pyo.Param(model.P, initialize=pho_minus)
    model.mi = pyo.Param(model.P, initialize=mi)

    model.b = pyo.Var(model.K, model.I, model.T, within=pyo.Binary)
    model.x = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.y = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.z = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.u = pyo.Var(model.L, domain=pyo.Binary)
    model.c = pyo.Var(model.K, model.T, domain=pyo.Binary)
    model.a = pyo.Var(model.T, domain=pyo.Binary)
    model.z_up = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.z_down = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.e = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.w_buy = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.w_sell = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.w_cap = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.d = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)

    def rule_obj(mod):
        return (sum(mod.pho_plus[p]*mod.w_buy[t] for p in mod.P for t in range(mod.Q_begin[p], mod.Q_end[p]))
              - sum(mod.pho_minus[p]*mod.w_sell[t] for p in mod.P for t in range(mod.Q_begin[p], mod.Q_end[p]))
              + sum(mod.d[k, t] for k in mod.K for t in mod.T)
              + sum(mod.U_price[l]*mod.u[l] for l in mod.L)
              - sum(mod.mi[p]*mod.w_cap[t] for p in mod.P for t in range(mod.Q_begin[p], mod.Q_end[p])))

    model.obj = pyo.Objective(rule=rule_obj, sense=pyo.minimize)

    model.constraints = pyo.ConstraintList()

    for k in model.K:
        for t in model.T:
            model.constraints.add(sum(model.b[k, i, t] for i in model.I) + model.c[k, t] <= 1)
    for i in model.I:
        for t in range(model.T_start[i], model.T_end[i]):
            model.constraints.add(sum(model.b[k, i, t] for k in model.K) == 1)
    for i in model.I:
        for k in model.K:
            for t in range(model.T_start[i], model.T_end[i]-1):
                model.constraints.add(model.b[k, i, t+1] >= model.b[k, i, t])
    for n in model.N:
        for t in model.T:
            model.constraints.add(sum(model.x[k, n, t] for k in model.K) + sum(model.y[k, n, t] for k in model.K) + sum(model.z[k, n, t] for k in model.K) <= 1)
    for k in model.K:
        for t in model.T:
            model.constraints.add(sum(model.x[k, n, t] for n in model.N) + sum(model.y[k, n, t] for n in model.N) + sum(model.z[k, n, t] for n in model.N) <= model.c[k, t])
    for k in model.K:
        for t in range(2, T+1):
            model.constraints.add(model.e[k, t] == model.e[k, t-1] + sum(model.ch_eff*model.alpha[n]*model.x[k, n, t] for n in model.N) - sum(model.gama[i]*model.b[k, i, t] for i in model.I) - sum(model.dch_eff*model.beta[n]*model.y[k, n, t] for n in model.N))
    for t in model.T:
        model.constraints.add(sum(model.ch_eff*model.alpha[n]*model.x[k, n, t] for n in model.N for k in model.K) == model.w_buy[t])
    for t in model.T:
        model.constraints.add(sum(model.dch_eff*model.beta[n]*model.y[k, n, t] for n in model.N for k in model.K) == model.w_sell[t])
    model.constraints.add(sum(model.dch_eff*model.beta[n]*model.y[k, n, 1] for n in model.N for k in model.K) == 0)
    for k in model.K:
        for n in model.N:
            for t in range(2, T-d_on):
                model.constraints.add(1 - model.x[k, n, t] + model.x[k, n, t-1] + ((1/d_on)*sum(model.x[k, n, j] for j in range(t, t+d_on))) >= 1)
    for k in model.K:
        for n in model.N:
            for t in range(T-d_on+1, T):
                model.constraints.add(1 - model.x[k, n, t] + model.x[k, n, t-1] + ((1/(T-t+1))*sum(model.x[k, n, j] for j in range(t, T))) >= 1)
    for k in model.K:
        for n in model.N:
            for t in range(2, T-d_off):
                model.constraints.add(1 - model.x[k, n, t] + model.x[k, n, t-1] + ((1/d_off)*sum(model.x[k, n, j] for j in range(t, t+d_off))) <= 2)
    for k in model.K:
        for n in model.N:
            for t in range(T-d_off+1, T):
                model.constraints.add(1 - model.x[k, n, t] + model.x[k, n, t-1] + ((1/(T-t+1))*sum(model.x[k, n, j] for j in range(t, T))) <= 2)
    model.constraints.add(sum(model.u[l] for l in model.L) == 1)
    for t in model.T:
        model.constraints.add(sum(model.alpha[n]*model.x[k, n, t] for k in model.K for n in model.N) <= sum(model.U_pow[l]*model.u[l] for l in model.L))
    for t in model.T:
        model.constraints.add(sum(model.alpha[n]*model.x[k, n, t] for k in model.K for n in model.N) <= model.U_max)
    for k in model.K:
        for t in model.T:
            model.constraints.add(model.e[k, t] >= model.C_bat[k] * model.E_min)
    for k in model.K:
        for t in model.T:
            model.constraints.add(E_max * model.C_bat[k] >= model.e[k, t] + sum(model.ch_eff*model.alpha[n]*model.x[k, n, t] for n in model.N))
    for k in model.K:
        model.constraints.add(model.e[k, 1] == model.E_0*model.C_bat[k])
    for k in model.K:
        model.constraints.add(model.e[k, T-1] + sum(model.ch_eff*model.alpha[n]*model.x[k, n, T] for n in model.N) >= model.E_end*model.C_bat[k])
    for k in model.K:
        for t in model.T:
            model.constraints.add(model.d[k, t] == ((model.R*model.C_bat[1]*1000)/(4*model.Ah*model.V)) * (sum(model.dch_eff*model.beta[n]*model.y[k, n, t] for n in model.N)))
    for n in model.N:
        for t in model.T:
            model.constraints.add(sum(model.z_up[k, n, t] for k in model.K)+sum(model.z_down[k, n, t] for k in model.K) == sum(model.z[k, n, t] for k in model.K))
    for t in model.T:
        model.constraints.add(sum(model.ch_eff*model.alpha[n]*model.z_down[k, n, t] for k in model.K for n in model.N)*delta_t + sum(model.dch_eff*model.beta[n]*model.z_up[k, n, t] for k in model.K for n in model.N)*delta_t == model.w_cap[t])
    for k in model.K:
        for t in model.T:
            model.constraints.add(E_max * model.C_bat[k] >= model.e[k, t] + sum(model.ch_eff*model.alpha[n]*model.z_down[k, n, t] for n in model.N))
    for k in model.K:
        for t in model.T:
            model.constraints.add(E_min * model.C_bat[k] <= model.e[k, t] - sum(model.dch_eff*model.beta[n]*model.z_up[k, n, t] for k in model.K for n in model.N))
    for t in model.T:
        model.constraints.add(model.w_cap[t] >= model.U_cap - M*(1-model.a[t]))
    for t in model.T:
        model.constraints.add(model.w_cap[t] <= M*model.a[t])
    model.constraints.add(sum(model.z_up[k, n, t] for k in model.K for n in model.N for t in model.T) == sum(model.z_down[k, n, t] for k in model.K for n in model.N for t in model.T))
    for k in model.K:
        for n in model.N:
            for t in range(2, T-d_cap):
                model.constraints.add(1 - model.z[k, n, t] + model.z[k, n, t-1] + ((1/d_cap)*sum(model.z[k, n, j] for j in range(t, t+d_cap))) >= 1)
    for k in model.K:
        for n in model.N:
            for t in range(T-d_cap+1, T):
                model.constraints.add(1 - model.z[k, n, t] + model.z[k, n, t-1] + ((1/(T-t+1))*sum(model.z[k, n, j] for j in range(t, T))) >= 1)

    print('Solving LL')
    opt = pyo.SolverFactory('appsi_highs')
    results = opt.solve(model, tee=True)

    return model


@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        input_data = request.json['input']
        data = build_dataframes(input_data)

        UB = float('inf')
        LB = float('-inf')
        k = 1
        y_buy_l = y_sell_l = y_cap_l = d_l = u_l = 0
        epsilon = 0.0001

        while UB != LB:
            model_HRP = solveHRP(data, y_buy_l, y_sell_l, y_cap_l, d_l, u_l, k)
            pho_plus  = model_HRP.pho_plus
            pho_minus = model_HRP.pho_minus
            mi        = model_HRP.mi
            y_buy_u   = model_HRP.w_buy
            y_sell_u  = model_HRP.w_sell
            y_cap_u   = model_HRP.w_cap
            UB        = model_HRP.obj()

            model_LL  = solveLL(data, pho_plus, pho_minus, mi)
            y_buy_l   = model_LL.w_buy
            y_sell_l  = model_LL.w_sell
            y_cap_l   = model_LL.w_cap
            d_l       = model_LL.d
            u_l       = model_LL.u

            f_y_u = (sum(model_LL.pho_plus[p]*y_buy_u[t] for p in model_LL.P for t in range(model_LL.Q_begin[p], model_LL.Q_end[p]))
                   - sum(model_LL.pho_minus[p]*y_sell_u[t] for p in model_LL.P for t in range(model_LL.Q_begin[p], model_LL.Q_end[p]))
                   + sum(model_LL.d[k, t] for k in model_LL.K for t in model_LL.T)
                   - sum(model_LL.mi[p]*y_cap_u[t] for p in model_LL.P for t in range(model_LL.Q_begin[p], model_LL.Q_end[p]))
                   + sum(model_LL.U_price[l]*model_LL.u[l] for l in model_LL.L))

            if pyo.value(f_y_u) - model_LL.obj() <= epsilon:
                LB = UB
            k += 1
            if k == 5:
                break

        result = {
            "status": "optimal",
            "upper_bound": UB,
            "lower_bound": LB,
            "w_buy":  [pyo.value(model_LL.w_buy[t])  for t in model_LL.T],
            "w_sell": [pyo.value(model_LL.w_sell[t]) for t in model_LL.T],
            "w_cap":  [pyo.value(model_LL.w_cap[t])  for t in model_LL.T],
            "energy": [[pyo.value(model_LL.e[k, t]) for t in model_LL.T] for k in model_LL.K],
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
