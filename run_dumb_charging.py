from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyomo.environ as pyo

from app import build_dataframes, extract_scalars
from generate_benchmark_files import build_input_data
from scenario_summary import (
    append_agent_reasoning,
    append_day_ahead_summary,
    build_agent_reasoning_row,
    build_day_ahead_summary_row,
)


def attach_grid_tariffs(input_data: dict) -> dict:
    price_rows = input_data.get('grid_prices') or input_data.get('prices') or []
    tariffs = []
    for row in price_rows:
        if row is None:
            continue
        spot = row.get('spot_market', row.get('Spot Market', row.get('price', row.get('Price'))))
        time_value = row.get('time', row.get('Time', row.get('timestep')))
        if spot in (None, ''):
            continue
        tariffs.append({
            'time': time_value,
            'buy_tariff': spot,
            'sell_tariff': spot,
        })
    input_data['tariffs'] = tariffs
    input_data['v2g_enabled'] = False
    return input_data


def solve_dumb_charging(sc: dict):
    t_steps = sc['T_steps']
    p = sc['P']
    s_buy = sc['S_buy']
    s_sell = sc['S_sell']
    alpha = sc['alpha']
    beta = sc['beta']
    gama = sc['gama']
    c_bat = sc['C_bat']
    ch_eff = sc['ch_eff']
    dch_eff = sc['dch_eff']
    e_0 = sc['E_0']
    e_min = sc['E_min']
    e_max = sc['E_max']
    e_end = sc['E_end']
    u_max = sc['U_max']
    t_start = sc['T_start']
    t_end_ = sc['T_end']

    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(sc['i_count'])
    model.T = pyo.RangeSet(t_steps)
    model.K = pyo.RangeSet(sc['k_count'])
    model.N = pyo.RangeSet(sc['n_count'])

    model.T_start = pyo.Param(model.I, initialize=lambda m, i: t_start[i - 1])
    model.T_end = pyo.Param(model.I, initialize=lambda m, i: t_end_[i - 1])
    model.alpha = pyo.Param(model.N, initialize=lambda m, n: alpha[n - 1])
    model.beta = pyo.Param(model.N, initialize=lambda m, n: beta[n - 1])
    model.gama = pyo.Param(model.I, initialize=lambda m, i: gama[i - 1])
    model.P = pyo.Param(model.T, initialize=lambda m, t: p[t - 1])
    model.S_buy = pyo.Param(model.T, initialize=lambda m, t: s_buy[t - 1])
    model.S_sell = pyo.Param(model.T, initialize=lambda m, t: s_sell[t - 1])
    model.C_bat = pyo.Param(model.K, initialize=lambda m, k: c_bat[k - 1])

    model.b = pyo.Var(model.K, model.I, model.T, domain=pyo.Binary)
    model.x = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.y = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.c = pyo.Var(model.K, model.T, domain=pyo.Binary)
    model.e = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.w_buy = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.w_sell = pyo.Var(model.T, within=pyo.NonNegativeReals)

    for k in model.K:
        for i in model.I:
            if i != k:
                for t in model.T:
                    model.b[k, i, t].fix(0)

    model.constraints = pyo.ConstraintList()

    for k in model.K:
        for t in model.T:
            model.constraints.add(sum(model.b[k, i, t] for i in model.I) + model.c[k, t] <= 1)

    for i in model.I:
        for t in range(model.T_start[i], model.T_end[i]):
            model.constraints.add(sum(model.b[k, i, t] for k in model.K) == 1)

    for i in model.I:
        for k in model.K:
            for t in range(model.T_start[i], model.T_end[i] - 1):
                model.constraints.add(model.b[k, i, t + 1] >= model.b[k, i, t])

    for n in model.N:
        for t in model.T:
            model.constraints.add(sum(model.x[k, n, t] for k in model.K) + sum(model.y[k, n, t] for k in model.K) <= 1)

    for k in model.K:
        for t in model.T:
            model.constraints.add(sum(model.x[k, n, t] for n in model.N) + sum(model.y[k, n, t] for n in model.N) <= model.c[k, t])

    for k in model.K:
        model.constraints.add(model.e[k, 1] == e_0[k - 1] * c_bat[k - 1])
        for t in range(2, t_steps + 1):
            trip_energy_t = sum(model.gama[i] * model.b[k, i, t] for i in model.I)
            charge_energy_t = sum(ch_eff * model.alpha[n] * model.x[k, n, t] for n in model.N)
            discharge_energy_t = sum(dch_eff * model.beta[n] * model.y[k, n, t] for n in model.N)
            model.constraints.add(model.e[k, t] == model.e[k, t - 1] - trip_energy_t + charge_energy_t - discharge_energy_t)

    for t in model.T:
        model.constraints.add(sum(ch_eff * alpha[n - 1] * model.x[k, n, t] for n in model.N for k in model.K) == model.w_buy[t])
        model.constraints.add(sum(dch_eff * beta[n - 1] * model.y[k, n, t] for n in model.N for k in model.K) == model.w_sell[t])

    model.constraints.add(sum(dch_eff * beta[n - 1] * model.y[k, n, 1] for n in model.N for k in model.K) == 0)

    for t in model.T:
        model.constraints.add(model.w_sell[t] == 0)
        for k in model.K:
            for n in model.N:
                model.constraints.add(model.y[k, n, t] == 0)

    for t in model.T:
        model.constraints.add(sum(alpha[n - 1] * model.x[k, n, t] for k in model.K for n in model.N) <= u_max)

    for k in model.K:
        for t in model.T:
            model.constraints.add(model.e[k, t] >= c_bat[k - 1] * e_min)
            model.constraints.add(model.e[k, t] <= e_max * c_bat[k - 1])

    for k in model.K:
        model.constraints.add(model.e[k, t_steps - 1] + sum(ch_eff * alpha[n - 1] * model.x[k, n, t_steps] for n in model.N) >= e_end * c_bat[k - 1])

    first_departure_t = max(1, min(t_steps, min(t_start)))
    pre_service_t = max(1, first_departure_t - 1)

    departure_bonus_terms = []
    for i in model.I:
        departure_t = max(1, min(t_steps, t_start[i - 1]))
        departure_bonus_terms.append(model.e[i, departure_t])

    def rule_obj(mod):
        # Benchmark intent:
        # 1. Fill buses as much as possible before the first trips start.
        # 2. Avoid repeated daytime charging unless it is required for later trips.
        pre_day_energy = sum(mod.e[k, pre_service_t] for k in mod.K)
        post_start_charge = sum(mod.w_buy[t] for t in range(first_departure_t, t_steps + 1))
        post_start_sessions = sum(mod.x[k, n, t] for k in mod.K for n in mod.N for t in range(first_departure_t, t_steps + 1))
        total_charge = sum(mod.w_buy[t] for t in mod.T)
        departure_bonus = sum(departure_bonus_terms)
        return (
            1000.0 * pre_day_energy
            + 1.0 * departure_bonus
            - 10.0 * post_start_charge
            - 25.0 * post_start_sessions
            - 0.1 * total_charge
        )

    model.obj = pyo.Objective(rule=rule_obj, sense=pyo.maximize)

    print('Solving dumb charging model')
    opt = pyo.SolverFactory('gurobi')
    opt.options['TimeLimit'] = 60
    opt.options['MIPGap'] = 0.04
    results = opt.solve(model, load_solutions=False, tee=True)

    tc = results.solver.termination_condition
    print(f'Dumb charging termination: {tc}')
    if tc in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible):
        model.solutions.load_from(results)
        print('Dumb charging done')
        return model
    print(f'Dumb charging infeasible: {tc}')
    return None


def compute_bus_power_kw(energy, timestep_minutes: float):
    timestep_hours = float(timestep_minutes) / 60.0
    if timestep_hours <= 0:
        return [[0.0 for _ in bus_energy] for bus_energy in energy]
    power = []
    for bus_energy in energy:
        bus_power = []
        for idx, value in enumerate(bus_energy):
            if idx == 0:
                bus_power.append(0.0)
            else:
                bus_power.append((float(value) - float(bus_energy[idx - 1])) / timestep_hours)
        power.append(bus_power)
    return power


def summarize_solution(sc: dict, model) -> dict:
    t_steps = sc['T_steps']
    total_pto_buy_cost = sum(sc['S_buy'][t - 1] * pyo.value(model.w_buy[t]) for t in range(1, t_steps + 1))
    total_kwh_bought = sum(pyo.value(model.w_buy[t]) for t in range(1, t_steps + 1))
    return {
        'scenario': 'dumb_charging_no_v2g',
        'optimization_mode': sc['optimization_mode'],
        'timestep_minutes': sc['timestep_minutes'],
        'v2g_enabled': sc['v2g_enabled'],
        'avg_grid_price': sc['avg_P'],
        'avg_buy_price': sc['avg_S_buy'],
        'pto_daily_cost': total_pto_buy_cost,
        'total_buy_cost': total_pto_buy_cost,
        'total_kwh_bought': total_kwh_bought,
        'total_kwh_sold': 0.0,
        'w_buy': [float(pyo.value(model.w_buy[t])) for t in range(1, t_steps + 1)],
        'w_sell': [0.0 for _ in range(t_steps)],
        'energy': [[float(pyo.value(model.e[k, t])) for t in range(1, t_steps + 1)] for k in range(1, sc['k_count'] + 1)],
        'power_kw': compute_bus_power_kw([[float(pyo.value(model.e[k, t])) for t in range(1, t_steps + 1)] for k in range(1, sc['k_count'] + 1)], sc['timestep_minutes']),
        'S_buy': sc['S_buy'],
        'S_sell': sc['S_sell'],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run dumb charging with direct grid tariffs and no V2G.')
    parser.add_argument('--input', default='Files/Inputs.xlsx', help='Path to the input workbook.')
    parser.add_argument('--spot-prices-file', default='', help='Optional workbook containing spot prices.')
    parser.add_argument('--output', default='Files/dumb_charging_result.json', help='Path to the JSON result file.')
    parser.add_argument('--summary-workbook', default='', help='Workbook where day_ahead_summary rows should be appended. Defaults to --input.')
    parser.add_argument('--reasoning-source', default='', help='Optional source label for the agent reasoning row.')
    parser.add_argument('--reasoning-text', default='', help='Optional explicit reasoning text to store in the agent_reasoning sheet.')
    args = parser.parse_args()

    input_path = Path(args.input)
    spot_prices_path = Path(args.spot_prices_file) if args.spot_prices_file else None
    summary_workbook = Path(args.summary_workbook) if args.summary_workbook else input_path

    input_data, _ = build_input_data(input_path, spot_prices_path=spot_prices_path, tariffs_path=None)
    input_data = attach_grid_tariffs(input_data)

    data = build_dataframes(input_data)
    sc = extract_scalars(data, price_guidance={}, optimization_mode='day_ahead', current_timestep=1)
    model = solve_dumb_charging(sc)
    if model is None:
        raise RuntimeError('Dumb charging scenario was infeasible.')

    result = summarize_solution(sc, model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + '\n')

    summary_row = build_day_ahead_summary_row(
        sc,
        model,
        'dumb_charging_no_v2g',
        input_workbook=input_path,
        spot_prices_file=spot_prices_path,
        tariffs_file=None,
    )
    append_day_ahead_summary(summary_workbook, summary_row)
    reasoning_row = build_agent_reasoning_row(
        sc,
        model,
        summary_row['scenario'],
        input_workbook=input_path,
        spot_prices_file=spot_prices_path,
        tariffs_file=None,
        reasoning_source=args.reasoning_source or None,
        reasoning_text=args.reasoning_text or None,
    )
    append_agent_reasoning(summary_workbook, reasoning_row)

    print(f"Saved dumb charging result to {output_path}")
    print(f"Appended day_ahead_summary row to {summary_workbook}")
    print(f"Appended agent_reasoning row to {summary_workbook}")
    print(f"Total kWh bought: {result['total_kwh_bought']:.4f}")
    print(f"Charging cost at grid tariff: {result['pto_daily_cost']:.4f}")


if __name__ == '__main__':
    main()
