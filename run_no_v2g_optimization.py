from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyomo.environ as pyo

from app import build_dataframes, extract_scalars, solvePTO
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
    total_pto_sell_rev = sum(sc['S_sell'][t - 1] * pyo.value(model.w_sell[t]) for t in range(1, t_steps + 1))
    pto_daily_cost = total_pto_buy_cost - total_pto_sell_rev
    total_kwh_bought = sum(pyo.value(model.w_buy[t]) for t in range(1, t_steps + 1))
    total_kwh_sold = sum(pyo.value(model.w_sell[t]) for t in range(1, t_steps + 1))
    return {
        'scenario': 'optimization_no_v2g',
        'optimization_mode': sc['optimization_mode'],
        'timestep_minutes': sc['timestep_minutes'],
        'v2g_enabled': sc['v2g_enabled'],
        'avg_grid_price': sc['avg_P'],
        'avg_buy_price': sc['avg_S_buy'],
        'avg_sell_price': sc['avg_S_sell'],
        'pto_daily_cost': pto_daily_cost,
        'total_buy_cost': total_pto_buy_cost,
        'total_sell_revenue': total_pto_sell_rev,
        'total_kwh_bought': total_kwh_bought,
        'total_kwh_sold': total_kwh_sold,
        'w_buy': [float(pyo.value(model.w_buy[t])) for t in range(1, t_steps + 1)],
        'w_sell': [float(pyo.value(model.w_sell[t])) for t in range(1, t_steps + 1)],
        'energy': [[float(pyo.value(model.e[k, t])) for t in range(1, t_steps + 1)] for k in range(1, sc['k_count'] + 1)],
        'power_kw': compute_bus_power_kw([[float(pyo.value(model.e[k, t])) for t in range(1, t_steps + 1)] for k in range(1, sc['k_count'] + 1)], sc['timestep_minutes']),
        'S_buy': sc['S_buy'],
        'S_sell': sc['S_sell'],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run charging-only optimization with direct grid tariffs and no V2G.')
    parser.add_argument('--input', default='Files/Inputs.xlsx', help='Path to the input workbook.')
    parser.add_argument('--spot-prices-file', default='', help='Optional workbook containing spot prices.')
    parser.add_argument('--output', default='Files/no_v2g_optimization_result.json', help='Path to the JSON result file.')
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
    model = solvePTO(sc)
    if model is None:
        raise RuntimeError('Charging-only no-V2G optimization was infeasible.')

    result = summarize_solution(sc, model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + '\n')

    summary_row = build_day_ahead_summary_row(
        sc,
        model,
        'optimization_no_v2g',
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

    print(f"Saved no-V2G optimization result to {output_path}")
    print(f"Appended day_ahead_summary row to {summary_workbook}")
    print(f"Appended agent_reasoning row to {summary_workbook}")
    print(f"Total kWh bought: {result['total_kwh_bought']:.4f}")
    print(f"PTO daily cost: {result['pto_daily_cost']:.4f}")


if __name__ == '__main__':
    main()
