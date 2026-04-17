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
    "optimization_mode":     "day_ahead",
    "current_timestep":      1,
    "optimized_steps":       0,
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
    trip_source = pd.DataFrame(input_data['trip_time'])
    energy_source = pd.DataFrame(
        input_data.get('energy_consumption', input_data['trip_time']))

    buses = pd.DataFrame(input_data['buses']).rename(
        columns={
            'bus_kwh': 'Bus (kWh)',
            'initial_soc': 'Initial SOC',
            'initial_energy_kwh': 'Initial energy (kWh)',
        })
    chargers = pd.DataFrame(input_data['chargers']).rename(columns={
        'charger_kwhmin': 'Charger (kWh/min)',
        'charger_kw':     'Charger (kW)'})
    trip_time = trip_source.rename(columns={
        'time_begin_min':  'Time begin (min)',
        'time_finish_min': 'Time finish (min)',
        'time_begin':      'Time begin',
        'time_end':        'Time finish',
        'time_finish':     'Time finish',
    })
    energy_consumption = energy_source.rename(
        columns={
            'uncertain_energy_kwhkmmin': 'Uncertain energy (kWh/km*min)',
            'uncertain_energy_kwhkm': 'Uncertain energy (kWh/km)',
            'energy_kwhkm': 'Energy (kWh/km)',
            'average_velocity_kmh': 'Average velocity (km/h)',
            'average_velocity_kmmin': 'Average velocity (km/min)',
        })
    prices = pd.DataFrame(input_data['prices']).rename(
        columns={'spot_market': 'Spot Market', 'time': 'Time'})
    realtime_state = pd.DataFrame(input_data.get('realtime_state', [])).rename(
        columns={
            'bus_id':             'Bus ID',
            'current_timestep':   'Current timestep',
            'current_soc':        'Current SOC',
            'current_energy_kwh': 'Current energy (kWh)',
            'operation_status':   'Operation status',
            'delay_minutes':      'Delay minutes',
        })
    return {
        'Buses':              buses,
        'Chargers':           chargers,
        'Trip time':          trip_time,
        'Energy consumption': energy_consumption,
        'Prices':             prices,
        'Realtime state':     realtime_state,
        'timestep_minutes':   input_data.get('timestep_minutes'),
    }


# ==============================================================================
# PARAMETER EXTRACTION
# ==============================================================================
def normalize_soc(value):
    soc = float(value)
    if soc > 1.0:
        soc = soc / 100.0
    return max(0.0, min(1.0, soc))


def parse_time_to_step(value, timestep_minutes, full_horizon_steps, is_end=False):
    if pd.isna(value) or value == '':
        raise ValueError('Trip time values cannot be empty')

    text = str(value).strip()
    if ':' in text:
        hour_text, minute_text = text.split(':', 1)
        total_minutes = (int(hour_text) * 60) + int(minute_text)
        step_value = total_minutes / float(timestep_minutes)
        if is_end:
            step = int(step_value) + 1 if step_value.is_integer() else int(step_value) + 2
        else:
            step = int(step_value) + 1
        return max(1, min(full_horizon_steps, step))

    numeric = float(text)
    if numeric <= full_horizon_steps and float(numeric).is_integer():
        return max(1, min(full_horizon_steps, int(numeric)))

    if 0.0 <= numeric <= 24.0:
        total_minutes = numeric * 60.0
        step_value = total_minutes / float(timestep_minutes)
        if is_end:
            step = int(step_value) + 1 if step_value.is_integer() else int(step_value) + 2
        else:
            step = int(step_value) + 1
        return max(1, min(full_horizon_steps, step))

    return max(1, min(full_horizon_steps, int(round(numeric))))


def get_timestep_minutes(input_data, prices):
    if 'timestep_minutes' in input_data and input_data['timestep_minutes'] not in (None, ''):
        return float(input_data['timestep_minutes'])

    if len(prices) > 0:
        inferred = 1440.0 / float(len(prices))
        if inferred > 0:
            return inferred

    if 'Time' in prices.columns:
        time_values = pd.to_numeric(prices['Time'], errors='coerce').dropna().tolist()
        if len(time_values) >= 2:
            inferred = (time_values[1] - time_values[0]) * 60.0
            if inferred > 0:
                return float(inferred)

    return 15.0


def get_speed_km_per_hour_series(energy_consumption, count):
    speed_columns = [
        ('Average velocity (km/h)', 1.0),
        ('avg_speed_kmph', 1.0),
        ('average_speed_kmph', 1.0),
        ('Average velocity (km/min)', 60.0),
        ('avg_speed_kmmin', 60.0),
        ('average_speed_kmmin', 60.0),
    ]
    for column, multiplier in speed_columns:
        if column in energy_consumption.columns:
            values = pd.to_numeric(
                energy_consumption[column], errors='coerce').tolist()[:count]
            return [
                (float(value) * multiplier) if pd.notna(value) else 12.0
                for value in values
            ]
    return [12.0] * count


def get_duration_steps(start_value, end_value, timestep_minutes):
    if pd.isna(start_value) or pd.isna(end_value):
        raise ValueError('Trip time values cannot be empty')

    start_text = str(start_value).strip()
    end_text = str(end_value).strip()

    if ':' in start_text and ':' in end_text:
        start_hour, start_minute = start_text.split(':', 1)
        end_hour, end_minute = end_text.split(':', 1)
        start_total = (int(start_hour) * 60) + int(start_minute)
        end_total = (int(end_hour) * 60) + int(end_minute)
        if end_total < start_total:
            raise ValueError('Trip end time cannot be earlier than start time')
        duration_steps = (end_total - start_total) / float(timestep_minutes)
        if not duration_steps.is_integer():
            raise ValueError(
                f'Trip duration {start_text} -> {end_text} is not aligned '
                f'to the timestep of {timestep_minutes} minutes')
        return int(duration_steps)

    start_numeric = float(start_value)
    end_numeric = float(end_value)
    duration_steps = end_numeric - start_numeric
    if duration_steps < 0:
        raise ValueError('Trip end time cannot be earlier than start time')
    if not float(duration_steps).is_integer():
        raise ValueError('Trip duration must align with the model timestep')
    return int(duration_steps)


def validate_timestep_consistency(data, timestep_minutes, alpha, charger_kw,
                                  energy_per_step, start_column, end_column,
                                  T_start_raw, T_end_raw):
    expected_alpha = [float(value) * (timestep_minutes / 60.0) for value in charger_kw]
    for idx, (actual_alpha, expected) in enumerate(zip(alpha, expected_alpha), start=1):
        if abs(actual_alpha - expected) > 1e-9:
            raise ValueError(
                f'Charger {idx} energy-per-step mismatch: expected {expected}, '
                f'got {actual_alpha}')

    trip_time = data['Trip time']
    raw_starts = trip_time[start_column].tolist()[:len(T_start_raw)]
    raw_ends = trip_time[end_column].tolist()[:len(T_end_raw)]
    for idx, (start_value, end_value, modeled_start, modeled_end, step_energy) in enumerate(
        zip(raw_starts, raw_ends, T_start_raw, T_end_raw, energy_per_step), start=1
    ):
        expected_steps = get_duration_steps(start_value, end_value, timestep_minutes)
        modeled_steps = modeled_end - modeled_start
        if expected_steps != modeled_steps:
            raise ValueError(
                f'Trip {idx} duration mismatch: input gives {expected_steps} steps, '
                f'model maps to {modeled_steps} steps')
        if step_energy <= 0:
            raise ValueError(f'Trip {idx} must have positive energy consumption per step')


def extract_realtime_state(data, current_timestep):
    realtime = data['Realtime state'].copy()
    if realtime.empty:
        raise ValueError('real_time mode requires a non-empty realtime_state input')

    realtime['Bus ID'] = pd.to_numeric(realtime['Bus ID'], errors='coerce').fillna(0).astype(int)
    realtime = realtime[realtime['Bus ID'] > 0].drop_duplicates(subset=['Bus ID'], keep='last')

    if 'Current timestep' in realtime.columns and realtime['Current timestep'].notna().any():
        current_timestep = int(pd.to_numeric(
            realtime['Current timestep'], errors='coerce').dropna().iloc[0])

    return realtime, current_timestep


def extract_scalars(data, price_guidance=None, optimization_mode='day_ahead',
                    current_timestep=1):
    price_guidance = price_guidance or {}
    optimization_mode = (optimization_mode or 'day_ahead').lower()

    # ── Fleet size ──────────────────────────────────────────────────────────
    k_count = len(data['Buses'])
    n_count = len(data['Chargers'])
    i_count = len(data['Trip time'])

    if not (k_count and n_count and i_count):
        raise ValueError('buses, chargers, and trip_time inputs must all be non-empty')

    # ── Grid buying price P — from data, not LLM-controlled ────────────────
    P_raw = pd.to_numeric(data['Prices']['Spot Market'], errors='coerce').dropna().tolist()
    if not P_raw:
        raise ValueError('prices input must include at least one valid Spot Market value')

    full_horizon_steps = len(P_raw)
    current_timestep = max(1, min(int(current_timestep or 1), full_horizon_steps))
    timestep_minutes = get_timestep_minutes(data, data['Prices'])
    timestep_hours = timestep_minutes / 60.0

    realtime = data['Realtime state']
    if optimization_mode == 'real_time':
        realtime, current_timestep = extract_realtime_state(data, current_timestep)
        current_timestep = max(1, min(int(current_timestep or 1), full_horizon_steps))

    P = P_raw[current_timestep - 1:] if optimization_mode == 'real_time' else P_raw
    T_steps = len(P)
    if T_steps <= 0:
        raise ValueError('No optimization horizon remains after applying current_timestep')

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
    C_bat = pd.to_numeric(data['Buses']['Bus (kWh)'], errors='coerce').tolist()[:k_count]

    charger_kw = pd.to_numeric(
        data['Chargers']['Charger (kW)'], errors='coerce').tolist()[:n_count]
    alpha = [float(value) * timestep_hours for value in charger_kw]
    beta = list(alpha)

    energy_columns = [
        'Energy (kWh/km)',
        'Uncertain energy (kWh/km)',
        'uncertain_energy_kwhkm',
        'energy_kwhkm',
        'Uncertain energy (kWh/km*min)',
    ]
    energy_series = None
    for column in energy_columns:
        if column in data['Energy consumption'].columns:
            energy_series = pd.to_numeric(
                data['Energy consumption'][column], errors='coerce')
            if energy_series.notna().any():
                break
    if energy_series is None or not energy_series.notna().any():
        raise ValueError('energy_consumption input must include a valid kWh/km column')

    avg_speed_km_hour = get_speed_km_per_hour_series(
        data['Energy consumption'], i_count)
    gama = [
        float(energy_value) * float(speed_value) * timestep_hours
        for energy_value, speed_value in zip(
            energy_series.tolist()[:i_count], avg_speed_km_hour)
    ]

    U_max_kw = sum(charger_kw)  # site cap = sum of all charger ratings
    U_max = U_max_kw * timestep_hours

    # ── Trip times ──────────────────────────────────────────────────────────
    start_column = (
        'Time begin' if 'Time begin' in data['Trip time'].columns else 'Time begin (min)'
    )
    end_column = (
        'Time finish' if 'Time finish' in data['Trip time'].columns else 'Time finish (min)'
    )
    T_start_raw = [
        parse_time_to_step(value, timestep_minutes, full_horizon_steps, is_end=False)
        for value in data['Trip time'][start_column].tolist()[:i_count]
    ]
    T_end_raw = [
        parse_time_to_step(value, timestep_minutes, full_horizon_steps, is_end=True)
        for value in data['Trip time'][end_column].tolist()[:i_count]
    ]

    validate_timestep_consistency(
        data,
        timestep_minutes,
        alpha,
        charger_kw,
        gama,
        start_column,
        end_column,
        T_start_raw,
        T_end_raw,
    )

    if optimization_mode == 'real_time':
        delay_steps = [0] * i_count
        in_trip_flags = [False] * i_count
        if not realtime.empty and 'Delay minutes' in realtime.columns:
            for _, row in realtime.iterrows():
                bus_idx = int(row['Bus ID']) - 1
                if 0 <= bus_idx < i_count:
                    delay_minutes = pd.to_numeric(
                        pd.Series([row.get('Delay minutes', 0)]),
                        errors='coerce').fillna(0).iloc[0]
                    delay_steps[bus_idx] = int(round(float(delay_minutes) / timestep_minutes))
                    operation_status = str(row.get('Operation status', '')).strip().lower()
                    in_trip_flags[bus_idx] = operation_status in {
                        'in_trip', 'operating', 'on_trip', 'running'
                    }

        adjusted_starts = [
            current_timestep if in_trip_flags[i] else T_start_raw[i] + delay_steps[i]
            for i in range(i_count)
        ]
        adjusted_ends = [T_end_raw[i] + delay_steps[i] for i in range(i_count)]
        T_start = [s - current_timestep + 1 for s in adjusted_starts]
        T_end = [e - current_timestep + 1 for e in adjusted_ends]
    else:
        T_start = [max(1, min(T_steps - 1, t)) for t in T_start_raw]
        T_end = [max(T_start[i] + 1, min(T_steps, t))
                 for i, t in enumerate(T_end_raw)]

    if optimization_mode == 'real_time':
        active_trip_rows = []
        active_energy = []
        active_starts = []
        active_ends = []
        for i in range(i_count):
            if T_end[i] >= 1:
                active_trip_rows.append(i)
                active_energy.append(gama[i])
                active_starts.append(max(1, min(T_steps, T_start[i])))
                active_ends.append(max(1, min(T_steps, T_end[i])))

        if not active_trip_rows:
            raise ValueError('No remaining trips to optimize for the requested real-time horizon')

        gama = active_energy
        T_start = active_starts
        T_end = active_ends
        i_count = len(active_trip_rows)

    default_initial_soc = [0.2] * k_count
    if optimization_mode == 'day_ahead' and (
        'Initial SOC' in data['Buses'].columns
        or 'Initial energy (kWh)' in data['Buses'].columns
    ):
        E_0 = default_initial_soc.copy()
        for bus_idx in range(k_count):
            initial_energy = (
                data['Buses']['Initial energy (kWh)'].iloc[bus_idx]
                if 'Initial energy (kWh)' in data['Buses'].columns else None
            )
            initial_soc = (
                data['Buses']['Initial SOC'].iloc[bus_idx]
                if 'Initial SOC' in data['Buses'].columns else None
            )
            if pd.notna(initial_energy):
                E_0[bus_idx] = max(0.0, min(1.0, float(initial_energy) / C_bat[bus_idx]))
            elif pd.notna(initial_soc):
                E_0[bus_idx] = normalize_soc(initial_soc)
    elif optimization_mode == 'real_time':
        E_0 = default_initial_soc.copy()
        for _, row in realtime.iterrows():
            bus_idx = int(row['Bus ID']) - 1
            if not 0 <= bus_idx < k_count:
                continue
            current_energy = row.get('Current energy (kWh)')
            current_soc = row.get('Current SOC')
            if pd.notna(current_energy):
                E_0[bus_idx] = max(0.0, min(1.0, float(current_energy) / C_bat[bus_idx]))
            elif pd.notna(current_soc):
                E_0[bus_idx] = normalize_soc(current_soc)
    else:
        E_0 = default_initial_soc

    # ── Sanity check ────────────────────────────────────────────────────────
    charge_per_step = 0.90 * alpha[0]
    drain_per_step  = gama[0]
    usable_kWh      = (1.0 - 0.2) * C_bat[0]
    print(f'mode={optimization_mode}, current_timestep={current_timestep}')
    print(f'timestep_minutes={timestep_minutes}')
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
        'full_horizon_steps': full_horizon_steps,
        'current_timestep': current_timestep,
        'timestep_minutes': timestep_minutes,
        'optimization_mode': optimization_mode,
        'k_count': k_count, 'n_count': n_count,
        'i_count': i_count,
        'P': P, 'S_buy': S_buy, 'S_sell': S_sell,
        'avg_P': avg_P, 'avg_S_buy': avg_S_buy, 'avg_S_sell': avg_S_sell,
        'buy_multipliers': buy_multipliers,
        'sell_multipliers': sell_multipliers,
        'boundaries': boundaries,
        'C_bat': C_bat, 'alpha': alpha, 'beta': beta, 'gama': gama,
        'U_max': U_max,
        'T_start': T_start, 'T_end': T_end,
        'ch_eff': 0.90, 'dch_eff': 1.0 / 0.90,
        'E_0': E_0, 'E_min': 0.2, 'E_max': 1.0, 'E_end': 0.2,
    }



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
            delay_steps = int(round(delay / sc['timestep_minutes']))
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
# Objective: min DOC = S_buy*w_buy - S_sell*w_sell
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
    ch_eff  = sc['ch_eff']
    dch_eff = sc['dch_eff']
    E_0     = sc['E_0']
    E_min   = sc['E_min']
    E_max   = sc['E_max']
    E_end   = sc['E_end']
    U_max   = sc['U_max']
    T_start = sc['T_start']
    T_end_  = sc['T_end']

    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(sc['i_count'])
    model.T = pyo.RangeSet(T)
    model.K = pyo.RangeSet(sc['k_count'])
    model.N = pyo.RangeSet(sc['n_count'])
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

    # Variables
    model.b      = pyo.Var(model.K, model.I, model.T, domain=pyo.Binary)
    model.x      = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.y      = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.c      = pyo.Var(model.K, model.T, domain=pyo.Binary)
    model.e      = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.w_buy  = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.w_sell = pyo.Var(model.T, within=pyo.NonNegativeReals)

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
        model.constraints.add(model.e[k,1] == E_0[k-1] * C_bat[k-1])
        for t in range(2, T+1):
            model.constraints.add(
                model.e[k,t] == model.e[k,t-1]
                + sum(ch_eff  * alpha[n-1] * model.x[k,n,t] for n in model.N)
                - sum(gama[i-1] * model.b[k,i,t] for i in model.I)
                - sum(dch_eff * beta[n-1]  * model.y[k,n,t] for n in model.N))

    # ── Constraints 8.1-8.2: w_buy and w_sell definitions ────────────────
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

    # ── Constraint 9: site charging power limit ──────────────────────────
    for t in model.T:
        model.constraints.add(
            sum(alpha[n-1]*model.x[k,n,t] for k in model.K for n in model.N)
            <= U_max)

    # ── Constraints 10-11: SOC bounds ────────────────────────────────────
    for k in model.K:
        for t in model.T:
            model.constraints.add(model.e[k,t] >= C_bat[k-1] * E_min)
            model.constraints.add(model.e[k,t] <= E_max * C_bat[k-1])

    # ── Constraint 12: end-of-day minimum SOC ────────────────────────────
    for k in model.K:
        model.constraints.add(
            model.e[k, T-1]
            + sum(ch_eff * alpha[n-1] * model.x[k,n,T] for n in model.N)
            >= E_end * C_bat[k-1])

    # ── Objective: min PTO daily cost ────────────────────────────────────
    # PTO pays S_buy[t] per kWh to EBA for charging
    # PTO receives S_sell[t] per kWh from EBA for V2G
    def rule_obj(mod):
        return (sum(S_buy[t-1]  * mod.w_buy[t]  for t in mod.T)
              - sum(S_sell[t-1] * mod.w_sell[t] for t in mod.T))

    model.obj = pyo.Objective(rule=rule_obj, sense=pyo.minimize)

    print('Solving PTO')
    opt = pyo.SolverFactory('highs')
    results = opt.solve(
        model,
        load_solutions=False,
        tee=False,
        options={
            'threads': os.cpu_count(),  # use all available cores
            'parallel': 'on',
        }
    )
    results = opt.solve(model, load_solutions=False, tee=False,
                    options={'threads': 8, 'time_limit': 12000})
    tc = results.solver.termination_condition
    print(f'PTO termination: {tc}')
    if tc in (pyo.TerminationCondition.optimal,
              pyo.TerminationCondition.feasible,
              pyo.TerminationCondition.maxTimeLimit):
        if results.solver.status == pyo.SolverStatus.ok or len(model.solutions) > 0:
            model.solutions.load_from(results)
        print('PTO done')
        return model
    print(f'PTO infeasible: {tc}')
    return None


# ==============================================================================
# OPTIMIZATION RUNNER
# ==============================================================================
def run_optimization(job_id, input_data, price_guidance=None,
                     disturbances=None, optimization_mode='day_ahead',
                     current_timestep=1):
    try:
        price_guidance = price_guidance or {}
        disturbances = disturbances or []
        optimization_mode = (optimization_mode or 'day_ahead').lower()

        print(f'Job {job_id} started')
        print(f'Optimization mode: {optimization_mode}')
        print(f'Current timestep:  {current_timestep}')
        print(f'Price guidance: {price_guidance}')
        print(f'Disturbances:   {disturbances}')

        data = build_dataframes(input_data)
        sc = extract_scalars(
            data,
            price_guidance=price_guidance,
            optimization_mode=optimization_mode,
            current_timestep=current_timestep,
        )

        if disturbances:
            sc = apply_disturbances(sc, disturbances)
            print(f'Applied {len(disturbances)} disturbances')
            print(f'After disturbances: T_start={sc["T_start"]}, T_end={sc["T_end"]}')

        T = sc['T_steps']

        model = solvePTO(sc)
        if model is None:
            mock = dict(MOCK_RESULT)
            mock.update({
                'optimization_mode':     optimization_mode,
                'current_timestep':      sc['current_timestep'],
                'optimized_steps':       sc['T_steps'],
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
        pto_daily_cost = total_pto_buy_cost - total_pto_sell_rev

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
            "optimization_mode":      optimization_mode,
            "current_timestep":       sc['current_timestep'],
            "optimized_steps":        T,
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
            'optimization_mode':     optimization_mode,
            'current_timestep':      current_timestep,
            'optimized_steps':       0,
            'price_guidance_used':   price_guidance,
            'disturbances_applied':  disturbances,
            'mock_reason':           f'Unexpected error: {str(e)}',
        })
        save_job(job_id, mock)


# ==============================================================================
# FLASK ENDPOINTS
# ==============================================================================
@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        payload = request.json or {}
        input_data = payload['input']
        price_guidance = payload.get('price_guidance', {})
        disturbances = payload.get('disturbances', [])
        optimization_mode = payload.get('optimization_mode', 'day_ahead')
        current_timestep = payload.get('current_timestep', 1)
        job_id = str(uuid.uuid4())
        save_job(job_id, {"status": "running"})
        threading.Thread(
            target=run_optimization,
            args=(
                job_id,
                input_data,
                price_guidance,
                disturbances,
                optimization_mode,
                current_timestep,
            )
        ).start()
        return jsonify({
            "job_id": job_id,
            "status": "running",
            "optimization_mode": optimization_mode,
            "current_timestep": current_timestep,
        })
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
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        threaded=True  # add this
    )
