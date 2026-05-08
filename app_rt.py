from flask import Flask, jsonify, request
import json
import os
import threading
import traceback
import uuid
import pandas as pd
import pyomo.environ as pyo

# This is a mock implementation of a real-time optimization endpoint for electric bus fleet scheduling.

app = Flask(__name__)

JOBS_FILE = "/tmp/jobs_rt.json"
_jobs_lock = threading.Lock()


def load_jobs():
    try:
        with open(JOBS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_job(job_id, data):
    with _jobs_lock:
        jobs = load_jobs()
        jobs[job_id] = data
        with open(JOBS_FILE, "w") as f:
            json.dump(jobs, f)


def get_job(job_id):
    return load_jobs().get(job_id)


MOCK_RESULT = {
    "status": "complete",
    "is_mock": True,
    "optimization_mode": "real_time",
    "current_timestep": 1,
    "optimized_steps": 0,
    "mock_reason": "RT optimization infeasible or failed",
    "price_guidance_used": {},
    "disturbances_applied": [],
    "buy_multipliers": [],
    "sell_multipliers": [],
    "period_boundaries": [],
    "avg_grid_price": None,
    "avg_buy_price": None,
    "avg_sell_price": None,
    "pto_daily_cost": None,
    "aggregator_revenue": None,
    "aggregator_buy_margin": None,
    "aggregator_sell_margin": None,
    "total_buy_cost": None,
    "total_sell_revenue": None,
    "total_kwh_sold": None,
    "total_kwh_bought": None,
    "w_buy": [],
    "w_sell": [],
    "energy": [],
    "trip_assignment_by_timestep": {},
    "trip_coverage_by_timestep": {},
    "temporarily_unserved_trip_ids": [],
    "service_interruption_events": [],
    "service_restoration_events": [],
    "reassignment_mapping": {},
    "service_unmet_count": 0,
    "service_unmet_duration": 0,
    "soc_violation_count": 0,
    "availability_conflicts": [],
    "solver_status": "mock",
    "remaining_horizon_start": None,
    "remaining_horizon_end": None,
}


def parse_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def normalize_soc(value):
    soc = float(value)
    if soc > 1.0:
        soc = soc / 100.0
    return max(0.0, min(1.0, soc))


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def get_timestep_minutes(input_data, prices_df):
    if input_data.get("timestep_minutes") not in (None, ""):
        return float(input_data["timestep_minutes"])
    if not prices_df.empty:
        return 1440.0 / float(len(prices_df))
    return 30.0


def infer_full_horizon_steps(timestep_minutes, series_length):
    expected_day_steps = max(1, int(round(1440.0 / float(timestep_minutes))))
    if series_length <= 0:
        return expected_day_steps
    if series_length == expected_day_steps:
        return expected_day_steps
    if expected_day_steps % series_length == 0:
        return expected_day_steps
    return series_length


def expand_series_to_horizon(values, target_steps, series_name):
    values = [float(v) for v in values]
    if not values:
        raise ValueError(f"{series_name} cannot be empty")
    if len(values) == target_steps:
        return values
    if len(values) == 1:
        return values * target_steps
    if target_steps % len(values) == 0:
        repeat = target_steps // len(values)
        expanded = []
        for value in values:
            expanded.extend([value] * repeat)
        return expanded
    expanded = []
    for idx in range(target_steps):
        src_idx = min(len(values) - 1, int(idx * len(values) / target_steps))
        expanded.append(values[src_idx])
    return expanded


def scale_boundaries_to_horizon(boundaries, target_steps, source_steps=None):
    if not boundaries:
        return [target_steps]
    cleaned = [max(1, int(round(float(b)))) for b in boundaries]
    if source_steps is None:
        source_steps = cleaned[-1]
    source_steps = max(1, int(round(float(source_steps))))
    if source_steps == target_steps:
        scaled = cleaned
    else:
        scaled = [
            max(1, min(target_steps, int(round((b / source_steps) * target_steps))))
            for b in cleaned
        ]
    monotonic = []
    prev = 0
    for idx, boundary in enumerate(scaled):
        minimum = prev + 1 if idx < len(scaled) - 1 else target_steps
        boundary = max(minimum, boundary)
        boundary = min(target_steps, boundary)
        monotonic.append(boundary)
        prev = boundary
    monotonic[-1] = target_steps
    return monotonic


def build_multiplier_schedule(multiplier_values, boundaries, target_steps, series_name):
    multipliers = [float(v) for v in multiplier_values]
    if not multipliers:
        raise ValueError(f"{series_name} cannot be empty")
    if len(multipliers) == target_steps:
        return multipliers, list(range(1, target_steps + 1))

    source_steps = boundaries[-1] if boundaries else len(multipliers)
    scaled_boundaries = scale_boundaries_to_horizon(
        boundaries or [len(multipliers)],
        target_steps,
        source_steps,
    )

    if len(multipliers) == len(scaled_boundaries):
        schedule = []
        start = 0
        for multiplier, end in zip(multipliers, scaled_boundaries):
            end = max(start + 1, min(target_steps, int(end)))
            schedule.extend([multiplier] * (end - start))
            start = end
        if len(schedule) < target_steps:
            schedule.extend([multipliers[-1]] * (target_steps - len(schedule)))
        return schedule[:target_steps], scaled_boundaries

    expanded = expand_series_to_horizon(multipliers, target_steps, series_name)
    return expanded, list(range(1, target_steps + 1))


def parse_time_to_step(value, timestep_minutes, full_horizon_steps, is_end=False):
    if pd.isna(value) or value == "":
        raise ValueError("Trip time values cannot be empty")

    text = str(value).strip()
    if ":" in text:
        hour_text, minute_text = text.split(":", 1)
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


def build_dataframes(input_data):
    trip_source = pd.DataFrame(input_data["trip_time"])
    energy_source = pd.DataFrame(input_data.get("energy_consumption", input_data["trip_time"]))
    buses = pd.DataFrame(input_data["buses"])
    chargers = pd.DataFrame(input_data["chargers"])
    prices = pd.DataFrame(input_data.get("grid_prices", input_data.get("prices", [])))
    realtime_state = pd.DataFrame(input_data.get("realtime_state", []))
    trip_state = pd.DataFrame(input_data.get("trip_state", []))

    return {
        "Buses": buses,
        "Chargers": chargers,
        "Trip time": trip_source,
        "Energy consumption": energy_source,
        "Prices": prices,
        "Realtime state": realtime_state,
        "Trip state": trip_state,
        "timestep_minutes": input_data.get("timestep_minutes"),
        "v2g_enabled": input_data.get("v2g_enabled"),
    }


def get_energy_per_step(trip_row, energy_row, timestep_hours):
    energy_kwhkm = (
        energy_row.get("energy_kwhkm")
        if "energy_kwhkm" in energy_row
        else energy_row.get("uncertain_energy_kwhkm")
    )
    if energy_kwhkm in (None, "") or pd.isna(energy_kwhkm):
        energy_kwhkm = trip_row.get("energy_kwhkm")
    speed_kmh = (
        energy_row.get("average_velocity_kmh")
        if "average_velocity_kmh" in energy_row
        else trip_row.get("average_velocity_kmh", 12.0)
    )
    return max(0.0, safe_float(energy_kwhkm) * safe_float(speed_kmh, 12.0) * timestep_hours)


def build_rt_context(data, payload_price_guidance, current_timestep, disturbances):
    buses_df = data["Buses"].copy()
    chargers_df = data["Chargers"].copy()
    trips_df = data["Trip time"].copy()
    energy_df = data["Energy consumption"].copy()
    prices_df = data["Prices"].copy()
    realtime_df = data["Realtime state"].copy()
    trip_state_df = data["Trip state"].copy()

    if buses_df.empty or chargers_df.empty or trips_df.empty or prices_df.empty:
        raise ValueError("buses, chargers, trip_time, and prices inputs must all be non-empty")

    timestep_minutes = get_timestep_minutes(data, prices_df)
    timestep_hours = timestep_minutes / 60.0
    full_horizon_steps = infer_full_horizon_steps(timestep_minutes, len(prices_df))
    current_timestep = max(1, min(int(current_timestep or 1), full_horizon_steps))

    prices_df = prices_df.rename(
        columns={
            "spot_market": "spot_market",
            "spot_price": "spot_market",
            "price": "spot_market",
            "timestep": "timestep",
            "time": "timestep",
        }
    )
    spot_source = pd.to_numeric(prices_df["spot_market"], errors="coerce").dropna().tolist()
    if not spot_source:
        raise ValueError("prices input must include a valid spot_market series")
    spot_full = expand_series_to_horizon(spot_source, full_horizon_steps, "spot_market")

    if not realtime_df.empty and "current_timestep" in realtime_df.columns:
        rt_current = pd.to_numeric(realtime_df["current_timestep"], errors="coerce").dropna()
        if not rt_current.empty:
            current_timestep = max(1, min(int(rt_current.iloc[0]), full_horizon_steps))

    realtime_by_bus = {}
    for row in realtime_df.to_dict(orient="records"):
        bus_id = safe_int(row.get("bus_id") or row.get("Bus ID"))
        if bus_id > 0:
            realtime_by_bus[bus_id] = row

    buses = []
    for row in buses_df.to_dict(orient="records"):
        bus_id = safe_int(row.get("bus_id"))
        bus_kwh = safe_float(row.get("bus_kwh"))
        rt_row = realtime_by_bus.get(bus_id, {})
        current_energy = rt_row.get("current_energy_kwh")
        current_soc = rt_row.get("current_soc", row.get("initial_soc", 20.0))
        if pd.notna(current_energy):
            initial_soc_rt = max(0.0, min(1.0, safe_float(current_energy) / bus_kwh))
        else:
            initial_soc_rt = normalize_soc(current_soc)
        buses.append(
            {
                "bus_id": bus_id,
                "physical_bus_id": safe_int(row.get("physical_bus_id", bus_id), bus_id),
                "bus_kwh": bus_kwh,
                "initial_soc": normalize_soc(row.get("initial_soc", 20.0)),
                "initial_soc_rt": initial_soc_rt,
                "availability_status": str(
                    rt_row.get("availability_status", row.get("availability_status", "available"))
                ).strip().lower(),
                "operation_status": str(rt_row.get("operation_status", "idle")).strip().lower(),
                "current_trip_id": safe_int(rt_row.get("current_trip_id"), 0),
                "reassignable": parse_bool(rt_row.get("reassignable", row.get("reassignable", True)), True),
            }
        )

    chargers = []
    for row in chargers_df.to_dict(orient="records"):
        charger_kw = safe_float(row.get("max_power_kw", row.get("charger_kw")))
        chargers.append(
            {
                "charger_id": safe_int(row.get("charger_id")),
                "charger_kw": charger_kw,
                "alpha": charger_kw * timestep_hours,
            }
        )

    price_guidance = payload_price_guidance or {}
    price_multiplier_vector = pd.to_numeric(
        prices_df.get("price_multiplier", pd.Series([1.0] * len(prices_df))),
        errors="coerce",
    ).fillna(1.0).tolist()
    price_multiplier_full = expand_series_to_horizon(
        price_multiplier_vector,
        full_horizon_steps,
        "price_multiplier",
    )

    explicit_buy = prices_df.get("buy_price_rt")
    explicit_sell = prices_df.get("sell_price_rt")
    has_explicit_prices = explicit_buy is not None and explicit_sell is not None

    if has_explicit_prices and pd.to_numeric(explicit_buy, errors="coerce").notna().any():
        buy_full = expand_series_to_horizon(
            pd.to_numeric(explicit_buy, errors="coerce").fillna(method="ffill").fillna(method="bfill").tolist(),
            full_horizon_steps,
            "buy_price_rt",
        )
        sell_full = expand_series_to_horizon(
            pd.to_numeric(explicit_sell, errors="coerce").fillna(method="ffill").fillna(method="bfill").tolist(),
            full_horizon_steps,
            "sell_price_rt",
        )
        buy_mult_full = [(buy_full[t] / spot_full[t]) if spot_full[t] else 0.0 for t in range(full_horizon_steps)]
        sell_mult_full = [(sell_full[t] / spot_full[t]) if spot_full[t] else 0.0 for t in range(full_horizon_steps)]
        boundaries = list(range(1, full_horizon_steps + 1))
    else:
        buy_mults_raw = price_guidance.get("buy_multipliers", [1.05, 1.10, 1.05])
        sell_mults_raw = price_guidance.get("sell_multipliers", [0.80, 0.85, 0.80])
        raw_boundaries = price_guidance.get("period_boundaries")
        default_boundaries = list(range(1, len(buy_mults_raw) + 1))
        buy_mult_full, boundaries = build_multiplier_schedule(
            buy_mults_raw,
            raw_boundaries or default_boundaries,
            full_horizon_steps,
            "buy multipliers",
        )
        sell_mult_full, _ = build_multiplier_schedule(
            sell_mults_raw,
            raw_boundaries or default_boundaries,
            full_horizon_steps,
            "sell multipliers",
        )
        sell_mult_full = [
            min(sell_mult_full[t], buy_mult_full[t] - 0.01) for t in range(full_horizon_steps)
        ]
        buy_full = [spot_full[t] * buy_mult_full[t] for t in range(full_horizon_steps)]
        sell_full = [spot_full[t] * sell_mult_full[t] for t in range(full_horizon_steps)]

    price_disturbance = {}
    delay_disturbance = {}
    energy_disturbance = {}
    unavailable_buses = set()
    for disturbance in disturbances or []:
        family = str(
            disturbance.get("disturbance_type")
            or disturbance.get("family")
            or disturbance.get("scenario_family")
            or ""
        ).strip().lower()
        target_bus = safe_int(disturbance.get("bus_id"), 0)
        if family in {"price", "price_deviation", "rt_price"}:
            multiplier = safe_float(
                disturbance.get("multiplier"),
                1.0 + (safe_float(disturbance.get("percent"), 0.0) / 100.0),
            )
            price_disturbance["multiplier"] = multiplier
        elif family in {"delay", "late", "early_return"}:
            delay_disturbance[target_bus] = safe_int(disturbance.get("delay_minutes"), 0)
        elif family in {"energy", "energy_deviation"}:
            multiplier = safe_float(
                disturbance.get("multiplier"),
                1.0 + (safe_float(disturbance.get("percent"), 0.0) / 100.0),
            )
            energy_disturbance[target_bus] = multiplier
        elif family in {"availability", "breakdown", "unavailable"}:
            if target_bus > 0:
                unavailable_buses.add(target_bus)

    if "multiplier" in price_disturbance:
        factor = price_disturbance["multiplier"]
        spot_full = [p * factor for p in spot_full]
        buy_full = [p * factor for p in buy_full]
        sell_full = [p * factor for p in sell_full]

    prices = {
        "spot": spot_full[current_timestep - 1 :],
        "buy": buy_full[current_timestep - 1 :],
        "sell": sell_full[current_timestep - 1 :],
        "buy_multipliers": buy_mult_full[current_timestep - 1 :],
        "sell_multipliers": sell_mult_full[current_timestep - 1 :],
        "boundaries": [
            boundary - current_timestep + 1
            for boundary in boundaries
            if current_timestep <= boundary <= full_horizon_steps
        ],
    }
    if not prices["boundaries"] or prices["boundaries"][-1] != len(prices["spot"]):
        prices["boundaries"].append(len(prices["spot"]))

    trip_state_map = {
        safe_int(row.get("trip_id")): row for row in trip_state_df.to_dict(orient="records")
    }

    trips = []
    energy_rows = energy_df.to_dict(orient="records")
    for idx, row in enumerate(trips_df.to_dict(orient="records")):
        trip_id = safe_int(row.get("trip_id"), idx + 1)
        bus_id = safe_int(row.get("bus_id"), trip_id)
        start_raw = parse_time_to_step(row.get("time_begin"), timestep_minutes, full_horizon_steps, False)
        end_raw = parse_time_to_step(row.get("time_end"), timestep_minutes, full_horizon_steps, True)
        delay_minutes = safe_int(row.get("delay_minutes"), 0)
        delay_minutes = safe_int(delay_disturbance.get(bus_id, delay_minutes), delay_minutes)
        delay_steps = int(round(delay_minutes / timestep_minutes))
        start = min(full_horizon_steps, max(1, start_raw + delay_steps))
        end = min(full_horizon_steps, max(start + 1, end_raw + delay_steps))

        state_row = trip_state_map.get(trip_id, {})
        if state_row:
            remaining_active = safe_int(state_row.get("remaining_active_timesteps"), max(0, end - current_timestep + 1))
            remaining_energy_need = safe_float(state_row.get("remaining_energy_need"), 0.0)
            interruption_allowed = parse_bool(state_row.get("interruption_allowed", True), True)
        else:
            remaining_active = max(0, end - max(start, current_timestep) + 1)
            remaining_energy_need = None
            interruption_allowed = True

        energy_row = energy_rows[idx] if idx < len(energy_rows) else row
        energy_multiplier = safe_float(row.get("energy_multiplier"), 1.0)
        energy_multiplier = safe_float(energy_disturbance.get(bus_id, energy_multiplier), energy_multiplier)
        energy_per_step = get_energy_per_step(row, energy_row, timestep_hours) * energy_multiplier
        if remaining_energy_need is None or remaining_energy_need <= 0:
            remaining_energy_need = energy_per_step * remaining_active

        if end < current_timestep:
            continue

        active_now = start <= current_timestep < end
        trip_progress_status = str(
            row.get("trip_progress_status", state_row.get("currently_operated", "active" if active_now else "pending"))
        ).strip().lower()
        planned_bus_id = safe_int(row.get("planned_bus_id", row.get("assigned_bus_id_rt", bus_id)), bus_id)

        trips.append(
            {
                "trip_id": trip_id,
                "planned_bus_id": planned_bus_id,
                "route_id": row.get("route_id", trip_id),
                "start_abs": start,
                "end_abs": end,
                "start_rt": max(1, start - current_timestep + 1),
                "end_rt": min(full_horizon_steps - current_timestep + 1, end - current_timestep + 1),
                "active_now": active_now,
                "remaining_active_steps": remaining_active,
                "energy_per_step": max(0.0, energy_per_step),
                "remaining_energy_need": max(0.0, remaining_energy_need),
                "trip_progress_status": trip_progress_status,
                "interruption_allowed": interruption_allowed,
                "status": str(row.get("status", "scheduled")).strip().lower(),
            }
        )

    for bus in buses:
        if bus["bus_id"] in unavailable_buses:
            bus["availability_status"] = "unavailable"

    return {
        "buses": buses,
        "chargers": chargers,
        "trips": trips,
        "prices": prices,
        "current_timestep": current_timestep,
        "full_horizon_steps": full_horizon_steps,
        "timestep_minutes": timestep_minutes,
        "timestep_hours": timestep_hours,
        "v2g_enabled": parse_bool(data.get("v2g_enabled", True), True),
        "price_guidance": price_guidance,
    }


def solve_rt_rescheduling(ctx):
    buses = ctx["buses"]
    chargers = ctx["chargers"]
    trips = ctx["trips"]
    prices = ctx["prices"]
    timestep_hours = ctx["timestep_hours"]
    v2g_enabled = ctx["v2g_enabled"]
    T = len(prices["spot"])

    if not trips:
        raise ValueError("No remaining trips to optimize for the real-time horizon")

    K = [bus["bus_id"] for bus in buses]
    I = [trip["trip_id"] for trip in trips]
    N = [charger["charger_id"] for charger in chargers]
    trip_by_id = {trip["trip_id"]: trip for trip in trips}
    bus_by_id = {bus["bus_id"]: bus for bus in buses}
    charger_by_id = {charger["charger_id"]: charger for charger in chargers}

    active_trip_steps = {
        trip["trip_id"]: [
            t for t in range(1, T + 1) if trip["start_rt"] <= t < trip["end_rt"]
        ]
        for trip in trips
    }
    initial_active_bus = {
        trip["trip_id"]: trip["planned_bus_id"] if trip["active_now"] else None
        for trip in trips
    }

    model = pyo.ConcreteModel()
    model.K = pyo.Set(initialize=K, ordered=True)
    model.I = pyo.Set(initialize=I, ordered=True)
    model.N = pyo.Set(initialize=N, ordered=True)
    model.T = pyo.RangeSet(1, T)

    model.p_buy = pyo.Param(model.T, initialize=lambda m, t: prices["buy"][t - 1])
    model.p_sell = pyo.Param(model.T, initialize=lambda m, t: prices["sell"][t - 1])
    model.spot = pyo.Param(model.T, initialize=lambda m, t: prices["spot"][t - 1])
    model.c_bat = pyo.Param(model.K, initialize=lambda m, k: bus_by_id[k]["bus_kwh"])
    model.e0 = pyo.Param(model.K, initialize=lambda m, k: bus_by_id[k]["initial_soc_rt"] * bus_by_id[k]["bus_kwh"])
    model.alpha = pyo.Param(model.N, initialize=lambda m, n: charger_by_id[n]["alpha"])
    model.trip_energy = pyo.Param(model.I, initialize=lambda m, i: trip_by_id[i]["energy_per_step"])
    model.start_rt = pyo.Param(model.I, initialize=lambda m, i: trip_by_id[i]["start_rt"])
    model.end_rt = pyo.Param(model.I, initialize=lambda m, i: trip_by_id[i]["end_rt"])
    model.remaining_need = pyo.Param(model.I, initialize=lambda m, i: trip_by_id[i]["remaining_energy_need"])

    model.s = pyo.Var(model.K, model.I, model.T, domain=pyo.Binary)
    model.x = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.y = pyo.Var(model.K, model.N, model.T, domain=pyo.Binary)
    model.c = pyo.Var(model.K, model.T, domain=pyo.Binary)
    model.e = pyo.Var(model.K, model.T, domain=pyo.NonNegativeReals)
    model.w_buy = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.w_sell = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.u = pyo.Var(model.I, model.T, domain=pyo.Binary)
    model.switch = pyo.Var(model.K, model.I, model.T, domain=pyo.Binary)

    model.constraints = pyo.ConstraintList()

    service_penalty = 1e5
    dry_run_penalty = 5e4
    switch_penalty = 50.0
    interruption_penalty = 2e3
    active_break_penalty = 1e4
    site_cap = sum(charger["alpha"] for charger in chargers)
    e_min_fraction = 0.2
    e_max_fraction = 1.0
    e_end_fraction = 0.2
    ch_eff = 0.90
    dch_eff = 1.0 / 0.90

    for i in model.I:
        active_steps = set(active_trip_steps[i])
        for t in model.T:
            if t not in active_steps:
                model.constraints.add(sum(model.s[k, i, t] for k in model.K) == 0)
                model.constraints.add(model.u[i, t] == 0)
            else:
                model.constraints.add(sum(model.s[k, i, t] for k in model.K) + model.u[i, t] == 1)

    for k in model.K:
        if bus_by_id[k]["availability_status"] == "unavailable":
            for i in model.I:
                for t in model.T:
                    model.constraints.add(model.s[k, i, t] == 0)
            for n in model.N:
                for t in model.T:
                    model.constraints.add(model.x[k, n, t] == 0)
                    model.constraints.add(model.y[k, n, t] == 0)
            continue

        for t in model.T:
            model.constraints.add(sum(model.s[k, i, t] for i in model.I) + model.c[k, t] <= 1)
            model.constraints.add(sum(model.x[k, n, t] + model.y[k, n, t] for n in model.N) == model.c[k, t])

    for n in model.N:
        for t in model.T:
            model.constraints.add(sum(model.x[k, n, t] + model.y[k, n, t] for k in model.K) <= 1)

    for t in model.T:
        total_charge = sum(model.alpha[n] * model.x[k, n, t] for k in model.K for n in model.N)
        total_discharge = sum(model.alpha[n] * model.y[k, n, t] for k in model.K for n in model.N)
        model.constraints.add(model.w_buy[t] == total_charge)
        model.constraints.add(model.w_sell[t] == total_discharge)
        model.constraints.add(total_charge <= site_cap)
        model.constraints.add(total_discharge <= site_cap)
        if not v2g_enabled:
            model.constraints.add(model.w_sell[t] == 0)

    for k in model.K:
        cap = bus_by_id[k]["bus_kwh"]
        for t in model.T:
            trip_draw = sum(model.trip_energy[i] * model.s[k, i, t] for i in model.I)
            charge_in = sum(model.alpha[n] * ch_eff * model.x[k, n, t] for n in model.N)
            discharge_out = sum(model.alpha[n] * dch_eff * model.y[k, n, t] for n in model.N)
            if t == 1:
                model.constraints.add(
                    model.e[k, t] == model.e0[k] + charge_in - discharge_out - trip_draw
                )
            else:
                model.constraints.add(
                    model.e[k, t] == model.e[k, t - 1] + charge_in - discharge_out - trip_draw
                )
            model.constraints.add(model.e[k, t] >= e_min_fraction * cap)
            model.constraints.add(model.e[k, t] <= e_max_fraction * cap)
        model.constraints.add(model.e[k, T] >= e_end_fraction * cap)

    for i in model.I:
        trip = trip_by_id[i]
        total_served_energy = sum(
            model.trip_energy[i] * model.s[k, i, t]
            for k in model.K
            for t in active_trip_steps[i]
        )
        required_service_steps = max(0, int(round(trip["remaining_energy_need"] / max(trip["energy_per_step"], 1e-6))))
        required_service_steps = min(required_service_steps, len(active_trip_steps[i]))
        model.constraints.add(
            sum(model.s[k, i, t] for k in model.K for t in active_trip_steps[i]) + sum(model.u[i, t] for t in active_trip_steps[i]) == len(active_trip_steps[i])
        )
        model.constraints.add(total_served_energy + sum(model.u[i, t] * model.trip_energy[i] for t in active_trip_steps[i]) >= model.remaining_need[i])
        if not trip["interruption_allowed"] and trip["active_now"]:
            for t in active_trip_steps[i]:
                model.constraints.add(model.u[i, t] == 0)

        initial_bus = initial_active_bus[i]
        if trip["active_now"] and initial_bus in K:
            model.constraints.add(model.switch[initial_bus, i, 1] >= 1 - model.s[initial_bus, i, 1])

    for k in model.K:
        for i in model.I:
            for t in model.T:
                if t == 1:
                    base_assigned = 1 if initial_active_bus.get(i) == k else 0
                    model.constraints.add(model.switch[k, i, t] >= model.s[k, i, t] - base_assigned)
                    model.constraints.add(model.switch[k, i, t] >= base_assigned - model.s[k, i, t])
                else:
                    model.constraints.add(model.switch[k, i, t] >= model.s[k, i, t] - model.s[k, i, t - 1])
                    model.constraints.add(model.switch[k, i, t] >= model.s[k, i, t - 1] - model.s[k, i, t])

    total_buy_cost = sum(model.p_buy[t] * model.w_buy[t] for t in model.T)
    total_sell_revenue = sum(model.p_sell[t] * model.w_sell[t] for t in model.T)
    total_unserved = sum(model.u[i, t] for i in model.I for t in active_trip_steps[i])
    total_switching = sum(model.switch[k, i, t] for k in model.K for i in model.I for t in model.T)
    active_breaks = sum(
        model.u[i, 1]
        for i in model.I
        if trip_by_id[i]["active_now"]
    )

    model.obj = pyo.Objective(
        expr=service_penalty * total_unserved
        + interruption_penalty * total_unserved
        + active_break_penalty * active_breaks
        + switch_penalty * total_switching
        + dry_run_penalty * 0
        + total_buy_cost
        - total_sell_revenue,
        sense=pyo.minimize,
    )

    solver = None
    for solver_name in ("gurobi", "appsi_highs", "highs", "cbc", "glpk"):
        try:
            candidate = pyo.SolverFactory(solver_name)
            if candidate is not None and candidate.available(False):
                solver = candidate
                break
        except Exception:
            continue
    if solver is None:
        raise RuntimeError("No supported MILP solver available for app_rt.py")

    solved = solver.solve(model, tee=False)
    term = str(solved.solver.termination_condition).lower()
    status = str(solved.solver.status).lower()
    if "optimal" not in term and "feasible" not in term:
        return None, {"solver_status": f"{status}/{term}"}
    return model, {"solver_status": f"{status}/{term}"}


def extract_time_series_results(model, ctx):
    buses = ctx["buses"]
    trips = ctx["trips"]
    prices = ctx["prices"]
    T = len(prices["spot"])
    K = [bus["bus_id"] for bus in buses]
    I = [trip["trip_id"] for trip in trips]

    energy = [
        [float(pyo.value(model.e[k, t])) for t in range(1, T + 1)]
        for k in K
    ]
    w_buy = [float(pyo.value(model.w_buy[t])) for t in range(1, T + 1)]
    w_sell = [float(pyo.value(model.w_sell[t])) for t in range(1, T + 1)]

    trip_assignment_by_timestep = {}
    trip_coverage_by_timestep = {}
    temporarily_unserved = set()
    interruption_events = []
    restoration_events = []
    reassignment_mapping = {}

    previous_covered = {}
    previous_bus = {}

    active_trip_windows = {
        trip["trip_id"]: set(range(trip["start_rt"], trip["end_rt"])) for trip in trips
    }

    for t in range(1, T + 1):
        step_assignments = {}
        step_coverage = {}
        for trip in trips:
            i = trip["trip_id"]
            is_active = t in active_trip_windows[i]
            assigned_bus = None
            for k in K:
                if float(pyo.value(model.s[k, i, t])) > 0.5:
                    assigned_bus = k
                    break
            covered = assigned_bus is not None if is_active else None
            step_assignments[str(i)] = assigned_bus if is_active else None
            step_coverage[str(i)] = covered
            if is_active and not assigned_bus:
                temporarily_unserved.add(i)

            prior_covered = previous_covered.get(i, covered)
            prior_bus = previous_bus.get(i, assigned_bus)
            abs_t = ctx["current_timestep"] + t - 1
            if prior_covered is True and covered is False:
                interruption_events.append(
                    {
                        "trip_id": i,
                        "timestep": abs_t,
                        "previous_bus_id": prior_bus,
                    }
                )
            if prior_covered is False and covered is True:
                restoration_events.append(
                    {
                        "trip_id": i,
                        "timestep": abs_t,
                        "bus_id": assigned_bus,
                    }
                )
            if covered:
                reassignment_mapping[str(i)] = assigned_bus

            previous_covered[i] = covered
            previous_bus[i] = assigned_bus if is_active else prior_bus

        trip_assignment_by_timestep[str(ctx["current_timestep"] + t - 1)] = step_assignments
        trip_coverage_by_timestep[str(ctx["current_timestep"] + t - 1)] = step_coverage

    return {
        "energy": energy,
        "w_buy": w_buy,
        "w_sell": w_sell,
        "trip_assignment_by_timestep": trip_assignment_by_timestep,
        "trip_coverage_by_timestep": trip_coverage_by_timestep,
        "temporarily_unserved_trip_ids": sorted(temporarily_unserved),
        "service_interruption_events": interruption_events,
        "service_restoration_events": restoration_events,
        "reassignment_mapping": reassignment_mapping,
    }


def run_optimization(
    job_id,
    input_data,
    price_guidance=None,
    disturbances=None,
    optimization_mode="real_time",
    current_timestep=1,
):
    try:
        price_guidance = price_guidance or {}
        disturbances = disturbances or []
        optimization_mode = (optimization_mode or "real_time").lower()
        if optimization_mode != "real_time":
            raise ValueError("app_rt.py only supports optimization_mode='real_time'")

        print(f"RT job {job_id} started")
        print(f"Optimization mode: {optimization_mode}")
        print(f"Current timestep: {current_timestep}")
        print(f"Price guidance: {price_guidance}")
        print(f"Disturbances: {disturbances}")

        data = build_dataframes(input_data)
        ctx = build_rt_context(
            data,
            payload_price_guidance=price_guidance,
            current_timestep=current_timestep,
            disturbances=disturbances,
        )

        model, solve_meta = solve_rt_rescheduling(ctx)
        if model is None:
            mock = dict(MOCK_RESULT)
            mock.update(
                {
                    "optimization_mode": optimization_mode,
                    "current_timestep": ctx["current_timestep"],
                    "optimized_steps": len(ctx["prices"]["spot"]),
                    "v2g_enabled": ctx["v2g_enabled"],
                    "price_guidance_used": price_guidance,
                    "disturbances_applied": disturbances,
                    "mock_reason": "RT fleet-rescheduling model infeasible",
                    "buy_multipliers": ctx["prices"]["buy_multipliers"],
                    "sell_multipliers": ctx["prices"]["sell_multipliers"],
                    "period_boundaries": ctx["prices"]["boundaries"],
                    "avg_grid_price": sum(ctx["prices"]["spot"]) / len(ctx["prices"]["spot"]),
                    "solver_status": solve_meta.get("solver_status"),
                    "remaining_horizon_start": ctx["current_timestep"],
                    "remaining_horizon_end": ctx["current_timestep"] + len(ctx["prices"]["spot"]) - 1,
                }
            )
            save_job(job_id, mock)
            return

        series = extract_time_series_results(model, ctx)
        T = len(ctx["prices"]["spot"])
        total_buy_cost = sum(ctx["prices"]["buy"][t - 1] * pyo.value(model.w_buy[t]) for t in range(1, T + 1))
        total_sell_revenue = sum(ctx["prices"]["sell"][t - 1] * pyo.value(model.w_sell[t]) for t in range(1, T + 1))
        pto_daily_cost = total_buy_cost - total_sell_revenue
        agg_buy_margin = sum(
            (ctx["prices"]["buy"][t - 1] - ctx["prices"]["spot"][t - 1]) * pyo.value(model.w_buy[t])
            for t in range(1, T + 1)
        )
        agg_sell_margin = sum(
            (ctx["prices"]["spot"][t - 1] - ctx["prices"]["sell"][t - 1]) * pyo.value(model.w_sell[t])
            for t in range(1, T + 1)
        )
        aggregator_revenue = agg_buy_margin + agg_sell_margin
        total_unserved_duration = len(
            [
                1
                for step_coverage in series["trip_coverage_by_timestep"].values()
                for covered in step_coverage.values()
                if covered is False
            ]
        )

        result = {
            "status": "complete",
            "is_mock": False,
            "optimization_mode": optimization_mode,
            "current_timestep": ctx["current_timestep"],
            "optimized_steps": T,
            "v2g_enabled": ctx["v2g_enabled"],
            "price_guidance_used": price_guidance,
            "disturbances_applied": disturbances,
            "buy_multipliers": ctx["prices"]["buy_multipliers"],
            "sell_multipliers": ctx["prices"]["sell_multipliers"],
            "period_boundaries": ctx["prices"]["boundaries"],
            "avg_grid_price": sum(ctx["prices"]["spot"]) / len(ctx["prices"]["spot"]),
            "avg_buy_price": sum(ctx["prices"]["buy"]) / len(ctx["prices"]["buy"]),
            "avg_sell_price": sum(ctx["prices"]["sell"]) / len(ctx["prices"]["sell"]),
            "pto_daily_cost": pto_daily_cost,
            "aggregator_revenue": aggregator_revenue,
            "aggregator_buy_margin": agg_buy_margin,
            "aggregator_sell_margin": agg_sell_margin,
            "total_buy_cost": total_buy_cost,
            "total_sell_revenue": total_sell_revenue,
            "total_kwh_sold": sum(series["w_sell"]),
            "total_kwh_bought": sum(series["w_buy"]),
            "w_buy": series["w_buy"],
            "w_sell": series["w_sell"],
            "energy": series["energy"],
            "trip_assignment_by_timestep": series["trip_assignment_by_timestep"],
            "trip_coverage_by_timestep": series["trip_coverage_by_timestep"],
            "temporarily_unserved_trip_ids": series["temporarily_unserved_trip_ids"],
            "service_interruption_events": series["service_interruption_events"],
            "service_restoration_events": series["service_restoration_events"],
            "reassignment_mapping": series["reassignment_mapping"],
            "service_unmet_count": len(series["temporarily_unserved_trip_ids"]),
            "service_unmet_duration": total_unserved_duration,
            "soc_violation_count": 0,
            "availability_conflicts": [],
            "solver_status": solve_meta.get("solver_status"),
            "remaining_horizon_start": ctx["current_timestep"],
            "remaining_horizon_end": ctx["current_timestep"] + T - 1,
        }
        save_job(job_id, result)
        print(f"RT job {job_id} complete")

    except Exception as e:
        traceback.print_exc()
        mock = dict(MOCK_RESULT)
        mock.update(
            {
                "optimization_mode": optimization_mode,
                "current_timestep": current_timestep,
                "optimized_steps": 0,
                "price_guidance_used": price_guidance or {},
                "disturbances_applied": disturbances or [],
                "mock_reason": f"Unexpected RT error: {str(e)}",
            }
        )
        save_job(job_id, mock)


@app.route("/optimize", methods=["POST"])
def optimize():
    try:
        payload = request.json or {}
        input_data = payload["input"]
        price_guidance = payload.get("price_guidance", {})
        disturbances = payload.get("disturbances", [])
        optimization_mode = payload.get("optimization_mode", "real_time")
        current_timestep = payload.get("current_timestep", 1)
        if "v2g_enabled" in payload:
            input_data["v2g_enabled"] = payload.get("v2g_enabled")
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
            ),
        ).start()
        return jsonify(
            {
                "job_id": job_id,
                "status": "running",
                "optimization_mode": optimization_mode,
                "current_timestep": current_timestep,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/result/<job_id>", methods=["GET"])
def get_result(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


@app.route("/health", methods=["GET"])
def health():
    jobs = load_jobs()
    return jsonify(
        {
            "status": "ok",
            "active_jobs": len([j for j in jobs.values() if j.get("status") == "running"]),
            "total_jobs": len(jobs),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5003)))
