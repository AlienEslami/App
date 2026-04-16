# PTO (Public Transport Operator) Optimization System

A Pyomo-based optimization engine for electric bus fleet charging and vehicle-to-grid (V2G) scheduling with decoupled pricing inputs and two-level operational modes.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Features](#key-features)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [API Reference](#api-reference)
9. [Input File Format](#input-file-format)
10. [Output Metrics](#output-metrics)
11. [Examples](#examples)

---

## Overview

This system optimizes electric bus fleet operations by determining:
- **When and where buses charge** at available chargers
- **Vehicle-to-grid discharge** patterns (if enabled) to generate revenue
- **Cost minimization** under dynamic energy pricing

### Design Philosophy

**Decoupled Pricing Model:**
- **Grid Spot Price (P)**: Wholesale market price (fixed input, read-only)
- **Aggregator Buy Tariff (S_buy)**: What the operator pays per kWh (aggregator-controlled)
- **Aggregator Sell Tariff (S_sell)**: What the operator receives per kWh for V2G (aggregator-controlled)

This separation allows the aggregator (LLM or external service) to dynamically adjust S_buy and S_sell independent of grid market conditions.

**Two-Level Optimization Modes:**
1. **Day-Ahead**: Full 24-hour horizon planning (creates a baseline schedule)
2. **Real-Time**: Rolling re-optimization from current timestep onward (adapts to fleet state changes, delays, price updates)

---

## Architecture

### Workflow

```
Input Data (Excel)
    ↓
build_dataframes()
    ↓ (parses: buses, chargers, trips, prices, tariffs, realtime state)
    ↓
extract_scalars()
    ↓ (converts to optimization parameters, applies real-time filtering)
    ↓
apply_disturbances()
    ↓ (optional: delay, breakdown, early return)
    ↓
solvePTO()
    ↓ (Pyomo + HiGHS solver)
    ↓
run_optimization()
    ↓ (computes economic metrics, returns results)
    ↓
Output: JSON with cost, energy metrics, time series
```

### Core Components

#### 1. **build_dataframes()** (app.py, lines 59–138)
Parses input Excel files into standardized DataFrames:
- **Buses**: bus_id, battery capacity (kWh), initial SOC
- **Chargers**: charger_id, power rating (kW)
- **Trips**: bus_id, trip start/end times, energy consumption (kWh)
- **Spot Prices**: timestep, grid price per kWh
- **Tariffs**: (optional) timestep, buy_price, sell_price
- **Realtime State**: (optional) current_timestep, per-bus SOC/energy, status, delays

**Key Feature**: Auto-detection of column names (flexible Excel naming conventions)

#### 2. **extract_scalars()** (app.py, lines 305–605)
Converts DataFrames → optimization parameters:
- **Price handling** (lines 338–405):
  - If explicit tariffs provided: Uses 'Buy Tariff' and 'Sell Tariff' columns
  - Otherwise: Derives via multiplier-based pricing (with validation constraints)
- **Real-time filtering** (lines 330–509):
  - Clamps current_timestep to valid range [1, total_steps]
  - Applies delays to trip start/end times
  - Filters only active trips (trip end ≥ current_timestep)
  - Remaps horizons (P, tariffs) from current_timestep onward
- **Fleet state ingestion**: Loads current SOC, energy, status from realtime_state sheet
- **Initial conditions**: Day-ahead uses fleet baseline; real-time uses realtime_state values

#### 3. **solvePTO()** (app.py, lines 673–835)
Pyomo optimization model definition:
- **Decision Variables**:
  - `b[k,i,t]`: Bus k on trip i at timestep t (binary)
  - `x[k,n,t]`: Charging power at bus k, charger n, time t (continuous)
  - `y[k,n,t]`: V2G discharge at bus k, charger n, time t (continuous)
  - `e[k,t]`: Battery energy for bus k at time t (continuous)
  - `w_buy[t]`: Total grid power purchased at time t
  - `w_sell[t]`: Total V2G power sold at time t

- **Objective**:
  ```
  minimize: Σ_t S_buy[t] · w_buy[t] - Σ_t S_sell[t] · w_sell[t]
  ```
  (Minimize cost of grid purchases, maximize revenue from V2G sales)

- **Key Constraints**:
  - Trip continuity: Bus must be on same trip entire duration
  - Energy balance: e[k,t+1] = e[k,t] + charge_kWh - discharge_kWh - consumption_kWh
  - SOC bounds: E_min ≤ e[k,t] ≤ E_max
  - End-of-day SOC: e[k,T] ≥ E_end (minimum reserve for next day)
  - Charger capacity: Only one bus per charger
  - V2G gate: If `v2g_enabled=false`, all discharging disabled

- **Solver**: HiGHS (linear/MIP via Pyomo)

#### 4. **apply_disturbances()** (app.py, lines 608–632)
Simulates operational events:
- **'late'**: Delays trip start and end
- **'breakdown'**: Extends trip duration (late start + late end)
- **'early_return'**: Shortens trip end time
- **'delay_minutes'**: Shifts trip timing via realtime_state

#### 5. **run_optimization()** (app.py, lines 859–930)
Orchestrates full pipeline:
1. Calls build_dataframes → extract_scalars → apply_disturbances → solvePTO
2. Computes economic metrics:
   - `pto_daily_cost`: Total buy cost minus V2G revenue
   - `aggregator_revenue`: Buy margin + sell margin
   - `total_kwh_bought`: Grid energy purchases
   - `total_kwh_sold`: V2G energy sales
3. Saves results to JSON with time series

---

## Key Features

### ✅ Decoupled Pricing
- **Independent Input Files**: Spot prices (grid market) separate from tariffs (aggregator-set)
- **Backward Compatible**: Falls back to multiplier-based pricing if tariffs not provided
- **Dynamic Aggregator Control**: LLM/aggregator can update S_buy, S_sell without changing grid input

### ✅ Two-Level Optimization
| Aspect | Day-Ahead | Real-Time |
|--------|-----------|-----------|
| **Horizon** | Full 24 hours (T_steps) | Current timestep to remaining steps |
| **Trip Set** | All scheduled trips | Only active trips (T_end ≥ current_timestep) |
| **Fleet State** | Baseline SOC (initial_soc) | Observed SOC (realtime_state) |
| **Use Case** | Planning & baseline | Reactive re-optimization |

### ✅ V2G Support
- Configurable via `v2g_enabled` flag in Settings
- When enabled: Buses can discharge to grid (w_sell) at aggregator's sell price
- When disabled: All discharging blocked (emergency mode)

### ✅ Fleet State Tracking
Real-time mode ingests per-bus metrics:
- Current SOC or energy (kWh)
- Operation status: in_trip, charging, discharging, idle
- Delay flags (e.g., traffic delays shift trip timing)

### ✅ Flexible Input
- Multiple sheet name aliases (e.g., "Spot Prices" or "Prices")
- External file support (separate Excel files for spot prices and tariffs)
- Auto-inferred timestep_minutes from price series length

---

## Project Structure

```
App/
├── app.py                          # Core optimization engine
├── generate_benchmark_files.py     # Benchmark generation utility
├── requirements.txt                # Python dependencies
├── Procfile                        # Deployment config (Heroku)
├── Dockerfile                      # Container build
├── nixpack.toml                    # Nix/Flake config
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
│
└── Files/
    ├── Inputs.xlsx                 # Main input template
    ├── SpotPrices.xlsx             # (Optional) External spot prices
    ├── Tariffs.xlsx                # (Optional) External tariffs
    │
    └── day_ahead_benchmark/
        ├── benchmark_summary.txt   # Day-ahead statistics
        ├── benchmark_timestep_01.xlsx
        ├── benchmark_timestep_02.xlsx
        └── ...
```

### File Descriptions

#### **app.py** (Main Engine)
- **build_dataframes()**: Parse Excel inputs into DataFrames
- **extract_scalars()**: Convert DataFrames to optimization parameters
- **extract_realtime_state()**: Parse fleet state from realtime_state sheet
- **apply_disturbances()**: Apply operational disruptions
- **solvePTO()**: Build and solve Pyomo model
- **run_optimization()**: Full pipeline orchestration
- **Flask API**: Endpoints `/optimize`, `/result/<job_id>`, `/health`
- **Job Management**: Persistent JSON job store in `/tmp/jobs.json`

#### **generate_benchmark_files.py** (Benchmark Tool)
- **build_input_data()**: Load Excel templates + optional spot_prices/tariffs files
- **load_first_available_sheet_records()**: Flexible sheet loader with fallback
- **generate_benchmark_files()**: Day-ahead optimization + rolling timestep files
- **CLI Args**: 
  - `--input` (default: Files/Inputs.xlsx)
  - `--output-dir` (default: Files/day_ahead_benchmark)
  - `--spot-prices-file` (optional external spot prices)
  - `--tariffs-file` (optional external tariffs)

#### **Files/Inputs.xlsx** (Template)
Main input workbook with sheets:
- **Settings**: Global configuration (timestep_minutes, optimization_mode)
- **Buses**: Fleet vehicle data (bus_id, battery_kwh, initial_soc)
- **Chargers**: Charging assets (charger_id, charger_kw)
- **Trips**: Trip schedule (trip_id, bus_id, time_begin, time_end, energy_kwhkm, velocity)
- **Prices**: Energy prices (timestep, spot_market)
- **Realtime state**: Fleet observations (current_timestep, bus_id, soc, energy_kwh, status, delay_minutes)
- **README**: Column definitions and usage guide

#### **Files/SpotPrices.xlsx** (Optional)
External spot prices file:
- Columns: timestep, spot_market (or price)
- Used when `--spot-prices-file` passed to generate_benchmark_files.py
- Allows decoupling grid input from main template

#### **Files/Tariffs.xlsx** (Optional)
External aggregator tariffs file:
- Columns: timestep, buy_price (or buy_tariff), sell_price (or sell_tariff)
- Used when `--tariffs-file` passed to generate_benchmark_files.py
- Enables LLM/aggregator to set independent buy/sell prices

---

## Installation

### Prerequisites
- Python 3.9+
- pip or uv (package manager)

### Setup

```bash
# Clone repository
git clone https://github.com/AlienEslami/App.git
cd App

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- **pyomo**: Optimization framework
- **openpyxl**: Excel I/O
- **pandas**: Data manipulation
- **flask**: Web API
- **highspy**: HiGHS solver binding

---

## Configuration

### Settings Sheet (Inputs.xlsx)

| Field | Default | Description |
|-------|---------|-------------|
| timestep_minutes | 30 | Discretization (15, 30, or 60 min) |
| optimization_mode | day_ahead | 'day_ahead' or 'real_time' |
| v2g_enabled | true | Enable V2G discharging |

### Parameters (Hardcoded, tune in extract_scalars())
- **ch_eff**: 0.90 (charging efficiency)
- **dch_eff**: 1/0.90 ≈ 1.111 (discharging efficiency)
- **E_min**: 0.2 (minimum SOC, 20%)
- **E_max**: 1.0 (maximum SOC, 100%)
- **E_end**: 0.2 (end-of-day minimum SOC)

---

## Usage

### 1. Day-Ahead Optimization (Batch)

```bash
python generate_benchmark_files.py
```

**Output**: Creates `Files/day_ahead_benchmark/` with:
- `benchmark_summary.txt`: Statistics (T_steps, avg prices, trip durations, etc.)
- `benchmark_timestep_NN.xlsx`: Full solution for each timestep (for real-time re-optimization)

**With External Files**:
```bash
python generate_benchmark_files.py \
  --spot-prices-file Files/SpotPrices.xlsx \
  --tariffs-file Files/Tariffs.xlsx \
  --output-dir Files/custom_benchmark
```

### 2. API Server (Real-Time)

```bash
# Start Flask server
python app.py

# Server runs on http://localhost:5000
```

### 3. API Request (Real-Time Optimization)

```bash
curl -X POST http://localhost:5000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Files/Inputs.xlsx",
    "optimization_mode": "real_time",
    "current_timestep": 15,
    "price_guidance": {}
  }'
```

---

## API Reference

### POST /optimize
**Request:**
```json
{
  "input": "Files/Inputs.xlsx",
  "optimization_mode": "day_ahead",
  "current_timestep": 1,
  "price_guidance": {},
  "disturbances": [],
  "spot_prices_file": null,
  "tariffs_file": null
}
```

**Parameters:**
- `input` (string): Path to input Excel template
- `optimization_mode` (string): 'day_ahead' or 'real_time'
- `current_timestep` (int): Current step (only used in real_time mode)
- `price_guidance` (dict, optional): Legacy multiplier-based pricing (deprecated; use tariffs_file)
- `disturbances` (list, optional): Array of disruption events
  ```json
  {
    "bus_id": 1,
    "trip_id": 1,
    "disturbance_type": "late|breakdown|early_return",
    "duration_minutes": 30
  }
  ```
- `spot_prices_file` (string, optional): Path to external spot prices Excel
- `tariffs_file` (string, optional): Path to external tariffs Excel

**Response:**
```json
{
  "status": "success|error",
  "job_id": "uuid",
  "message": "Optimization queued",
  "result_url": "/result/uuid"
}
```

### GET /result/{job_id}
**Response:**
```json
{
  "status": "pending|completed|failed",
  "job_id": "uuid",
  "result": {
    "pto_daily_cost": 1234.56,
    "aggregator_revenue": 89.01,
    "total_kwh_bought": 500,
    "total_kwh_sold": 50,
    "solver_status": "optimal",
    "time_series": {
      "w_buy": [10, 20, 15, ...],
      "w_sell": [0, 5, 2, ...],
      "energy_per_bus": {...}
    }
  }
}
```

### GET /health
**Response:**
```json
{
  "status": "ok",
  "timestamp": "2026-04-16T10:30:00Z"
}
```

---

## Input File Format

### Buses Sheet

| bus_id | bus_kwh | initial_soc |
|--------|---------|------------|
| 1 | 365 | 35 |
| 2 | 365 | 35 |

- `bus_id` (int): Unique bus identifier (1-indexed)
- `bus_kwh` (float): Battery capacity in kWh
- `initial_soc` (float): Starting state-of-charge (0–100% or 0.0–1.0)

### Chargers Sheet

| charger_id | charger_kw |
|------------|-----------|
| 1 | 200 |
| 2 | 200 |

- `charger_id` (int): Unique charger identifier
- `charger_kw` (float): Charging power in kW

### Trips Sheet

| trip_id | bus_id | time_begin | time_end | energy_kwhkm | average_velocity_kmh |
|---------|--------|------------|----------|--------------|---------------------|
| 1 | 1 | 07:00 | 22:30 | 0.923 | 12 |

- `trip_id` (int): Unique trip identifier
- `bus_id` (int): Assigned bus
- `time_begin`, `time_end` (string): HH:MM format
- `energy_kwhkm` (float): Energy per km
- `average_velocity_kmh` (float): Average speed (calculates trip distance from duration)

### Prices Sheet

| timestep | spot_market |
|----------|------------|
| 1 | 0.0962 |
| 2 | 0.0962 |

- `timestep` (int): Step number (1 to T_steps)
- `spot_market` (float): Grid energy price per kWh

### Tariffs Sheet (Optional)

| timestep | buy_price | sell_price |
|----------|-----------|-----------|
| 1 | 0.0950 | 0.0750 |
| 2 | 0.0950 | 0.0750 |

- `timestep` (int): Step number
- `buy_price` (float): Aggregator's buy tariff (what operator pays)
- `sell_price` (float): Aggregator's sell tariff (V2G revenue)

### Realtime State Sheet (Optional)

| current_timestep | bus_id | current_soc | current_energy_kwh | operation_status | delay_minutes |
|------------------|--------|------------|-------------------|------------------|---------------|
| 15 | 1 | 50 | 182.5 | in_trip | 0 |
| 15 | 2 | 35 | 127.75 | charging | 10 |

- `current_timestep` (int): Current optimization step
- `bus_id` (int): Bus identifier
- `current_soc` (float): Current state-of-charge (0–100%)
- `current_energy_kwh` (float): Alternative to SOC (absolute energy in kWh)
- `operation_status` (string): 'in_trip', 'charging', 'discharging', 'idle'
- `delay_minutes` (int): Minutes delayed from scheduled start

---

## Output Metrics

### Economic Metrics
- **pto_daily_cost** (€): Total cost of operations
  ```
  = Σ S_buy[t] · w_buy[t] - Σ S_sell[t] · w_sell[t]
  ```
- **aggregator_revenue** (€): Buy margin + sell margin
  ```
  = Σ (S_buy[t] - P[t]) · w_buy[t] + Σ (S_sell[t] - P[t]) · w_sell[t]
  ```
- **total_kwh_bought** (kWh): Grid energy purchases
- **total_kwh_sold** (kWh): V2G energy sales

### Time Series
- **w_buy[t]**: Total grid power at timestep t (kW)
- **w_sell[t]**: Total V2G power at timestep t (kW)
- **energy_per_bus**: Per-bus energy trajectory over horizon

### Solver Status
- **optimal**: Solution is proven optimal
- **feasible**: Solution is feasible (time limit reached)
- **infeasible**: No solution exists (troubleshoot constraints)

---

## Examples

### Example 1: Basic Day-Ahead Optimization

```bash
python generate_benchmark_files.py
```

**Expected Output:**
```
Buy multipliers rescaled: avg was 1.0667, now 1.0240
T=48, k=8, n=8, i=8
avg_P=0.090340
avg_S_buy=0.092367 (102.2% of grid)
avg_S_sell=0.073614 (81.5% of grid)
PTO termination: optimal
Generated benchmark files in Files/day_ahead_benchmark
```

### Example 2: Real-Time Re-Optimization with Delays

**Request JSON** (`optimize_realtime.json`):
```json
{
  "input": "Files/Inputs.xlsx",
  "optimization_mode": "real_time",
  "current_timestep": 20,
  "disturbances": [
    {
      "bus_id": 3,
      "trip_id": 3,
      "disturbance_type": "late",
      "duration_minutes": 30
    }
  ]
}
```

```bash
curl -X POST http://localhost:5000/optimize \
  -H "Content-Type: application/json" \
  -d @optimize_realtime.json
```

**Response**:
```json
{
  "status": "success",
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "result_url": "/result/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

Check result:
```bash
curl http://localhost:5000/result/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

### Example 3: Decoupled Pricing (Aggregator Control)

**Scenario**: Grid price is €0.090/kWh, but aggregator wants to:
- Charge buses at €0.095/kWh (5% margin)
- Buy V2G power at €0.070/kWh (23% margin)

**Tariffs.xlsx**:
```
timestep | buy_price | sell_price
1        | 0.095     | 0.070
2        | 0.095     | 0.070
...
```

```bash
python generate_benchmark_files.py \
  --tariffs-file Files/Tariffs.xlsx
```

**Output**: Optimizer uses explicit tariffs instead of multiplier-based derivation:
```
Using explicit tariffs input (Buy Tariff / Sell Tariff), not multiplier guidance
avg_S_buy=0.095000 (105.3% of grid)
avg_S_sell=0.070000 (77.5% of grid)
```

---

## Troubleshooting

### Infeasible Optimization

**Symptom**: "PTO termination: infeasible"

**Common Causes**:
1. **Insufficient charger capacity**: Trip energy > battery capacity
   - Solution: Increase charger power or bus battery size
2. **Conflicting trip timing**: Overlapping trips exceed available chargers
   - Solution: Add chargers or adjust trip times
3. **SOC constraints too tight**: E_end (end-of-day SOC) leaves insufficient energy
   - Solution: Reduce E_end or increase charger power

**Diagnostic**:
Check trip durations vs. battery capacity:
```python
max_trip_on_full = battery_kwh / (energy_consumption_per_step)
if trip_duration > max_trip_on_full:
    # Infeasible
```

### Solver Timeout

**Symptom**: Solver hangs on large instances

**Mitigation**:
- Reduce T_steps (use 30-min or 60-min timesteps instead of 15-min)
- Simplify trip schedule (fewer overlapping trips)
- Add solver time limit in solvePTO() (lines ~830):
  ```python
  solver.options['time_limit'] = 60  # seconds
  ```

### Missing Tariffs (Fallback to Multipliers)

**Symptom**: "not multiplier guidance" message doesn't appear

**Cause**: Tariffs sheet or external tariffs_file not found

**Solution**:
- Ensure Tariffs sheet exists in Inputs.xlsx, OR
- Pass `--tariffs-file` to generate_benchmark_files.py

---

## Performance Tuning

### Solver Options (app.py, line ~825)
```python
solver = pyo.SolverFactory('highs')
# Add time limit (seconds)
solver.options['time_limit'] = 120
# Add MIP gap tolerance (%)
solver.options['mip_gap'] = 0.01
# Enable presolve
solver.options['presolve'] = 'on'
```

### Optimization Parameters (extract_scalars())
- **ch_eff**: Increase to reduce charging time
- **E_min**: Lower to allow deeper discharge
- **E_end**: Lower to reduce end-of-day waste

---

## Development & Contributing

### Running Tests

```bash
# (Add tests to be defined)
pytest tests/
```

### Code Structure

- **app.py**: Core optimization logic (900 lines)
  - Functions: build_dataframes, extract_scalars, solvePTO, run_optimization, API routes
  - Main entry: `if __name__ == '__main__': app.run()`

- **generate_benchmark_files.py**: Benchmark generation (250 lines)
  - Main entry: `generate_benchmark_files()` at module level

### Git Workflow

**Branch**: `Optimization-Jonatas` (pricing decoupling & real-time features)

**Recent Changes**:
- Decoupled S_buy/S_sell from grid spot price P
- Added real-time horizon trimming and fleet state ingestion
- Created external tariffs file support
- Added .gitignore

**To Merge to `main`**:
```bash
git checkout main
git pull origin main
git merge Optimization-Jonatas
git push origin main
```

---

## License

(Specify your license here, e.g., MIT, Apache 2.0)

---

## Contact & Support

For issues, questions, or contributions, open an issue on the GitHub repository.

**Repository**: https://github.com/AlienEslami/App

---

## Appendix: Mathematical Model

### Sets
- **K**: Buses (k = 1 to k_count)
- **N**: Chargers (n = 1 to n_count)
- **I**: Trips (i = 1 to i_count)
- **T**: Timesteps (t = 1 to T_steps)

### Parameters
- **P[t]**: Grid spot price at timestep t (€/kWh)
- **S_buy[t]**: Aggregator buy tariff at timestep t (€/kWh)
- **S_sell[t]**: Aggregator sell tariff at timestep t (€/kWh)
- **C_bat[k]**: Battery capacity of bus k (kWh)
- **E_0[k]**: Initial energy of bus k (kWh)
- **α[k,i]**: Charging rate for bus k at charger n (kWh/step)
- **β**: Discharging efficiency (≈1.111)
- **γ[k,i]**: Consumption rate for bus k on trip i (kWh/step)
- **T_start[i], T_end[i]**: Trip i start and end timesteps

### Variables
- **b[k,i,t]** ∈ {0,1}: Bus k on trip i at timestep t
- **x[k,n,t]** ≥ 0: Charging power at charger n, bus k, time t (kW)
- **y[k,n,t]** ≥ 0: V2G discharge at charger n, bus k, time t (kW)
- **e[k,t]** ≥ 0: Energy stored in bus k at timestep t (kWh)
- **w_buy[t]** ≥ 0: Total grid purchase at timestep t (kW)
- **w_sell[t]** ≥ 0: Total V2G sale at timestep t (kW)

### Objective
```
minimize: Σ_t S_buy[t] · w_buy[t] - Σ_t S_sell[t] · w_sell[t]
```

### Constraints (Simplified)
```
1. Energy balance:
   e[k,t+1] = e[k,t] + Σ_n x[k,n,t] - Σ_n β·y[k,n,t] - γ[k,i]·b[k,i,t]

2. SOC bounds:
   E_min·C_bat[k] ≤ e[k,t] ≤ E_max·C_bat[k]

3. End-of-day SOC:
   e[k,T] ≥ E_end·C_bat[k]

4. Trip continuity:
   b[k,i,t] = 1 for t ∈ [T_start[i], T_end[i]] (exactly one bus per trip)

5. Charger exclusivity:
   Σ_k x[k,n,t] ≤ α[n] (one bus per charger)

6. V2G gate (if v2g_enabled = false):
   w_sell[t] = 0, y[k,n,t] = 0 ∀ k,n,t
```

---

**Last Updated**: April 16, 2026  
**Version**: 1.0 (Optimization-Jonatas branch)
