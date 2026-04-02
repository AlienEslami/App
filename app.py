from flask import Flask, request, jsonify
import pyomo.environ as pyo
import pandas as pd

app = Flask(__name__)

def build_dataframes(input_data):
    """Convert preprocessed JSON from n8n into DataFrames matching original data['SheetName'] structure"""
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

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        input_data = request.json['input']
        data = build_dataframes(input_data)

        # --- your existing solving loop ---
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

            # convergence check (same as your notebook)
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
        # --- end solving loop ---

        # Extract and return results
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
import os
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
