# src/model.py
from __future__ import annotations

import pandas as pd

from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum, LpStatus,
    LpInteger, LpContinuous, LpBinary, PULP_CBC_CMD, value
)

from .utils import (
    trim, normalize_cols, to_float_money, to_rate, is_intish, pick_col
)


def solve_mps(
    data: dict[str, pd.DataFrame],
    *,
    inv0_fg: float = 0.0,
    inv0_l1: float = 0.0,
    hold_rate_annual: float = 0.20,
    eps: float = 1e-9,
    time_limit_sec: int = 180,
    gap_rel: float = 0.01,
    solver_msg: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Core MILP solver.
    Input data expects these keys:
      BOM_FG_L1, BOM_L1_L2, DEMAND_FG, INV_START, CAP_FG, ITEM_MASTER, COST_BUY_MONTHLY

    Returns:
      FG_PLAN, L1_PLAN, L2_PLAN, COST_SUMMARY, COST_DETAIL
    """

    HOLD_RATE_MONTHLY = hold_rate_annual / 12.0

    # ---- Load + normalize ----
    BOM_FG_L1 = normalize_cols(data["BOM_FG_L1"])
    BOM_L1_L2 = normalize_cols(data["BOM_L1_L2"])
    DEMAND = normalize_cols(data["DEMAND_FG"])
    INV_START = normalize_cols(data["INV_START"])
    CAP_FG = normalize_cols(data["CAP_FG"])
    ITEM_MASTER = normalize_cols(data["ITEM_MASTER"])
    COST = normalize_cols(data["COST_BUY_MONTHLY"])

    # ---- Clean keys ----
    if "month" not in DEMAND.columns and "Month" in DEMAND.columns:
        DEMAND = DEMAND.rename(columns={"Month": "month"})
    DEMAND["month"] = trim(DEMAND["month"])
    DEMAND["fg_code"] = trim(DEMAND["fg_code"])

    INV_START["item_code"] = trim(INV_START["item_code"])
    CAP_FG["fg_code"] = trim(CAP_FG["fg_code"])
    ITEM_MASTER["item_code"] = trim(ITEM_MASTER["item_code"])

    COST["month"] = trim(COST["month"])
    COST["item_code"] = trim(COST["item_code"])

    # ---- BOM standardized columns (robust) ----
    fg_col = pick_col(BOM_FG_L1, ["fg_code", "fg_code (Product Item Code)"])
    l1_col = pick_col(BOM_FG_L1, ["l1_code", "l1_code (Assembly Code)"])
    q_fg_l1_col = pick_col(BOM_FG_L1, ["qty_per_fg", "qty_per_fg (Quantity)", "qty"])

    BOM_FG_L1["fg_code"] = trim(BOM_FG_L1[fg_col])
    BOM_FG_L1["l1_code"] = trim(BOM_FG_L1[l1_col])
    BOM_FG_L1["qty"] = pd.to_numeric(BOM_FG_L1[q_fg_l1_col], errors="coerce").fillna(0.0)

    l1p_col = pick_col(BOM_L1_L2, ["l1_code", "l1_code (Parent Code)"])
    l2_col = pick_col(BOM_L1_L2, ["l2_code", "l2_code (Assembly Code)"])
    q_l1_l2_col = pick_col(BOM_L1_L2, ["qty_per_l1", "qty_per_l1 (Quantity)", "qty"])

    BOM_L1_L2["l1_code"] = trim(BOM_L1_L2[l1p_col])
    BOM_L1_L2["l2_code"] = trim(BOM_L1_L2[l2_col])
    BOM_L1_L2["qty"] = pd.to_numeric(BOM_L1_L2[q_l1_l2_col], errors="coerce").fillna(0.0)

    # ---- Enforce integer BOM ----
    bad_fg_l1 = BOM_FG_L1.loc[~BOM_FG_L1["qty"].apply(is_intish)]
    bad_l1_l2 = BOM_L1_L2.loc[~BOM_L1_L2["qty"].apply(is_intish)]
    if len(bad_fg_l1) > 0 or len(bad_l1_l2) > 0:
        raise ValueError(
            "Integer consumption requested, but BOM has non-integer qty.\n"
            "Scale BOM (e.g., multiply by 10/100) or change units so BOM qty become integers.\n"
            f"Non-integer rows BOM_FG_L1: {len(bad_fg_l1)}, BOM_L1_L2: {len(bad_l1_l2)}"
        )

    BOM_FG_L1["qty_int"] = BOM_FG_L1["qty"].round().astype(int)
    BOM_L1_L2["qty_int"] = BOM_L1_L2["qty"].round().astype(int)

    # ---- Sets ----
    months = sorted(DEMAND["month"].unique().tolist())
    fgs = sorted(DEMAND["fg_code"].unique().tolist())
    l1s = sorted(BOM_FG_L1["l1_code"].unique().tolist())
    l2s = sorted(BOM_L1_L2["l2_code"].unique().tolist())

    # ---- Parameters ----
    demand = DEMAND.set_index(["month", "fg_code"])["demand_qty"].to_dict()
    cap_fg = CAP_FG.set_index("fg_code")["cap_fg_per_month"].to_dict()

    # MOQ detection
    moq_col_candidates = ["moq", "MOQ", "MOQ_units", "moq_units", "minimum_order_qty", "Minimum Order Quantity"]
    moq_col = next((c for c in moq_col_candidates if c in ITEM_MASTER.columns), None)
    if moq_col is None:
        ITEM_MASTER["moq"] = 0.0
        moq_col = "moq"
    item_moq = ITEM_MASTER.set_index("item_code")[moq_col].map(to_float_money).to_dict()

    units_per_container_master = ITEM_MASTER.set_index("item_code")["units_per_container"].map(to_float_money).to_dict()
    unit_mat_cost_master = ITEM_MASTER.set_index("item_code")["unit_material_cost"].map(to_float_money).to_dict()

    # COST table params per month
    cost_key = ["month", "item_code"]
    COST["unit_material_cost"] = COST["unit_material_cost"].apply(to_float_money)
    COST["units_per_container"] = COST["units_per_container"].apply(to_float_money)
    COST["container_cost"] = COST["container_cost"].apply(to_float_money)
    COST["insurance"] = COST["insurance"].apply(to_rate)
    COST["import_duty"] = COST["import_duty"].apply(to_rate)
    COST["VAT"] = COST["VAT"].apply(to_rate)

    cost_unit_mat = COST.set_index(cost_key)["unit_material_cost"].to_dict()
    cost_upc = COST.set_index(cost_key)["units_per_container"].to_dict()
    cost_container = COST.set_index(cost_key)["container_cost"].to_dict()
    cost_ins = COST.set_index(cost_key)["insurance"].to_dict()
    cost_duty = COST.set_index(cost_key)["import_duty"].to_dict()
    cost_vat = COST.set_index(cost_key)["VAT"].to_dict()

    # BOM dicts (integer)
    bom_fg_l1_int = BOM_FG_L1.set_index(["fg_code", "l1_code"])["qty_int"].to_dict()
    bom_l1_l2_int = BOM_L1_L2.set_index(["l1_code", "l2_code"])["qty_int"].to_dict()

    # Starting inventory: FG & L1 start = given inv0_*; L2 from INV_START
    inv0_raw = INV_START.set_index("item_code")["on_hand_qty"].map(to_float_money).to_dict()
    inv0_l2 = {l2: float(inv0_raw.get(l2, 0.0)) for l2 in l2s}

    # Holding costs
    hold_cost_item = {it: HOLD_RATE_MONTHLY * float(v) for it, v in unit_mat_cost_master.items()}

    # FG material cost approx = sum(BOM * L1 material cost)
    fg_mat_cost = {}
    for fg in fgs:
        total = 0.0
        for l1 in l1s:
            q = bom_fg_l1_int.get((fg, l1), 0)
            if q:
                total += float(q) * float(unit_mat_cost_master.get(l1, 0.0))
        fg_mat_cost[fg] = total
    hold_cost_fg = {fg: HOLD_RATE_MONTHLY * fg_mat_cost.get(fg, 0.0) for fg in fgs}

    # Helpers
    def upc(month, item):
        v = cost_upc.get((month, item), None)
        if v is None or float(v) <= 0:
            v = units_per_container_master.get(item, 0.0)
        return float(v) if float(v) > 0 else 1.0

    def unit_mat(month, item):
        v = cost_unit_mat.get((month, item), None)
        if v is None:
            v = unit_mat_cost_master.get(item, 0.0)
        return float(v)

    def cont_cost(month, item):
        return float(cost_container.get((month, item), 0.0))

    def tax_mult(month, item):
        ins = float(cost_ins.get((month, item), 0.0))
        duty = float(cost_duty.get((month, item), 0.0))
        vat = float(cost_vat.get((month, item), 0.0))
        return (1.0 + ins) * (1.0 + duty) * (1.0 + vat)

    # -------------------------
    # Build MILP
    # -------------------------
    prob = LpProblem("MPS_containerized_MILP_IntegerConsumption", LpMinimize)

    prod_fg = LpVariable.dicts("prod_fg", (fgs, months), lowBound=0, cat=LpContinuous)
    inv_fg  = LpVariable.dicts("inv_fg",  (fgs, months), lowBound=0, cat=LpContinuous)

    make_l1 = LpVariable.dicts("make_l1", (l1s, months), lowBound=0, cat=LpInteger)
    inv_l1  = LpVariable.dicts("inv_l1",  (l1s, months), lowBound=0, cat=LpContinuous)

    buy_l1  = LpVariable.dicts("buy_l1",  (l1s, months), lowBound=0, cat=LpInteger)
    cont_l1 = LpVariable.dicts("cont_l1", (l1s, months), lowBound=0, cat=LpInteger)

    buy_l2  = LpVariable.dicts("buy_l2",  (l2s, months), lowBound=0, cat=LpInteger)
    cont_l2 = LpVariable.dicts("cont_l2", (l2s, months), lowBound=0, cat=LpInteger)
    inv_l2  = LpVariable.dicts("inv_l2",  (l2s, months), lowBound=0, cat=LpContinuous)

    cons_l1 = LpVariable.dicts("cons_l1", (l1s, months), lowBound=0, cat=LpInteger)
    cons_l2 = LpVariable.dicts("cons_l2", (l2s, months), lowBound=0, cat=LpInteger)

    order_l1 = {l1: LpVariable.dicts(f"order_l1_{l1}", months, cat=LpBinary)
                for l1 in l1s if float(item_moq.get(l1, 0.0)) > 0}
    order_l2 = {l2: LpVariable.dicts(f"order_l2_{l2}", months, cat=LpBinary)
                for l2 in l2s if float(item_moq.get(l2, 0.0)) > 0}

    # Constraints: FG balance + capacity
    for fg in fgs:
        for ti, t in enumerate(months):
            dem = float(demand.get((t, fg), 0.0))
            if not abs(dem - round(dem)) <= eps:
                raise ValueError(f"Demand is not integer for (month={t}, fg={fg}): {dem}.")
            dem = int(round(dem))

            if ti == 0:
                prob += inv_fg[fg][t] == inv0_fg + prod_fg[fg][t] - dem
            else:
                tprev = months[ti - 1]
                prob += inv_fg[fg][t] == inv_fg[fg][tprev] + prod_fg[fg][t] - dem

    for fg in fgs:
        cap = float(cap_fg.get(fg, 0.0))
        for t in months:
            prob += prod_fg[fg][t] <= cap

    # Consumption definitions
    for l1 in l1s:
        for t in months:
            prob += cons_l1[l1][t] == lpSum(bom_fg_l1_int.get((fg, l1), 0) * prod_fg[fg][t] for fg in fgs)

    for l2 in l2s:
        for t in months:
            prob += cons_l2[l2][t] == lpSum(bom_l1_l2_int.get((l1, l2), 0) * make_l1[l1][t] for l1 in l1s)

    # L1/L2 balances
    for l1 in l1s:
        for ti, t in enumerate(months):
            if ti == 0:
                prob += inv_l1[l1][t] == inv0_l1 + buy_l1[l1][t] + make_l1[l1][t] - cons_l1[l1][t]
            else:
                tprev = months[ti - 1]
                prob += inv_l1[l1][t] == inv_l1[l1][tprev] + buy_l1[l1][t] + make_l1[l1][t] - cons_l1[l1][t]

    for l2 in l2s:
        for ti, t in enumerate(months):
            if ti == 0:
                prob += inv_l2[l2][t] == float(inv0_l2.get(l2, 0.0)) + buy_l2[l2][t] - cons_l2[l2][t]
            else:
                tprev = months[ti - 1]
                prob += inv_l2[l2][t] == inv_l2[l2][tprev] + buy_l2[l2][t] - cons_l2[l2][t]

    # Container link
    for l1 in l1s:
        for t in months:
            prob += buy_l1[l1][t] <= cont_l1[l1][t] * upc(t, l1)

    for l2 in l2s:
        for t in months:
            prob += buy_l2[l2][t] <= cont_l2[l2][t] * upc(t, l2)

    # MOQ (binary)
    max_prod = {fg: float(cap_fg.get(fg, 0.0)) for fg in fgs}
    max_l1_need_per_month = {
        l1: sum(float(bom_fg_l1_int.get((fg, l1), 0)) * max_prod[fg] for fg in fgs)
        for l1 in l1s
    }
    BIGM_L1 = {l1: max_l1_need_per_month[l1] * len(months) + 10_000 for l1 in l1s}

    max_l2_need_per_month = {
        l2: sum(float(bom_l1_l2_int.get((l1, l2), 0)) * max_l1_need_per_month[l1] for l1 in l1s)
        for l2 in l2s
    }
    BIGM_L2 = {l2: max_l2_need_per_month[l2] * len(months) + 10_000 for l2 in l2s}

    for l1 in l1s:
        moq = float(item_moq.get(l1, 0.0))
        if moq > 0:
            for t in months:
                y = order_l1[l1][t]
                prob += buy_l1[l1][t] >= moq * y
                prob += buy_l1[l1][t] <= BIGM_L1[l1] * y

    for l2 in l2s:
        moq = float(item_moq.get(l2, 0.0))
        if moq > 0:
            for t in months:
                y = order_l2[l2][t]
                prob += buy_l2[l2][t] >= moq * y
                prob += buy_l2[l2][t] <= BIGM_L2[l2] * y

    # Objective
    purchase_terms = []
    holding_terms = []

    for l1 in l1s:
        for t in months:
            m = unit_mat(t, l1)
            cc = cont_cost(t, l1)
            mult = tax_mult(t, l1)
            purchase_terms.append(mult * (m * buy_l1[l1][t] + cc * cont_l1[l1][t]))
            holding_terms.append(hold_cost_item.get(l1, 0.0) * inv_l1[l1][t])

    for l2 in l2s:
        for t in months:
            m = unit_mat(t, l2)
            cc = cont_cost(t, l2)
            mult = tax_mult(t, l2)
            purchase_terms.append(mult * (m * buy_l2[l2][t] + cc * cont_l2[l2][t]))
            holding_terms.append(hold_cost_item.get(l2, 0.0) * inv_l2[l2][t])

    for fg in fgs:
        for t in months:
            holding_terms.append(hold_cost_fg.get(fg, 0.0) * inv_fg[fg][t])

    prob += lpSum(purchase_terms) + lpSum(holding_terms)

    # Solve
    solver = PULP_CBC_CMD(msg=solver_msg, timeLimit=time_limit_sec, gapRel=gap_rel)
    prob.solve(solver)

    if LpStatus[prob.status] not in ("Optimal", "Feasible"):
        raise RuntimeError(f"Solver status: {LpStatus[prob.status]}")

    # Extract helper
    def v(x) -> float:
        val = value(x)
        return 0.0 if val is None else float(val)

    # FG_PLAN
    fg_rows = []
    for fg in fgs:
        prev = inv0_fg
        for t in months:
            prod = int(round(v(prod_fg[fg][t])))
            endi = max(v(inv_fg[fg][t]), 0.0)
            dem = int(round(float(demand.get((t, fg), 0.0))))
            fg_rows.append((t, fg, prev, prod, dem, endi))
            prev = endi
    fg_plan = pd.DataFrame(fg_rows, columns=[
        "month", "fg_code", "opening_inv_fg", "produce_qty", "demand_qty", "ending_inv_fg"
    ])

    # L1_PLAN
    l1_rows = []
    for l1 in l1s:
        prev = inv0_l1
        for t in months:
            buyq = int(round(v(buy_l1[l1][t])))
            cont = int(round(v(cont_l1[l1][t])))
            cap = upc(t, l1)
            unused = cont * cap - buyq
            makeq = int(round(v(make_l1[l1][t])))
            cons = int(round(v(cons_l1[l1][t])))
            endi = max(v(inv_l1[l1][t]), 0.0)
            l1_rows.append((t, l1, prev, buyq, cont, unused, makeq, cons, endi))
            prev = endi
    l1_plan = pd.DataFrame(l1_rows, columns=[
        "month", "l1_code", "opening_inv_l1", "buy_qty", "containers_booked",
        "unused_container_capacity_qty", "make_qty", "consumption_l1", "ending_inv_l1"
    ])

    # L2_PLAN
    l2_rows = []
    for l2 in l2s:
        prev = float(inv0_l2.get(l2, 0.0))
        for t in months:
            buyq = int(round(v(buy_l2[l2][t])))
            cont = int(round(v(cont_l2[l2][t])))
            cap = upc(t, l2)
            unused = cont * cap - buyq
            cons = int(round(v(cons_l2[l2][t])))
            endi = max(v(inv_l2[l2][t]), 0.0)
            l2_rows.append((t, l2, prev, buyq, cont, unused, cons, endi))
            prev = endi
    l2_plan = pd.DataFrame(l2_rows, columns=[
        "month", "l2_code", "opening_inv_l2", "buy_qty", "containers_booked",
        "unused_container_capacity_qty", "consumption_l2", "ending_inv_l2"
    ])

    # COST_DETAIL + SUMMARY
    cost_rows = []

    def add_cost(level: str, month: str, item: str, buyq: int, cont: int, ending_inv: float):
        m = unit_mat(month, item)
        cc = cont_cost(month, item)
        mult = tax_mult(month, item)

        cargo = m * buyq
        logistics = cc * cont
        purchase_total = (cargo + logistics) * mult

        ending_inv = max(float(ending_inv), 0.0)
        holding = ending_inv * (HOLD_RATE_MONTHLY * m)

        cost_rows.append([
            level, month, item, buyq, cont,
            cargo, logistics, mult, purchase_total, holding, purchase_total + holding
        ])

    for _, r in l1_plan.iterrows():
        add_cost("L1", r["month"], r["l1_code"], int(r["buy_qty"]), int(r["containers_booked"]), r["ending_inv_l1"])
    for _, r in l2_plan.iterrows():
        add_cost("L2", r["month"], r["l2_code"], int(r["buy_qty"]), int(r["containers_booked"]), r["ending_inv_l2"])

    cost_detail = pd.DataFrame(cost_rows, columns=[
        "level", "month", "item_code", "buy_qty", "containers_booked",
        "cargo_value", "logistics_cost", "tax_multiplier",
        "purchase_total_cost", "holding_cost", "grand_total_cost"
    ])

    cost_summary = cost_detail.groupby("month", as_index=False).agg(
        cargo_value=("cargo_value", "sum"),
        logistics_cost=("logistics_cost", "sum"),
        purchase_total_cost=("purchase_total_cost", "sum"),
        holding_cost=("holding_cost", "sum"),
        grand_total_cost=("grand_total_cost", "sum"),
        containers_booked=("containers_booked", "sum"),
    )

    return {
        "FG_PLAN": fg_plan,
        "L1_PLAN": l1_plan,
        "L2_PLAN": l2_plan,
        "COST_SUMMARY": cost_summary,
        "COST_DETAIL": cost_detail,
    }
