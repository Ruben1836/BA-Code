"""Optimizer module for battery trading strategy.

This module defines the :class:`optimizer` used to calculate charge and
 discharge schedules on the Day Ahead Auction (DAA) and the Intraday
 Auction (IDA) electricity markets.

It reads price data from Excel spreadsheets and formulates two Pyomo
 optimisation problems:

* :meth:`step1_optimize_daa` optimises hourly trades for the day ahead
  market.
* :meth:`step2_optimize_ida` refines the schedule with quarter-hourly
  trades on the intraday market while respecting the day-ahead position.

The resulting charge/discharge profiles are returned as lists of values
 in per-unit of the battery power capability.
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo


class optimizer:
    """Battery storage optimiser."""

    def __init__(
        self,
        n_days: int,
        n_cycles: int,
        c_rate: float,
        energy_cap: float,
        eta_cha: float,
        eta_dis: float,
        min_soc: float,
        max_soc: float,
        excel_path_ida: str,
        excel_path_daa: str,
    ) -> None:
        # Battery parameters
        self.n_days = n_days
        self.n_hours = int(24 * n_days)
        self.n_cycles = n_cycles
        self.c_rate = c_rate
        self.energy_cap = energy_cap
        self.power_cap = energy_cap * c_rate
        self.eta_cha = eta_cha
        self.eta_dis = eta_dis
        self.min_soc = min_soc
        self.max_soc = max_soc

        # Load prices from Excel
        self.price_list_daa = self._load_prices(excel_path_daa)
        self.price_list_ida = self._load_prices(excel_path_ida)

        # quarter-hourly copy of day-ahead prices
        self.price_list_daa_q = np.repeat(self.price_list_daa, 4)

    @staticmethod
    def _calc_can_charge(prices: list[float], eta_cha: float, eta_dis: float) -> list[int]:
        """Return a binary list indicating profitable charging."""
        best_future = [
            max(prices[t:]) if t < len(prices) else prices[-1]
            for t in range(len(prices))
        ]
        return [
            1 if best_future[t] * eta_dis > prices[t] / eta_cha else 0
            for t in range(len(prices))
        ]

    @staticmethod
    def _load_prices(path: str) -> list[float]:
        """Return a list of prices (€/kWh) from an Excel file."""
        xls = pd.ExcelFile(path)
        df = xls.parse("1")
        price_column = df.iloc[24:, 1]
        prices: list[float] = []
        for val in price_column:
            try:
                price = float(str(val).replace(",", "."))
                prices.append(price / 1000)
            except ValueError:
                continue
        return prices

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def set_highs_solver():
        """Return pyomo solver using highs."""
        return pyo.SolverFactory("highs")

    # ------------------------------------------------------------------
    # Step 1 – Day Ahead optimisation
    # ------------------------------------------------------------------
    def step1_optimize_daa(self):
        """Optimise the day ahead (hourly) trading schedule."""
        model = pyo.ConcreteModel()

        n_hours = self.n_hours
        power_cap = self.power_cap
        eta_cha = self.eta_cha
        eta_dis = self.eta_dis
        min_soc = self.min_soc
        max_soc = self.max_soc

        # Profit trigger
        can_charge = self._calc_can_charge(self.price_list_daa, eta_cha, eta_dis)

        # Sets
        model.T = pyo.RangeSet(1, n_hours)
        model.T_plus_1 = pyo.RangeSet(1, n_hours + 1)

        # Variables
        model.soc = pyo.Var(model.T_plus_1, domain=pyo.Reals)
        model.cha_daa = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, 1))
        model.dis_daa = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, 1))
        model.charge_flag = pyo.Var(model.T, domain=pyo.Binary)
        model.discharge_flag = pyo.Var(model.T, domain=pyo.Binary)

        # Profit trigger param
        model.can_charge = pyo.Param(
            model.T,
            initialize={t + 1: can_charge[t] for t in range(n_hours)},
            within=pyo.Binary,
        )

        # Logical constraints
        model.cha_flag_constraint = pyo.Constraint(model.T, rule=lambda m, t: m.cha_daa[t] <= m.charge_flag[t])
        model.dis_flag_constraint = pyo.Constraint(model.T, rule=lambda m, t: m.dis_daa[t] <= m.discharge_flag[t])
        model.no_simultaneous = pyo.Constraint(model.T, rule=lambda m, t: m.charge_flag[t] + m.discharge_flag[t] <= 1)

        model.charge_profit_trigger = pyo.Constraint(model.T, rule=lambda m, t: m.cha_daa[t] <= m.can_charge[t])

        # SoC constraints
        model.set_maximum_soc = pyo.Constraint(model.T_plus_1, rule=lambda m, t: m.soc[t] <= max_soc)
        model.set_minimum_soc = pyo.Constraint(model.T_plus_1, rule=lambda m, t: m.soc[t] >= min_soc)
        model.set_first_soc_to_min = pyo.Constraint(rule=lambda m: m.soc[1] == min_soc)
        model.set_last_soc_to_min = pyo.Constraint(rule=lambda m: m.soc[n_hours + 1] == min_soc)

        volume_limit = (self.max_soc - self.min_soc) * self.n_cycles * (n_hours / 24)
        model.charge_cycle_limit = pyo.Constraint(
            rule=lambda m: sum(m.cha_daa[t] * power_cap * eta_cha for t in m.T) <= volume_limit
        )
        model.discharge_cycle_limit = pyo.Constraint(
            rule=lambda m: sum(m.dis_daa[t] * power_cap for t in m.T) <= volume_limit
        )

        # SoC dynamics
        def soc_step(m, t):
            return m.soc[t + 1] == m.soc[t] + eta_cha * power_cap * m.cha_daa[t] - (1 / eta_dis) * power_cap * m.dis_daa[t]

        model.soc_step_constraint = pyo.Constraint(model.T, rule=soc_step)

        # Objective
        model.obj = pyo.Objective(
            expr=sum(power_cap * self.price_list_daa[t - 1] * (model.dis_daa[t] - model.cha_daa[t]) for t in model.T),
            sense=pyo.maximize,
        )

        solver = self.set_highs_solver()
        solver.solve(model)

        soc = [model.soc[t].value for t in range(1, n_hours + 1)]
        cha = [model.cha_daa[t].value for t in range(1, n_hours + 1)]
        dis = [model.dis_daa[t].value for t in range(1, n_hours + 1)]

        profit = sum(
            power_cap * p * (d - c)
            for c, d, p in zip(cha, dis, self.price_list_daa)
        )

        chaq = np.repeat(cha, 4)
        disq = np.repeat(dis, 4)
        socq = np.repeat(soc, 4)
        cha_real = [float(x * power_cap / 4 * eta_cha) for x in chaq]
        dis_real = [float(x * power_cap / 4 / eta_dis) for x in disq]

        return soc, socq, chaq, disq, profit, cha, dis, cha_real, dis_real

    # ------------------------------------------------------------------
    # Step 2 – Intraday optimisation
    # ------------------------------------------------------------------
    def step2_optimize_ida(
        self,
        n_hours: int,
        energy_cap: float,
        power_cap: float,
        eta_cha: float,
        eta_dis: float,
        step1_cha_daa: list,
        step1_dis_daa: list,
    ):
        """Refine schedule quarter-hourly for the intraday market."""
        model = pyo.ConcreteModel()

        min_soc = self.min_soc
        max_soc = self.max_soc
        N = 4 * n_hours

        model.Q = pyo.RangeSet(1, N)
        model.Q_plus_1 = pyo.RangeSet(1, N + 1)

        can_close_buy = [1 if self.price_list_ida[i] < self.price_list_daa_q[i] else 0 for i in range(N)]
        can_close_sell = [1 if self.price_list_ida[i] > self.price_list_daa_q[i] else 0 for i in range(N)]
        model.can_close_buy = pyo.Param(model.Q, initialize={i + 1: can_close_buy[i] for i in range(N)}, within=pyo.Binary)
        model.can_close_sell = pyo.Param(
            model.Q, initialize={i + 1: can_close_sell[i] for i in range(N)}, within=pyo.Binary
        )

        # Profit trigger similar as step1
        can_charge = self._calc_can_charge(self.price_list_ida, eta_cha, eta_dis)
        model.can_charge = pyo.Param(model.Q, initialize={q: can_charge[q - 1] for q in range(1, N + 1)}, within=pyo.Binary)

        # Variables
        model.soc = pyo.Var(model.Q_plus_1, domain=pyo.Reals)
        model.cha_ida = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))
        model.dis_ida = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))
        model.cha_ida_close = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))
        model.dis_ida_close = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))

        model.charge_profit_trigger = pyo.Constraint(model.Q, rule=lambda m, t: m.cha_ida[t] <= m.can_charge[t])

        model.set_max_soc = pyo.Constraint(model.Q_plus_1, rule=lambda m, q: m.soc[q] <= max_soc)
        model.set_min_soc = pyo.Constraint(model.Q_plus_1, rule=lambda m, q: m.soc[q] >= min_soc)
        model.set_soc_start = pyo.Constraint(rule=lambda m: m.soc[1] == min_soc)
        model.set_soc_end = pyo.Constraint(rule=lambda m: m.soc[N + 1] == min_soc)

        def soc_step(m, q):
            return (
                m.soc[q + 1]
                == m.soc[q]
                + power_cap / 4
                * (
                    eta_cha * m.cha_ida[q]
                    - (1 / eta_dis) * m.dis_ida[q]
                    + m.cha_ida_close[q]
                    - m.dis_ida_close[q]
                    + step1_cha_daa[q - 1]
                    - step1_dis_daa[q - 1]
                )
            )

        model.soc_dynamics = pyo.Constraint(model.Q, rule=soc_step)

        volume_limit = energy_cap * n_hours / 24 * self.n_cycles
        model.charge_cycle_limit = pyo.Constraint(
            rule=lambda m: sum(m.cha_ida[q] for q in m.Q) * power_cap / 4 + sum(step1_cha_daa) * power_cap / 4
            <= volume_limit
        )
        model.discharge_cycle_limit = pyo.Constraint(
            rule=lambda m: sum(m.dis_ida[q] for q in m.Q) * power_cap / 4 + sum(step1_dis_daa) * power_cap / 4
            <= volume_limit
        )

        model.cha_close_logic = pyo.Constraint(
            model.Q, rule=lambda m, q: m.cha_ida_close[q] <= step1_dis_daa[q - 1] * m.can_close_buy[q]
        )
        model.dis_close_logic = pyo.Constraint(
            model.Q, rule=lambda m, q: m.dis_ida_close[q] <= step1_cha_daa[q - 1] * m.can_close_sell[q]
        )

        model.charge_rate_limit = pyo.Constraint(model.Q, rule=lambda m, q: m.cha_ida[q] + step1_cha_daa[q - 1] <= 1)
        model.discharge_rate_limit = pyo.Constraint(
            model.Q, rule=lambda m, q: m.dis_ida[q] + step1_dis_daa[q - 1] <= 1
        )

        # Objective: combine DAA and IDA trades
        model.obj = pyo.Objective(
            expr=sum(
                self.price_list_daa_q[q - 1] * power_cap / 4 * (step1_dis_daa[q - 1] - step1_cha_daa[q - 1])
                + self.price_list_ida[q - 1]
                * power_cap
                / 4
                * (
                    model.dis_ida[q]
                    + model.dis_ida_close[q]
                    - model.cha_ida[q]
                    - model.cha_ida_close[q]
                )
                for q in model.Q
            ),
            sense=pyo.maximize,
        )

        solver = self.set_highs_solver()
        solver.solve(model)

        soc_ida = [model.soc[q].value for q in range(1, N + 1)]
        cha_ida = [model.cha_ida[q].value for q in model.Q]
        dis_ida = [model.dis_ida[q].value for q in model.Q]
        cha_ida_close = [model.cha_ida_close[q].value for q in model.Q]
        dis_ida_close = [model.dis_ida_close[q].value for q in model.Q]

        profit_ida = sum(
            self.price_list_daa_q[q - 1]
            * (
                eta_dis * (power_cap / 4) * step1_dis_daa[q - 1]
                - (power_cap / 4) / eta_cha * step1_cha_daa[q - 1]
            )
            + self.price_list_ida[q - 1]
            * (
                eta_dis * (power_cap / 4) * (dis_ida[q - 1] + dis_ida_close[q - 1])
                - (power_cap / 4) / eta_cha * (cha_ida[q - 1] + cha_ida_close[q - 1])
            )
            for q in range(1, N + 1)
        )

        cha_combined = np.asarray(step1_cha_daa) - np.asarray(dis_ida_close) + np.asarray(cha_ida)
        dis_combined = np.asarray(step1_dis_daa) - np.asarray(cha_ida_close) + np.asarray(dis_ida)

        return (
            soc_ida,
            cha_ida,
            dis_ida,
            cha_ida_close,
            dis_ida_close,
            profit_ida,
            cha_combined.tolist(),
            dis_combined.tolist(),
        )
