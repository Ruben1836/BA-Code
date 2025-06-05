"""Example script running the battery optimisation workflow."""

from pathlib import Path

import numpy as np

from V9_optimizer import optimizer
from visualizer import (
    analyze_step1_results,
    calculate_real_cycles,
    combine_charge_discharge,
    export_all_time_series_with_charts,
    export_full_results_to_excel_premium,
    auswertung_transaktionen_stuendlich,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_DAYS = 1
N_CYCLES = 4
C_RATE = 1.0
ENERGY_CAP = 10000  # kWh
MIN_SOC = 0
MAX_SOC = 10000
ETA_CHA = 1.0
ETA_DIS = 1.0

EXCEL_PATH_DAA = Path("path_to_day_ahead_prices.xlsx")
EXCEL_PATH_IDA = Path("path_to_intraday_prices.xlsx")

# ---------------------------------------------------------------------------
# Run optimisation
# ---------------------------------------------------------------------------


def main() -> None:
    opt = optimizer(
        n_days=N_DAYS,
        n_cycles=N_CYCLES,
        c_rate=C_RATE,
        energy_cap=ENERGY_CAP,
        eta_cha=ETA_CHA,
        eta_dis=ETA_DIS,
        min_soc=MIN_SOC,
        max_soc=MAX_SOC,
        excel_path_ida=str(EXCEL_PATH_IDA),
        excel_path_daa=str(EXCEL_PATH_DAA),
    )

    (
        soc_daa_h,
        soc_q,
        cha_daa_quarters,
        dis_daa_quarters,
        profit_daa,
        cha_daa_h,
        dis_daa_h,
        cha_daa_q_real,
        dis_daa_q_real,
    ) = opt.step1_optimize_daa()

    (
        step2_soc_ida,
        cha_ida,
        dis_ida,
        cha_ida_close,
        dis_ida_close,
        profit_ida,
        combined_cha,
        combined_dis,
    ) = opt.step2_optimize_ida(
        N_DAYS * 24,
        ENERGY_CAP,
        ENERGY_CAP * C_RATE,
        ETA_CHA,
        ETA_DIS,
        cha_daa_quarters,
        dis_daa_quarters,
    )

    print(f"Profit DAA: {profit_daa:.2f} €")
    print(f"Profit IDA: {profit_ida:.2f} €")

    # -----------------------------------------------------------------------
    # Visualisation & export
    # -----------------------------------------------------------------------
    analyze_step1_results(
        soc_daa_hours=soc_daa_h,
        cha_daa_quarters=cha_daa_quarters,
        dis_daa_quarters=dis_daa_quarters,
        price_list_hourly=opt.price_list_daa,
        power_cap=opt.power_cap,
        n_hours=opt.n_hours,
    )

    calculate_real_cycles(
        cha_quarters=cha_daa_quarters,
        dis_quarters=dis_daa_quarters,
        power_cap=opt.power_cap,
        n_hours=opt.n_hours,
        min_soc=MIN_SOC,
        max_soc=MAX_SOC,
        allowed_cycles=N_CYCLES,
        eta_cha=ETA_CHA,
        eta_dis=ETA_DIS,
    )

    energy_profile = combine_charge_discharge(
        cha_quarters=cha_daa_quarters,
        dis_quarters=dis_daa_quarters,
        power_cap=opt.power_cap,
    )

    auswertung_transaktionen_stuendlich(
        cha_daa_h,
        dis_daa_h,
        opt.price_list_daa,
        opt.power_cap,
    )

    export_all_time_series_with_charts(
        soc_q,
        cha_daa_quarters,
        dis_daa_quarters,
        cha_daa_q_real,
        dis_daa_q_real,
        step2_soc_ida,
        cha_ida,
        dis_ida,
        cha_ida_close,
        dis_ida_close,
        combined_cha,
        combined_dis,
        opt.price_list_daa,
        opt.price_list_ida,
        folder_path="ergebnisse",
    )

    export_full_results_to_excel_premium(
        soc_hours=soc_daa_h,
        cha_quarters=cha_daa_quarters,
        dis_quarters=dis_daa_quarters,
        energy_profile=energy_profile,
        profit=profit_ida,
        n_days=N_DAYS,
        power_cap=opt.power_cap,
        energy_cap=ENERGY_CAP,
        min_soc=MIN_SOC,
        max_soc=MAX_SOC,
        eta_cha=ETA_CHA,
        eta_dis=ETA_DIS,
        n_cycles=N_CYCLES,
        cha_daa_h=cha_daa_h,
        dis_daa_h=dis_daa_h,
        folder_path="ergebnisse",
        price_list_hourly=opt.price_list_daa,
        c_rate=C_RATE,
        price_list_quarter=np.repeat(opt.price_list_daa, 4).tolist(),
    )


if __name__ == "__main__":
    main()
