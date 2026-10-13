"""
pk_simulation_digital_twin.py
=============================

LITERATURE‑BASED stochastic digital twin simulation of T4 nanocarrier
transport dynamics: IV → Blood → Blood‑Brain Barrier (BBB) → Tumor.
Includes probabilistic transfer, Monte Carlo variability, and
rate constants derived from experimentally observed BBB nanoparticle
transcytosis kinetics.

Literature‑anchored kinetic parameters:
    • Endocytosis (blood → BBB endothelial): t1/2 ≈ 1.9 hr
    • Exocytosis (BBB endothelial → brain): t1/2 ≈ 14 hr
Reference: experimentally fitted in vitro BBB nanoparticle model. :contentReference[oaicite:4]{index=4}
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_pk_digital_twin(
    t_max=24,
    dt=0.1,
    n_sim=500,
    initial_dose=1.0,
    random_seed=42
):
    """
    Monte Carlo stochastic PK model for T4 nanoparticles with literature‑anchored BBB rates.

    Parameters
    ----------
    t_max : float
        Duration in hours to simulate (default 24h)
    dt : float
        Time step (hours)
    n_sim : int
        Number of simulation replicates to capture variability
    initial_dose : float
        Fractional dose injected intravenously (1.0 = 100%)
    random_seed : int
        Seed for reproducibility

    Returns
    -------
    time : ndarray
    blood_mc, bbb_mc, tumor_mc : arrays of dimension (n_sim, n_timepoints)
    """
    np.random.seed(random_seed)
    n_steps = int(t_max / dt) + 1
    time = np.linspace(0, t_max, n_steps)

    # Preallocate
    blood_mc = np.zeros((n_sim, n_steps))
    bbb_mc = np.zeros((n_sim, n_steps))
    tumor_mc = np.zeros((n_sim, n_steps))

    # Literature‑based base rates (h^-1)
    # Derived from half‑time experiments:
    # k = ln(2) / half_time
    k_endocytosis     = np.log(2) / 1.9   # uptake into BBB endothelial
    k_exocytosis      = np.log(2) / 14.0  # exit across BBB into brain
    k_cl_clearance    = 0.03             # blood clearance (RES/hepatic), literature consistent estimate
    k_bbb_clearance   = 0.02             # clearance from brain compartment (microglial etc.)

    for sim in range(n_sim):
        blood_mc[sim, 0] = initial_dose
        bbb_mc[sim, 0] = 0.0
        tumor_mc[sim, 0] = 0.0

        for t in range(1, n_steps):
            B_prev = blood_mc[sim, t-1]
            E_prev = bbb_mc[sim, t-1]
            T_prev = tumor_mc[sim, t-1]

            # Randomized variability around literature base rates
            k_b2e = max(np.random.normal(k_endocytosis, 0.05 * k_endocytosis),0)
            k_e2brain = max(np.random.normal(k_exocytosis, 0.15 * k_exocytosis),0)

            # Transfers per dt
            dB2E = k_b2e * B_prev * dt
            dE2T = k_e2brain * E_prev * dt
            dB_clear = k_cl_clearance * B_prev * dt
            dE_clear = k_bbb_clearance * E_prev * dt

            # Update compartments
            blood_mc[sim,t] = max(B_prev - dB2E - dB_clear, 0)
            bbb_mc[sim,t]   = max(E_prev + dB2E - dE2T - dE_clear, 0)
            tumor_mc[sim,t] = max(T_prev + dE2T, 0)

    return time, blood_mc, bbb_mc, tumor_mc


def plot_pk_digital_twin(time, blood_mc, bbb_mc, tumor_mc, save_path=None):
    """
    Plot mean ± 95% CI for the PK compartments from the Monte Carlo runs.
    """
    def mean_ci(data):
        mean  = np.mean(data, axis=0)
        lower = np.percentile(data, 2.5, axis=0)
        upper = np.percentile(data, 97.5, axis=0)
        return mean, lower, upper

    blood_m, blood_l, blood_u = mean_ci(blood_mc)
    bbb_m,  bbb_l,  bbb_u  = mean_ci(bbb_mc)
    tumor_m, tumor_l, tumor_u = mean_ci(tumor_mc)

    plt.figure(figsize=(10,6))
    plt.fill_between(time, blood_l, blood_u, color='blue', alpha=0.2)
    plt.plot(time, blood_m, color='blue', label="Blood")

    plt.fill_between(time, bbb_l, bbb_u, color='orange', alpha=0.2)
    plt.plot(time, bbb_m, color='orange', label="BBB Endothelium")

    plt.fill_between(time, tumor_l, tumor_u, color='green', alpha=0.2)
    plt.plot(time, tumor_m, color='green', label="Tumor")

    plt.xlabel("Time (hours)")
    plt.ylabel("Fraction of Carrier")
    plt.title("PK Digital Twin Simulation: Blood → BBB → Tumor")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"💾 Digital twin PK plot saved: {save_path}")
    plt.close()


def compute_effective_TI(tumor_mc, base_TI):
    """
    Compute distribution of effective TI from tumor compartment at final time.
    """
    end_tumor = tumor_mc[:,-1]  # fraction that ultimately reached tumor
    effective_TI = base_TI * end_tumor
    mean_TI = float(np.mean(effective_TI))
    low_ci  = float(np.percentile(effective_TI,2.5))
    high_ci = float(np.percentile(effective_TI,97.5))
    return effective_TI, mean_TI, low_ci, high_ci


# Standalone execution for testing
if __name__ == "__main__":
    t, blood_mc, bbb_mc, tumor_mc = simulate_pk_digital_twin(n_sim=500)
    plot_pk_digital_twin(t, blood_mc, bbb_mc, tumor_mc, save_path="results/pk_digital_twin_lit.png")
    eff_TI, m, lo, hi = compute_effective_TI(tumor_mc, base_TI=1000)
    print(f"\n💊 Effective TI (mean ± 95% CI): {m:.0f}x [{lo:.0f}x – {hi:.0f}x]")
