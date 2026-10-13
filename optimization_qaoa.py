"""
T4 QAOA Optimization (SIMPLIFIED - PyCharm Compatible)
======================================================
9-qubit Th-QAOA with NO problematic primitives.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp
import time

QAOA_PARAMS = {
    "n_qubits": 9,
    "p_layers": 3,
    "n_shots": 2048,  # Reduced for speed
}

BIOMARKER_THRESHOLDS = {
    0: np.array([2.0, 3.5, 5.0]),  # EGFRvIII
    1: np.array([2.5, 4.0, 6.0]),  # MMP2
    2: np.array([4.0, 6.5, 10.0])  # miR210
}

class T4QAOAOptimizer:
    def __init__(self, tumor_samples, healthy_samples):
        self.tumor = np.asarray(tumor_samples)
        self.healthy = np.asarray(healthy_samples)
        self.backend = AerSimulator()
        self.n_samples = len(tumor_samples)

    def bits_to_thresholds(self, bitstring):
        """Decode 9-bit → [T1,T2,T3]."""
        bits = np.array([int(b) for b in bitstring[::-1]])
        T = np.zeros(3)
        for biomarker in range(3):
            qubit_start = biomarker * 3
            level = (bits[qubit_start] * 4 + bits[qubit_start+1] * 2 + bits[qubit_start+2])
            T[biomarker] = BIOMARKER_THRESHOLDS[biomarker][min(level, 2)]  # Cap at 2
        return T

    def quantum_cost(self, thresholds):
        """Clinical cost J(T)."""
        tumor_active = np.all(self.tumor >= thresholds, axis=1).mean()
        healthy_active = np.all(self.healthy >= thresholds, axis=1).mean()
        return 2.5 * healthy_active + (1.0 - tumor_active)

    def simple_qaoa_circuit(self, gammas, betas):
        """SIMPLIFIED QAOA - NO complex Hamiltonian."""
        n_qubits = QAOA_PARAMS["n_qubits"]
        p = QAOA_PARAMS["p_layers"]
        qc = QuantumCircuit(n_qubits)

        # Initial superposition
        for q in range(n_qubits):
            qc.h(q)

        # QAOA layers (standard implementation)
        for layer in range(p):
            # Cost layer (simple Z rotations)
            for q in range(n_qubits):
                qc.rz(2 * gammas[layer], q)

            # Mixer layer
            for q in range(n_qubits):
                qc.rx(2 * betas[layer], q)

        qc.measure_all()
        return qc

    def sample_qaoa(self, gammas, betas, n_shots=QAOA_PARAMS["n_shots"]):
        """Run QAOA circuit."""
        qc = self.simple_qaoa_circuit(gammas, betas)
        job = self.backend.run(qc, shots=n_shots)
        result = job.result()
        counts = result.get_counts()

        threshold_samples = []
        costs = []
        for bitstr, count in counts.items():
            T = self.bits_to_thresholds(bitstr)
            cost = self.quantum_cost(T)
            threshold_samples.append(T)
            costs.append(cost)

        return np.array(threshold_samples), np.array(costs)

    def qaoa_objective(self, params):
        """Expected cost over QAOA samples."""
        p = QAOA_PARAMS["p_layers"]
        gammas = params[:p]
        betas = params[p:]
        _, costs = self.sample_qaoa(gammas, betas)
        return np.mean(costs)

    def optimize_qaoa(self, maxiter=100):
        """QAOA optimization."""
        print("⚛️ Running 9-qubit Th-QAOA...")
        p = QAOA_PARAMS["p_layers"]
        params0 = np.random.uniform(0, np.pi/2, 2*p)

        opt = COBYLA(maxiter=maxiter)
        result = opt.minimize(self.qaoa_objective, params0)

        # Final evaluation
        gammas = result.x[:p]
        betas = result.x[p:]
        thresholds, costs = self.sample_qaoa(gammas, betas, 4096)

        best_idx = np.argmin(costs)
        best_T = thresholds[best_idx]
        best_cost = costs[best_idx]

        # Verify with classical oracle
        tpr = np.all(self.tumor >= best_T, axis=1).mean()
        fpr = np.all(self.healthy >= best_T, axis=1).mean()
        ti = tpr / max(fpr, 1e-8)

        return {
            "method": "QAOA (9-qubit)",
            "thresholds": best_T,
            "cost": float(best_cost),
            "TPR": float(tpr),
            "FPR": float(fpr),
            "TI": float(ti),
            "n_evals": int(result.nfev * 4096)
        }

if __name__ == "__main__":
    from biomarker_params import sample_biomarkers
    tumor, healthy = sample_biomarkers(n_samples=5000)  # Fast test
    optimizer = T4QAOAOptimizer(tumor, healthy)
    result = optimizer.optimize_qaoa()
    print(f"✅ QAOA: TI={result['TI']:.1f}x")
