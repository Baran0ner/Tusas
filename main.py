from __future__ import annotations

import random
import time
from typing import Dict, List, Tuple

import numpy as np
from flask import Flask, jsonify, request, send_from_directory


# =============================================================================
# Laminate Optimizer
# =============================================================================

class LaminateOptimizer:
    """
    Composite laminate stacking optimizer
    Fully aligned with Stacking Rules PDF (Rule 1–8)
    """

    def __init__(self, ply_counts: Dict[int, int], ply_thickness_mm: float = 0.184):
        self.ply_counts = ply_counts
        self.initial_pool: List[int] = []
        for angle, count in ply_counts.items():
            self.initial_pool.extend([angle] * int(count))

        self.total_plies = len(self.initial_pool)
        self.ply_thickness = ply_thickness_mm * 1e-3

    # -------------------------------------------------------------------------
    # CLT ABD MATRICES
    # -------------------------------------------------------------------------
    def calculate_abd_matrices(self, sequence: List[int]) -> Dict:
        E1 = 181e9
        E2 = 10.3e9
        G12 = 7.17e9
        v12 = 0.28

        v21 = v12 * (E2 / E1)
        Q11 = E1 / (1 - v12 * v21)
        Q22 = E2 / (1 - v12 * v21)
        Q12 = v12 * E2 / (1 - v12 * v21)
        Q66 = G12

        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))

        t = self.ply_thickness
        h = len(sequence) * t
        z = np.linspace(-h / 2, h / 2, len(sequence) + 1)

        for k, angle in enumerate(sequence):
            th = np.radians(angle)
            c, s = np.cos(th), np.sin(th)
            c2, s2 = c**2, s**2
            c4, s4 = c2**2, s2**2

            Qb = np.zeros((3, 3))
            Qb[0, 0] = Q11 * c4 + Q22 * s4 + 2 * (Q12 + 2 * Q66) * s2 * c2
            Qb[1, 1] = Q11 * s4 + Q22 * c4 + 2 * (Q12 + 2 * Q66) * s2 * c2
            Qb[0, 1] = (Q11 + Q22 - 4 * Q66) * s2 * c2 + Q12 * (c4 + s4)
            Qb[1, 0] = Qb[0, 1]
            Qb[2, 2] = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * s2 * c2 + Q66 * (c4 + s4)

            Qb[0, 2] = (Q11 - Q12 - 2 * Q66) * s * c**3 + (Q12 - Q22 + 2 * Q66) * s**3 * c
            Qb[2, 0] = Qb[0, 2]
            Qb[1, 2] = (Q11 - Q12 - 2 * Q66) * s**3 * c + (Q12 - Q22 + 2 * Q66) * s * c**3
            Qb[2, 1] = Qb[1, 2]

            zk, zk1 = z[k + 1], z[k]
            A += Qb * (zk - zk1)
            B += 0.5 * Qb * (zk**2 - zk1**2)
            D += (1 / 3) * Qb * (zk**3 - zk1**3)

        return {"A": A, "B": B, "D": D}

    # -------------------------------------------------------------------------
    # FITNESS FUNCTION (RULE 1–8)
    # -------------------------------------------------------------------------
    def calculate_fitness(self, seq: List[int]) -> Tuple[float, Dict[str, float]]:
        score = 100.0
        n = len(seq)
        mid = (n - 1) / 2
        details = {}

        abd = self.calculate_abd_matrices(seq)
        B = abd["B"]
        D = abd["D"]

        # ---------------- Rule 1: Symmetry (distance weighted)
        sym_pen = 0.0
        for i in range(n // 2):
            if seq[i] != seq[-1 - i]:
                dist = abs(i - mid) / mid
                sym_pen += 35 * dist
        sym_pen = min(sym_pen, 35)
        score -= sym_pen
        details["R1_Symmetry"] = round(sym_pen, 2)

        # ---------------- Rule 2: Balance (CLT based)
        bal_pen = min((abs(B[0, 2]) + abs(B[1, 2])) / 1e6, 15)
        score -= bal_pen
        details["R2_Balance"] = round(bal_pen, 2)

        # ---------------- Rule 3: Orientation Percentage
        perc_pen = 0.0
        for ang in [0, 45, -45, 90]:
            ratio = seq.count(ang) / n
            if ratio < 0.08 or ratio > 0.67:
                perc_pen += 5
        perc_pen = min(perc_pen, 15)
        score -= perc_pen
        details["R3_Percentage"] = round(perc_pen, 2)

        # ---------------- Rule 4: External plies (first 2)
        ext_pen = 0.0
        outer = seq[:2] + seq[-2:]
        for ply in outer:
            if abs(ply) != 45:
                ext_pen += 2.5
        score -= ext_pen
        details["R4_External"] = round(ext_pen, 2)

        # ---------------- Rule 5: Regular distribution
        dist_pen = 0.0
        for ang in [0, 45, -45, 90]:
            idx = [i for i, x in enumerate(seq) if x == ang]
            if len(idx) > 1 and np.std(idx) < n / 6:
                dist_pen += 1.25
        score -= dist_pen
        details["R5_Distribution"] = round(dist_pen, 2)

        # ---------------- Rule 6: Max grouping (max 3)
        grp_pen = 0.0
        cnt = 1
        for i in range(1, n):
            if seq[i] == seq[i - 1]:
                cnt += 1
                if cnt > 3:
                    grp_pen += 2
            else:
                cnt = 1
        grp_pen = min(grp_pen, 25)
        score -= grp_pen
        details["R6_Grouping"] = round(grp_pen, 2)

        # ---------------- Rule 7: Buckling (±45 far from mid-plane)
        buck_pen = 0.0
        for i, ang in enumerate(seq):
            if abs(ang) == 45:
                dist = abs(i - mid) / mid
                buck_pen += (1 - dist)
        buck_pen = min(buck_pen, 10)
        score -= buck_pen
        details["R7_Buckling"] = round(buck_pen, 2)

        # ---------------- Rule 8: 90 plies away from mid-plane
        r8_pen = 0.0
        for i, ang in enumerate(seq):
            if ang == 90:
                dist = abs(i - mid) / mid
                if dist < 0.3:
                    r8_pen += 2
        r8_pen = min(r8_pen, 10)
        score -= r8_pen
        details["R8_90_Position"] = round(r8_pen, 2)

        return max(score, 0.0), details

    # -------------------------------------------------------------------------
    # GENETIC ALGORITHM
    # -------------------------------------------------------------------------
    def run_genetic_algorithm(self, pop_size=120, generations=600):
        population = []
        for _ in range(pop_size):
            ind = self.initial_pool[:]
            random.shuffle(ind)
            population.append(ind)

        best_sol, best_fit, best_det = None, -1, {}
        history = []

        for _ in range(generations):
            scored = []
            for ind in population:
                fit, det = self.calculate_fitness(ind)
                scored.append((fit, ind))
                if fit > best_fit:
                    best_fit, best_sol, best_det = fit, ind[:], det

            history.append(best_fit)
            scored.sort(reverse=True, key=lambda x: x[0])
            elite = [x[1][:] for x in scored[: int(pop_size * 0.1)]]

            next_gen = elite[:]
            while len(next_gen) < pop_size:
                parent = max(random.sample(scored, 3), key=lambda x: x[0])[1][:]
                if random.random() < 0.3:
                    i, j = random.sample(range(len(parent)), 2)
                    parent[i], parent[j] = parent[j], parent[i]
                next_gen.append(parent)

            population = next_gen

        return best_sol, best_fit, best_det, history


# =============================================================================
# Flask App
# =============================================================================

def create_app() -> Flask:
    app = Flask(__name__, static_folder=".", static_url_path="")

    @app.route("/")
    def index():
        return send_from_directory(".", "index.html")

    @app.post("/optimize")
    def optimize():
        payload = request.get_json(force=True, silent=True) or {}
        ply_counts = payload.get("ply_counts", {0: 18, 90: 18, 45: 18, -45: 18})

        optimizer = LaminateOptimizer(ply_counts)
        start = time.time()
        seq, score, details, history = optimizer.run_genetic_algorithm()
        elapsed = time.time() - start

        abd = optimizer.calculate_abd_matrices(seq)

        return jsonify(
            {
                "sequence": seq,
                "fitness": round(score, 2),
                "penalties": details,
                "A": abd["A"].tolist(),
                "B": abd["B"].tolist(),
                "D": abd["D"].tolist(),
                "time_s": round(elapsed, 2),
                "history": history,
            }
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)

