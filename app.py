from __future__ import annotations
import random
import time
from typing import Dict, List, Tuple, Any

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# -----------------------------------------------------------------------------
# Genetic Algorithm Core
# -----------------------------------------------------------------------------


class LaminateOptimizer:
    """
    Composite laminate stacking optimizer.
    Rules implemented: 1–8 (including lateral bending stiffness – Rule 8)
    """

    def __init__(self, ply_counts: Dict[int, int]):
        self.ply_counts = ply_counts
        self.initial_pool: List[int] = []
        for angle, count in ply_counts.items():
            self.initial_pool.extend([angle] * int(count))

        self.total_plies = len(self.initial_pool)

    def calculate_clt_proxy(self, sequence: List[int]) -> float:
        return abs(sequence.count(45) - sequence.count(-45)) * 2.5

    def calculate_fitness(self, sequence: List[int]):
        WEIGHTS = {
            "R1": 27.0,   # Symmetry
            "R2": 14.0,   # ±45 balance
            "R3": 999.0,  # HARD constraint
            "R4": 9.0,    # Outer ±45
            "R5": 9.0,    # Distribution
            "R6": 18.0,   # Grouping
            "R7": 9.0,    # Buckling
            "R8": 14.0,   # Lateral bending
        }
        
        n = len(sequence)
        mid = (n - 1) / 2

        rules_result = {}

        # ---------------- RULE 3 (HARD) ----------------
        for i in range(n - 1):
            if {sequence[i], sequence[i + 1]} == {0, 90}:
                total_score = 0.0
                max_possible_score = 100.0  # Sum of all rule weights (excluding R3) = 100
                return total_score, {
                    "total_score": total_score,
                    "max_score": max_possible_score,
                    "rules": {
                        "R3": {
                            "weight": WEIGHTS["R3"],
                            "score": 0,
                            "penalty": WEIGHTS["R3"],
                            "reason": "0°–90° yan yana (yasak)"
                        }
                    }
                }

        # ---------------- RULE 1 – Symmetry ----------------
        sym_err = sum(sequence[i] != sequence[-1 - i] for i in range(n // 2))
        score = WEIGHTS["R1"] * (1 - sym_err / max(1, n // 2))
        score = max(0, min(score, WEIGHTS["R1"]))  # Clamp between 0 and weight
        penalty = WEIGHTS["R1"] - score
        rules_result["R1"] = {
            "weight": WEIGHTS["R1"],
            "score": round(score, 2),
            "penalty": round(penalty, 2),
            "reason": "Simetri bozulmuş" if sym_err > 0 else ""
        }

        # ---------------- RULE 2 – ±45 Balance ----------------
        diff = abs(sequence.count(45) - sequence.count(-45))
        score = max(0, WEIGHTS["R2"] * (1 - diff / max(1, n)))
        score = max(0, min(score, WEIGHTS["R2"]))
        penalty = WEIGHTS["R2"] - score
        rules_result["R2"] = {
            "weight": WEIGHTS["R2"],
            "score": round(score, 2),
            "penalty": round(penalty, 2),
            "reason": "+45 / -45 sayıları eşit değil" if diff > 0 else ""
        }

        # ---------------- RULE 4 – Outer ±45 ----------------
        if abs(sequence[0]) == 45 and abs(sequence[-1]) == 45:
            score = WEIGHTS["R4"]
            reason = ""
        else:
            score = WEIGHTS["R4"] * 0.5
            reason = "Dış katmanlar ±45 değil"
        score = max(0, min(score, WEIGHTS["R4"]))
        penalty = WEIGHTS["R4"] - score
        rules_result["R4"] = {
            "weight": WEIGHTS["R4"],
            "score": round(score, 2),
            "penalty": round(penalty, 2),
            "reason": reason
        }

        # ---------------- RULE 5 – Distribution ----------------
        adj_same = sum(sequence[i] == sequence[i + 1] for i in range(n - 1))
        score = WEIGHTS["R5"] * (1 - adj_same / max(1, n))
        score = max(0, min(score, WEIGHTS["R5"]))
        penalty = WEIGHTS["R5"] - score
        rules_result["R5"] = {
            "weight": WEIGHTS["R5"],
            "score": round(score, 2),
            "penalty": round(penalty, 2),
            "reason": "Aynı açılı katmanlar ardışık" if adj_same > 0 else ""
        }

        # ---------------- RULE 6 – Grouping ----------------
        max_group = 1
        curr = 1
        for i in range(1, n):
            curr = curr + 1 if sequence[i] == sequence[i - 1] else 1
            max_group = max(max_group, curr)
        if max_group <= 3:
            score = WEIGHTS["R6"]
            reason = ""
        else:
            score = WEIGHTS["R6"] * (3 / max_group)
            reason = f"{max_group} katman üst üste"
        score = max(0, min(score, WEIGHTS["R6"]))
        penalty = WEIGHTS["R6"] - score
        rules_result["R6"] = {
            "weight": WEIGHTS["R6"],
            "score": round(score, 2),
            "penalty": round(penalty, 2),
            "reason": reason
        }

        # ---------------- RULE 7 – Buckling ----------------
        zone = sequence[: n // 4] + sequence[-n // 4 :]
        if len(zone) > 0:
            ratio = sum(abs(x) == 45 for x in zone) / len(zone)
            score = WEIGHTS["R7"] * min(1.0, ratio / 0.5)
            reason = "Uç bölgelerde ±45 yetersiz" if ratio < 0.5 else ""
        else:
            score = WEIGHTS["R7"]
            reason = ""
        score = max(0, min(score, WEIGHTS["R7"]))
        penalty = WEIGHTS["R7"] - score
        rules_result["R7"] = {
            "weight": WEIGHTS["R7"],
            "score": round(score, 2),
            "penalty": round(penalty, 2),
            "reason": reason
        }

        # ---------------- RULE 8 – Lateral Bending ----------------
        penalty_sum = 0
        for i, ang in enumerate(sequence):
            if ang == 90:
                dist = abs(i - mid) / max(1, mid)
                if dist < 0.35:
                    penalty_sum += (0.35 - dist)
        # Normalize penalty to [0, 1] range
        penalty_norm = min(1.0, penalty_sum)  # Cap at 1.0
        score = WEIGHTS["R8"] * (1 - penalty_norm)
        score = max(0, min(score, WEIGHTS["R8"]))
        penalty = WEIGHTS["R8"] - score
        rules_result["R8"] = {
            "weight": WEIGHTS["R8"],
            "score": round(score, 2),
            "penalty": round(penalty, 2),
            "reason": "90° katmanlar orta düzleme yakın" if penalty_sum > 0 else ""
        }

        # ---------------- FINAL SCORE ----------------
        total_score = sum(r["score"] for r in rules_result.values())
        max_possible_score = 100.0  # Sum of all rule weights (excluding R3) = 100

        return total_score, {
            "total_score": round(total_score, 2),
            "max_score": max_possible_score,
            "rules": rules_result
        }

    def run_genetic_algorithm(
        self, population_size: int = 120, generations: int = 600
    ) -> Tuple[List[int], float, Dict[str, float], List[float]]:
        population: List[List[int]] = []
        for _ in range(population_size):
            ind = self.initial_pool[:]
            random.shuffle(ind)
            population.append(ind)

        best_sol: List[int] | None = None
        best_fit = -1.0
        best_det: Dict[str, float] = {}
        history: List[float] = []

        for gen in range(generations):
            scored_pop = []
            for ind in population:
                fit, det = self.calculate_fitness(ind)
                scored_pop.append((fit, ind))
                if fit > best_fit:
                    best_fit = fit
                    best_sol = ind[:]
                    best_det = det

            history.append(best_fit)
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            elite_idx = max(1, int(population_size * 0.1))
            next_gen = [x[1][:] for x in scored_pop[:elite_idx]]

            while len(next_gen) < population_size:
                parent = max(random.sample(scored_pop, 3), key=lambda x: x[0])[1][:]
                # swap mutation
                if random.random() < 0.2:
                    i1, i2 = random.sample(range(self.total_plies), 2)
                    parent[i1], parent[i2] = parent[i2], parent[i1]
                # symmetry helper mutation
                if random.random() < 0.4:
                    idx_left = random.randint(0, (self.total_plies // 2) - 1)
                    idx_right = self.total_plies - 1 - idx_left
                    val_left = parent[idx_left]
                    if parent[idx_right] != val_left:
                        candidates = [
                            i for i, x in enumerate(parent) if x == val_left and i != idx_left
                        ]
                        if candidates:
                            swap_target = random.choice(candidates)
                            parent[idx_right], parent[swap_target] = (
                                parent[swap_target],
                                parent[idx_right],
                            )
                next_gen.append(parent)
            population = next_gen

        return best_sol or [], best_fit, best_det, history

    def auto_optimize(
        self,
        runs: int = 10,
        population_size: int = 180,
        generations: int = 800,
        stagnation_window: int = 150,
    ) -> Dict[str, Any]:
        """
        Automatic multi-run optimization system.
        
        Runs the genetic algorithm multiple times and tracks the best solution
        across all runs. Detects early convergence using fitness stagnation.
        
        Args:
            runs: Number of independent GA runs to execute
            population_size: Population size for each run
            generations: Maximum generations per run
            stagnation_window: Number of generations to check for stagnation
            
        Returns:
            Dictionary containing:
                - best_sequence: Best stacking sequence found across all runs
                - best_fitness: Fitness score of the best sequence
                - penalties: Penalty details for the best sequence
                - history: Combined history from all runs (best fitness per generation)
        """
        # Track the best solution across all runs
        global_best_sequence: List[int] | None = None
        global_best_fitness = -1.0
        global_best_penalties: Dict[str, float] = {}
        all_histories: List[List[float]] = []
        
        print(f"Starting auto-optimization: {runs} runs, pop={population_size}, gen={generations}")
        
        # Run the genetic algorithm multiple times
        for run_num in range(1, runs + 1):
            print(f"Run {run_num}/{runs}...")
            
            # Run a single GA execution
            sequence, fitness, penalties, history = self.run_genetic_algorithm(
                population_size=population_size,
                generations=generations
            )
            
            # Track history for this run
            all_histories.append(history)
            
            # Check for early convergence using fitness stagnation
            if len(history) >= stagnation_window:
                # Get the last N generations
                recent_fitness = history[-stagnation_window:]
                max_recent = max(recent_fitness)
                min_recent = min(recent_fitness)
                fitness_range = max_recent - min_recent
                
                # If fitness has stagnated (range < 0.01), print convergence message
                if fitness_range < 0.01:
                    print(f"  Run {run_num}: Converged early (fitness range: {fitness_range:.6f})")
            
            # Update global best if this run found a better solution
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_sequence = sequence[:]
                global_best_penalties = penalties.copy()
                print(f"  Run {run_num}: New best fitness = {fitness:.2f}")
        
        # Combine histories: take the best fitness at each generation across all runs
        combined_history: List[float] = []
        max_gen_length = max(len(h) for h in all_histories) if all_histories else 0
        
        for gen_idx in range(max_gen_length):
            # Get the best fitness at this generation across all runs
            gen_best = -1.0
            for history in all_histories:
                if gen_idx < len(history):
                    gen_best = max(gen_best, history[gen_idx])
            combined_history.append(gen_best)
        
        print(f"Auto-optimization complete. Best fitness: {global_best_fitness:.2f}")
        
        return {
            "best_sequence": global_best_sequence or [],
            "best_fitness": round(global_best_fitness, 2),
            "penalties": global_best_penalties,
            "history": combined_history,
        }


class DropOffOptimizer:
    """
    Tapering optimizer for ply drop-off.
    """

    def __init__(self, master_sequence: List[int], base_optimizer: LaminateOptimizer):
        self.master_sequence = master_sequence
        self.base_opt = base_optimizer
        self.total_plies = len(master_sequence)

    def optimize_drop(self, target_ply: int) -> Tuple[List[int], float, List[int]]:
        remove_cnt = self.total_plies - target_ply
        if remove_cnt <= 0:
            return self.master_sequence, 0.0, []
        if remove_cnt % 2 != 0:
            remove_cnt += 1

        pairs_to_remove = remove_cnt // 2
        half_len = self.total_plies // 2
        search_indices = list(range(1, half_len))

        best_candidate = None
        best_key = None
        best_dropped = []

        attempts = 1200
        for _ in range(attempts):
            left_drops = random.sample(search_indices, min(pairs_to_remove, len(search_indices)))
            left_drops.sort()

            all_drops = []
            for idx in left_drops:
                all_drops.append(idx)
                all_drops.append(self.total_plies - 1 - idx)
            all_drops.sort()

            temp_seq = [ang for i, ang in enumerate(self.master_sequence) if i not in all_drops]

            total_score, details = self.base_opt.calculate_fitness(temp_seq)

            # 🚫 HARD FAIL
            if total_score <= 0:
                continue

            rules = details["rules"]

            # 🔒 DROP-OFF HARD GUARDS
            if rules["R1"]["score"] < 0.9 * rules["R1"]["weight"]:
                continue
            if rules["R8"]["score"] < 0.9 * rules["R8"]["weight"]:
                continue

            # 🔑 Selection key (lexicographic)
            key = (
                rules["R1"]["penalty"] + rules["R8"]["penalty"],  # primary
                -total_score                                     # secondary
            )

            if best_key is None or key < best_key:
                best_key = key
                best_candidate = temp_seq
                best_dropped = all_drops

        if best_candidate is None:
            return self.master_sequence, 0.0, []

        return best_candidate, best_key[1] * -1, best_dropped


# -----------------------------------------------------------------------------
# Flask App
# -----------------------------------------------------------------------------


def create_app() -> Flask:
    app = Flask(__name__, static_folder=".", static_url_path="")

    @app.route("/")
    def index():
        return send_from_directory(".", "index.html")

    @app.post("/optimize")
    def optimize():
        payload = request.get_json(force=True, silent=True) or {}
        ply_counts = payload.get("ply_counts", {})
        # Default example if nothing provided
        ply_counts = {
            int(k): int(v)
            for k, v in ply_counts.items()
            if str(v).isdigit()
        } or {0: 18, 90: 18, 45: 18, -45: 18}

        population_size = int(payload.get("population_size", 120))
        generations = int(payload.get("generations", 600))
        min_drop = int(payload.get("min_drop", 48))
        drop_step = int(payload.get("drop_step", 8))

        optimizer = LaminateOptimizer(ply_counts)
        start_time = time.time()
        master_seq, master_score, details, history = optimizer.run_genetic_algorithm(
            population_size=population_size, generations=generations
        )
        ga_elapsed = time.time() - start_time

        drop_targets = []
        temp = len(master_seq)
        while temp > min_drop:
            temp -= drop_step
            if temp > 0:
                drop_targets.append(temp)

        drop_opt = DropOffOptimizer(master_seq, optimizer)
        drop_results_list = []
        current_seq = master_seq
        for target in drop_targets:
            drop_opt.master_sequence = current_seq
            drop_opt.total_plies = len(current_seq)
            new_seq, sc, dropped_indices = drop_opt.optimize_drop(target)
            drop_results_list.append(
                {"target": target, "seq": new_seq, "score": sc, "dropped": dropped_indices}
            )
            current_seq = new_seq

        # details is now a dict with "total_score", "max_score", "rules"
        response = {
            "master_sequence": master_seq,
            "fitness_score": details.get("total_score", master_score),
            "max_score": details.get("max_score", 100),
            "penalties": details.get("rules", {}),
            "history": history,
            "drop_off_results": drop_results_list,
            "stats": {
                "plies": len(master_seq),
                "duration_seconds": round(ga_elapsed, 2),
                "population_size": population_size,
                "generations": generations,
            },
        }
        return jsonify(response)

    @app.post("/auto_optimize")
    def auto_optimize():
        """
        Automatic multi-run optimization endpoint.
        
        Runs the genetic algorithm multiple times with convergence detection
        and returns the best solution found across all runs.
        
        Request body (JSON):
            - ply_counts: Dict of ply angles and counts (e.g., {0: 18, 90: 18, 45: 18, -45: 18})
            - runs: Number of independent GA runs (default: 10)
            - population_size: Population size per run (default: 180)
            - generations: Maximum generations per run (default: 800)
            - stagnation_window: Generations to check for stagnation (default: 150)
        
        Returns:
            JSON response with:
                - best_sequence: Best stacking sequence found
                - best_fitness: Fitness score of best sequence
                - penalties: Penalty details for best sequence
                - history: Combined fitness history across all runs
        """
        payload = request.get_json(force=True, silent=True) or {}
        ply_counts = payload.get("ply_counts", {})
        # Default example if nothing provided
        ply_counts = {
            int(k): int(v)
            for k, v in ply_counts.items()
            if str(v).isdigit()
        } or {0: 18, 90: 18, 45: 18, -45: 18}

        # Auto-optimization parameters with defaults
        runs = int(payload.get("runs", 10))
        population_size = int(payload.get("population_size", 180))
        generations = int(payload.get("generations", 800))
        stagnation_window = int(payload.get("stagnation_window", 150))

        # Create optimizer and run auto-optimization
        optimizer = LaminateOptimizer(ply_counts)
        start_time = time.time()
        
        result = optimizer.auto_optimize(
            runs=runs,
            population_size=population_size,
            generations=generations,
            stagnation_window=stagnation_window,
        )
        
        elapsed = time.time() - start_time

        # Get the detailed fitness information
        optimizer = LaminateOptimizer(ply_counts)
        _, fitness_details = optimizer.calculate_fitness(result["best_sequence"])
        
        # Build response with additional stats
        response = {
            "master_sequence": result["best_sequence"],
            "fitness_score": fitness_details.get("total_score", result["best_fitness"]),
            "max_score": fitness_details.get("max_score", 100),
            "penalties": fitness_details.get("rules", {}),
            "history": result["history"],
            "stats": {
                "plies": len(result["best_sequence"]),
                "duration_seconds": round(elapsed, 2),
                "runs": runs,
                "population_size": population_size,
                "generations": generations,
                "stagnation_window": stagnation_window,
            },
        }
        return jsonify(response)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

