import csv
import math
import heapq
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ============================================================
# 0. USER SETTINGS — CHANGE ONLY THIS BLOCK FOR NORMAL USE
# ============================================================

# Choose one experiment:
#   "quantization"       -> comparison of three 3-bin threshold configurations
#   "reward_sensitivity" -> one-at-a-time ±20% sensitivity analysis for 5 reward parameters
EXPERIMENT = "reward_sensitivity"

# Use QUICK_TEST = True only for a short functional check.
# Final experiments for the paper require QUICK_TEST = False.
QUICK_TEST = False

QUICK_N_EPISODES = 20
FINAL_N_EPISODES = 250_000

QUICK_N_EVAL_SEEDS = 2
FINAL_N_EVAL_SEEDS = 50

# For a first functional check only one scenario is enough.
QUICK_SCENARIOS = ["shear_horizontal"]

# Final scenario set for the 3-bin threshold comparison: all seven variants.
QUANTIZATION_SCENARIOS = [
    "uniform",
    "cyclone_left",
    "cyclone_right",
    "radial_in",
    "radial_out",
    "shear_horizontal",
    "shear_vertical",
]

# Final scenario set for comment 3 (reward sensitivity):
# one representative variant from each wind-field family.
SENSITIVITY_SCENARIOS = [
    "uniform",
    "cyclone_left",
    "radial_in",
    "shear_horizontal",
]

# Three independent training runs are used in all reported additional experiments.
TRAINING_SEEDS = [0, 1, 2]

# Fixed evaluation seeds. The starting point on the left edge is derived from each seed.
EVAL_SEED_START = 1000

# Progress printing.
VERBOSE_EVERY_QUICK = 1000
VERBOSE_EVERY_FINAL = 25000

# Save a CSV in addition to printing a copyable summary to the terminal.
SAVE_CSV = True

# ============================================================
# 1. Effective settings derived from QUICK_TEST
# ============================================================

N_EPISODES = QUICK_N_EPISODES if QUICK_TEST else FINAL_N_EPISODES
N_EVAL_SEEDS = QUICK_N_EVAL_SEEDS if QUICK_TEST else FINAL_N_EVAL_SEEDS
VERBOSE_EVERY = VERBOSE_EVERY_QUICK if QUICK_TEST else VERBOSE_EVERY_FINAL

# ============================================================
# 2. Grid and actions
# ============================================================

W, H = 40, 40
goal = (W - 1, H - 1)

actions = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
]
n_actions = len(actions)


def is_valid(x: int, y: int) -> bool:
    return 0 <= x < W and 0 <= y < H


def sample_train_start_anywhere(rng: random.Random):
    while True:
        x = rng.randrange(W)
        y = rng.randrange(H)
        if (x, y) != goal:
            return x, y


def eval_left_edge_start_from_seed(seed: int):
    rng = np.random.RandomState(seed + 271828)
    return (0, int(rng.randint(0, H)))


@dataclass(frozen=True)
class WindScenario:
    key: str
    family: str
    cyclone_sign: int = 0       # +1 = CCW (left), -1 = CW (right)
    radial_outward: bool = False
    shear_vertical: bool = False


SCENARIOS = {
    "uniform": WindScenario("uniform", "type1"),
    "cyclone_left": WindScenario("cyclone_left", "cyclone", cyclone_sign=+1),
    "cyclone_right": WindScenario("cyclone_right", "cyclone", cyclone_sign=-1),
    "radial_in": WindScenario("radial_in", "radial", radial_outward=False),
    "radial_out": WindScenario("radial_out", "radial", radial_outward=True),
    "shear_horizontal": WindScenario("shear_horizontal", "shear", shear_vertical=False),
    "shear_vertical": WindScenario("shear_vertical", "shear", shear_vertical=True),
}

# ============================================================
# 3. Physical model constants
# ============================================================

CELL_SIZE_M = 10.0
DRONE_AIRSPEED_CRUISE = 10.0
P_HOVER_W = 220.0
K_DRAG_W_PER_MPS2 = 1.2

lambda_L = {
    "type1": 0.0,
    "cyclone": 0.0,
    "radial": 0.0,
    "shear": 0.0,
}

# ============================================================
# 4. Wind-field generation
# ============================================================


def smooth_field(field: np.ndarray, iters: int = 3) -> np.ndarray:
    """Same 3x3 clipped-neighbourhood averaging as in the previous files,
    implemented with NumPy slicing for speed."""
    arr = field.copy()
    ones = np.ones_like(arr)
    for _ in range(iters):
        p = np.pad(arr, 1, mode="constant", constant_values=0.0)
        m = np.pad(ones, 1, mode="constant", constant_values=0.0)

        total = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
            p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        )
        count = (
            m[:-2, :-2] + m[:-2, 1:-1] + m[:-2, 2:] +
            m[1:-1, :-2] + m[1:-1, 1:-1] + m[1:-1, 2:] +
            m[2:, :-2] + m[2:, 1:-1] + m[2:, 2:]
        )
        arr = total / count
    return arr


def generate_wind_type1(seed: int = 42):
    rng = np.random.RandomState(seed)
    U0 = rng.uniform(-5.0, 5.0)
    V0 = rng.uniform(-5.0, 5.0)

    u = np.full((H, W), U0, dtype=float)
    v = np.full((H, W), V0, dtype=float)

    a = 0.05
    u += smooth_field(rng.uniform(-a, a, size=(H, W)), iters=4)
    v += smooth_field(rng.uniform(-a, a, size=(H, W)), iters=4)
    return u, v


def generate_cyclone_wind(seed: int = 1, rotation_sign: int = +1):
    rng = np.random.RandomState(seed)

    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    dx = xx - cx
    dy = yy - cy
    r2 = dx * dx + dy * dy

    Omega = rotation_sign * rng.uniform(0.6, 0.95)
    R = rng.uniform(20.0, 30.0)
    f = np.exp(-r2 / (R * R))

    u = -Omega * dy * f
    v = +Omega * dx * f

    u += smooth_field(rng.uniform(-0.05, 0.05, size=(H, W)), iters=2)
    v += smooth_field(rng.uniform(-0.05, 0.05, size=(H, W)), iters=2)
    return u, v


def generate_radial_wind(seed: int = 2, outward: bool = False):
    rng = np.random.RandomState(seed)

    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    dx = xx - cx
    dy = yy - cy
    r2 = dx * dx + dy * dy
    r = np.sqrt(r2) + 1e-3

    V0 = rng.uniform(6.0, 9.0)
    R = rng.uniform(17.0, 27.0)
    mag = V0 * np.exp(-r2 / (R * R))

    ex = dx / r
    ey = dy / r
    sign = +1.0 if outward else -1.0

    u = sign * mag * ex
    v = sign * mag * ey

    u += smooth_field(rng.uniform(-0.12, 0.12, size=(H, W)), iters=2)
    v += smooth_field(rng.uniform(-0.12, 0.12, size=(H, W)), iters=2)
    return u, v


def generate_shear_wind(seed: int = 123, vertical: bool = False):
    rng = np.random.RandomState(seed)

    u = np.zeros((H, W), dtype=float)
    v = np.zeros((H, W), dtype=float)

    if not vertical:
        ys = np.linspace(0.0, 1.0, H)
        Y = np.tile(ys.reshape(H, 1), (1, W))
        U_min = rng.uniform(4.0, 9.0)
        U_max = rng.uniform(-9.0, -4.0)
        u = U_min + (U_max - U_min) * Y
    else:
        xs = np.linspace(0.0, 1.0, W)
        X = np.tile(xs.reshape(1, W), (H, 1))
        V_min = rng.uniform(4.0, 9.0)
        V_max = rng.uniform(-9.0, -4.0)
        v = V_min + (V_max - V_min) * X

    u += smooth_field(rng.uniform(-0.08, 0.08, size=(H, W)), iters=3)
    v += smooth_field(rng.uniform(-0.08, 0.08, size=(H, W)), iters=3)
    return u, v


u_field = np.zeros((H, W), dtype=float)
v_field = np.zeros((H, W), dtype=float)


def set_wind_field(scenario_key: str, seed: int):
    global u_field, v_field
    sc = SCENARIOS[scenario_key]

    if sc.family == "type1":
        u_field, v_field = generate_wind_type1(seed=seed)
    elif sc.family == "cyclone":
        u_field, v_field = generate_cyclone_wind(seed=seed, rotation_sign=sc.cyclone_sign)
    elif sc.family == "radial":
        u_field, v_field = generate_radial_wind(seed=seed, outward=sc.radial_outward)
    elif sc.family == "shear":
        u_field, v_field = generate_shear_wind(seed=seed, vertical=sc.shear_vertical)
    else:
        raise ValueError(f"Unknown family: {sc.family}")

# ============================================================
# 5. Transition cost model
# ============================================================


@dataclass
class StepInfo:
    feasible: bool
    energy_j: float
    distance_m: float
    J_step: float
    airspeed_req: float
    time_s: float
    ground_speed_mps: float
    wind_parallel: float
    wind_cross: float


def step_info(x0: int, y0: int, x1: int, y1: int, family: str) -> StepInfo:
    if not is_valid(x0, y0) or not is_valid(x1, y1):
        return StepInfo(False, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf)

    dx = x1 - x0
    dy = y1 - y0
    L_cells = math.hypot(dx, dy)
    if L_cells == 0.0:
        return StepInfo(False, math.inf, 0.0, math.inf, math.inf, math.inf, 0.0, 0.0, 0.0)

    distance_m = CELL_SIZE_M * L_cells
    ex = dx / L_cells
    ey = dy / L_cells
    nx = -ey
    ny = ex

    w_x = 0.5 * (u_field[y0, x0] + u_field[y1, x1])
    w_y = 0.5 * (v_field[y0, x0] + v_field[y1, x1])

    wind_parallel = w_x * ex + w_y * ey
    wind_cross = w_x * nx + w_y * ny
    airspeed_req = DRONE_AIRSPEED_CRUISE

    if abs(wind_cross) > airspeed_req + 1e-12:
        return StepInfo(False, math.inf, distance_m, math.inf, airspeed_req, math.inf, math.inf, wind_parallel, wind_cross)

    air_parallel = math.sqrt(max(0.0, airspeed_req ** 2 - wind_cross ** 2))
    ground_speed_mps = air_parallel + wind_parallel
    if ground_speed_mps <= 1e-12:
        return StepInfo(False, math.inf, distance_m, math.inf, airspeed_req, math.inf, ground_speed_mps, wind_parallel, wind_cross)

    time_s = distance_m / ground_speed_mps
    power_w = P_HOVER_W + K_DRAG_W_PER_MPS2 * (airspeed_req ** 2)
    energy_j = power_w * time_s
    J_step = energy_j + lambda_L[family] * distance_m

    return StepInfo(True, energy_j, distance_m, J_step, airspeed_req, time_s, ground_speed_mps, wind_parallel, wind_cross)

# ============================================================
# 6. Dijkstra reference
# ============================================================


def dijkstra_shortest_path(scenario_key: str, start_point):
    sc = SCENARIOS[scenario_key]
    sx, sy = start_point
    gx, gy = goal

    dist = np.full((H, W), np.inf)
    prev = np.full((H, W, 2), -1, dtype=int)
    dist[sy, sx] = 0.0
    heap = [(0.0, sx, sy)]

    while heap:
        cost, x, y = heapq.heappop(heap)
        if cost > dist[y, x] + 1e-12:
            continue
        if (x, y) == (gx, gy):
            break

        for dx, dy in actions:
            nx, ny = x + dx, y + dy
            if not is_valid(nx, ny):
                continue
            s = step_info(x, y, nx, ny, sc.family)
            if not s.feasible:
                continue
            new_cost = cost + s.J_step
            if new_cost + 1e-12 < dist[ny, nx]:
                dist[ny, nx] = new_cost
                prev[ny, nx] = (x, y)
                heapq.heappush(heap, (new_cost, nx, ny))

    if not np.isfinite(dist[gy, gx]):
        return [], math.inf, math.inf, math.inf

    path = []
    x, y = gx, gy
    while True:
        path.append((x, y))
        if (x, y) == (sx, sy):
            break
        x, y = prev[y, x]
    path.reverse()

    total_E = 0.0
    total_L = 0.0
    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]
        s = step_info(x0, y0, x1, y1, sc.family)
        total_E += s.energy_j
        total_L += s.distance_m

    total_J = total_E + lambda_L[sc.family] * total_L
    return path, total_E, total_L, total_J

# ============================================================
# 7. Wind discretization and Q-table
# ============================================================

# Baseline from the article.
BASE_WIND_EDGES = np.array([-2.0, 2.0], dtype=float)

# After selecting three bins in the preliminary experiment, the locations of
# the two symmetric thresholds were compared separately.
QUANTIZATION_SCHEMES = [
    np.array([-1.0, 1.0], dtype=float),
    np.array([-2.0, 2.0], dtype=float),
    np.array([-3.0, 3.0], dtype=float),
]

wind_edges = BASE_WIND_EDGES.copy()
N_BIN = len(wind_edges) + 1
Q = np.zeros((H, W, N_BIN, N_BIN, n_actions), dtype=float)


def set_quantization(edges):
    global wind_edges, N_BIN, Q
    wind_edges = np.array(edges, dtype=float)
    N_BIN = len(wind_edges) + 1
    Q = np.zeros((H, W, N_BIN, N_BIN, n_actions), dtype=float)


def wind_bin(value: float) -> int:
    idx = int(np.digitize(value, wind_edges))
    return max(0, min(N_BIN - 1, idx))


def get_wind_bins_at(x: int, y: int):
    return wind_bin(u_field[y, x]), wind_bin(v_field[y, x])

# ============================================================
# 8. Q-learning hyperparameters and reward
# ============================================================

alpha = 0.1
max_steps_rl = 200
eps_start = 1.0
eps_end = 0.05

BASE_REWARD_CFG = {
    "type1": {
        "gamma": 0.99,
        "energy_scale_j": 55.0,
        "alpha_d": 8.0,
        "beta_L": 0.80,
        "delta_dnext": -0.040,
        "goal_bonus": 120.0,
        "fail_penalty": 180.0,
    },
    "cyclone": {
        "gamma": 0.99,
        "energy_scale_j": 50.0,
        "alpha_d": 6.0,
        "beta_L": 0.90,
        "delta_dnext": -0.045,
        "goal_bonus": 130.0,
        "fail_penalty": 200.0,
    },
    "radial": {
        "gamma": 0.99,
        "energy_scale_j": 50.0,
        "alpha_d": 5.0,
        "beta_L": 0.90,
        "delta_dnext": -0.055,
        "goal_bonus": 145.0,
        "fail_penalty": 220.0,
    },
    "shear": {
        "gamma": 0.99,
        "energy_scale_j": 43.0,
        "alpha_d": 6.0,
        "beta_L": 0.90,
        "delta_dnext": -0.040,
        "goal_bonus": 125.0,
        "fail_penalty": 190.0,
    },
}

# Five reward-function parameters varied one at a time by ±20%.
SENSITIVITY_PARAMETERS = [
    "alpha_d",
    "beta_L",
    "delta_dnext",
    "goal_bonus",
    "fail_penalty",
]
SENSITIVITY_MULTIPLIERS = [0.8, 1.2]


def make_reward_cfg(family: str, changed_parameter=None, multiplier=1.0):
    cfg = dict(BASE_REWARD_CFG[family])
    if changed_parameter is not None:
        cfg[changed_parameter] = cfg[changed_parameter] * multiplier
    return cfg


def feasible_action_indices(x: int, y: int, family: str):
    out = []
    for a_idx, (dx, dy) in enumerate(actions):
        nx, ny = x + dx, y + dy
        if not is_valid(nx, ny):
            continue
        if step_info(x, y, nx, ny, family).feasible:
            out.append(a_idx)
    return out


def best_valid_q(x: int, y: int, bwx: int, bwy: int, family: str) -> float:
    vals = [Q[y, x, bwx, bwy, a_idx] for a_idx in feasible_action_indices(x, y, family)]
    return 0.0 if not vals else float(np.max(vals))


def train_q_single(
    scenario_key: str,
    reward_cfg_local: dict,
    random_seed: int = 0,
    n_episodes: int = N_EPISODES,
    verbose_every: int = VERBOSE_EVERY,
):
    """Train one Q-table.

    Environment-seed RNG, start-position RNG and action RNG are separated so that
    different reviewer-test configurations see the same sequence of wind fields and
    training starts for the same random_seed.
    """
    sc = SCENARIOS[scenario_key]

    global Q
    Q[:] = 0.0

    rng_env = random.Random(random_seed + 100_003)
    rng_start = random.Random(random_seed + 200_003)
    rng_action = random.Random(random_seed + 300_003)

    gx, gy = goal
    success_counter = 0

    def eps_by_episode(ep):
        frac = ep / max(1, n_episodes - 1)
        return eps_start + (eps_end - eps_start) * frac

    t0 = time.perf_counter()

    for ep in range(n_episodes):
        seed_ep = rng_env.randrange(10**9)
        set_wind_field(scenario_key, seed=seed_ep)

        x, y = sample_train_start_anywhere(rng_start)
        bwx, bwy = get_wind_bins_at(x, y)
        eps = eps_by_episode(ep)
        reached_goal = False

        for step in range(max_steps_rl):
            if (x, y) == (gx, gy):
                reached_goal = True
                break

            valid_idxs = feasible_action_indices(x, y, sc.family)
            if not valid_idxs:
                break

            if rng_action.random() < eps:
                a_idx = rng_action.choice(valid_idxs)
            else:
                qvals = Q[y, x, bwx, bwy, :]
                best_val = max(float(qvals[idx]) for idx in valid_idxs)
                best_idxs = [idx for idx in valid_idxs if abs(float(qvals[idx]) - best_val) < 1e-12]
                a_idx = rng_action.choice(best_idxs)

            dx, dy = actions[a_idx]
            nx, ny = x + dx, y + dy
            s = step_info(x, y, nx, ny, sc.family)
            if not s.feasible:
                break

            d_t = math.hypot(gx - x, gy - y)
            d_t1 = math.hypot(gx - nx, gy - ny)
            J_norm = s.J_step / reward_cfg_local["energy_scale_j"]

            r = (
                -J_norm
                + reward_cfg_local["alpha_d"] * (d_t - d_t1)
                - reward_cfg_local["beta_L"] * math.hypot(dx, dy)
                + reward_cfg_local["delta_dnext"] * d_t1
            )

            reached = (nx, ny) == (gx, gy)
            last_step = (step == max_steps_rl - 1)

            if reached:
                r += reward_cfg_local["goal_bonus"]
            if (not reached) and last_step:
                r -= reward_cfg_local["fail_penalty"]

            done = reached or last_step
            bwx_next, bwy_next = get_wind_bins_at(nx, ny)
            best_next = 0.0 if done else best_valid_q(nx, ny, bwx_next, bwy_next, sc.family)
            td_target = r + (0.0 if done else reward_cfg_local["gamma"] * best_next)
            td_error = td_target - Q[y, x, bwx, bwy, a_idx]
            Q[y, x, bwx, bwy, a_idx] += alpha * td_error

            x, y = nx, ny
            bwx, bwy = bwx_next, bwy_next

            if done:
                reached_goal = reached
                break

        if reached_goal:
            success_counter += 1

        if verbose_every and (ep + 1) % verbose_every == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"[train] scenario={scenario_key} | episode={ep+1}/{n_episodes} | "
                f"eps={eps:.3f} | train_success={success_counter/(ep+1):.3f} | "
                f"elapsed={elapsed:.1f}s"
            )

    return time.perf_counter() - t0

# ============================================================
# 9. RL policy and evaluation
# ============================================================


def rl_policy_path(scenario_key: str, start_point, forbid_revisits: bool = True):
    """Preserves the behavior of the previous code.

    First, visited nodes are excluded. If no such action exists, the previous fallback
    allowing a revisit is retained so that the reviewer experiments change only the
    tested factor (quantization or reward coefficient), not the policy extractor.
    """
    sc = SCENARIOS[scenario_key]
    sx, sy = start_point
    gx, gy = goal

    x, y = sx, sy
    path = [(x, y)]
    visited = {(x, y)}
    bwx, bwy = get_wind_bins_at(x, y)

    for _ in range(max_steps_rl):
        if (x, y) == (gx, gy):
            break

        qvals = Q[y, x, bwx, bwy, :]
        best_val = -1e18
        best_idxs = []

        for a_idx, (dx, dy) in enumerate(actions):
            nx, ny = x + dx, y + dy
            if not is_valid(nx, ny):
                continue
            if not step_info(x, y, nx, ny, sc.family).feasible:
                continue
            if forbid_revisits and (nx, ny) in visited:
                continue

            val = float(qvals[a_idx])
            if val > best_val + 1e-12:
                best_val = val
                best_idxs = [a_idx]
            elif abs(val - best_val) < 1e-12:
                best_idxs.append(a_idx)

        # Preserve the fallback from the existing program.
        if not best_idxs:
            best_val = -1e18
            for a_idx, (dx, dy) in enumerate(actions):
                nx, ny = x + dx, y + dy
                if not is_valid(nx, ny):
                    continue
                if not step_info(x, y, nx, ny, sc.family).feasible:
                    continue
                val = float(qvals[a_idx])
                if val > best_val + 1e-12:
                    best_val = val
                    best_idxs = [a_idx]
                elif abs(val - best_val) < 1e-12:
                    best_idxs.append(a_idx)

        if not best_idxs:
            break

        if len(best_idxs) == 1:
            a_idx = best_idxs[0]
        else:
            best_choice = None
            best_key = None
            for idx in best_idxs:
                dx2, dy2 = actions[idx]
                nx2, ny2 = x + dx2, y + dy2
                d_new = math.hypot(gx - nx2, gy - ny2)
                s2 = step_info(x, y, nx2, ny2, sc.family)
                key = (d_new, s2.energy_j, idx)
                if best_choice is None or key < best_key:
                    best_choice = idx
                    best_key = key
            a_idx = best_choice

        dx, dy = actions[a_idx]
        x, y = x + dx, y + dy
        path.append((x, y))
        visited.add((x, y))
        bwx, bwy = get_wind_bins_at(x, y)

    reached_goal = ((x, y) == (gx, gy))

    total_E = 0.0
    total_L = 0.0
    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]
        s = step_info(x0, y0, x1, y1, sc.family)
        if not s.feasible:
            return path, math.inf, math.inf, math.inf, False
        total_E += s.energy_j
        total_L += s.distance_m

    total_J = total_E + lambda_L[sc.family] * total_L
    return path, total_E, total_L, total_J, reached_goal


def evaluate_policy_fixed_seeds(scenario_key: str, seeds):
    E_dij = []
    E_rl = []
    L_rl = []
    failed = []

    for seed in seeds:
        set_wind_field(scenario_key, seed=seed)
        start_point = eval_left_edge_start_from_seed(seed)

        _, E_r, L_r, _, reached = rl_policy_path(scenario_key, start_point=start_point)
        if not reached:
            failed.append(seed)
            continue

        _, E_d, _, _ = dijkstra_shortest_path(scenario_key, start_point=start_point)
        E_dij.append(E_d)
        E_rl.append(E_r)
        L_rl.append(L_r)

    n_total = len(seeds)
    n_success = len(E_rl)

    if n_success == 0:
        return {
            "n_success": 0,
            "n_total": n_total,
            "success_rate": 0.0,
            "E_dij_mean_J": math.nan,
            "E_rl_mean_J": math.nan,
            "E_rl_std_J": math.nan,
            "L_rl_mean_m": math.nan,
            "delta_D_percent": math.inf,
            "failed_seeds": failed,
        }

    E_dij_arr = np.asarray(E_dij, dtype=float)
    E_rl_arr = np.asarray(E_rl, dtype=float)
    L_rl_arr = np.asarray(L_rl, dtype=float)

    # Same aggregate definition used in the existing program:
    # 100 * (mean(E_RL) - mean(E_Dijkstra)) / mean(E_Dijkstra)
    delta_D = 100.0 * (np.mean(E_rl_arr) - np.mean(E_dij_arr)) / np.mean(E_dij_arr)

    return {
        "n_success": n_success,
        "n_total": n_total,
        "success_rate": n_success / n_total,
        "E_dij_mean_J": float(np.mean(E_dij_arr)),
        "E_rl_mean_J": float(np.mean(E_rl_arr)),
        "E_rl_std_J": float(np.std(E_rl_arr, ddof=1)) if n_success > 1 else 0.0,
        "L_rl_mean_m": float(np.mean(L_rl_arr)),
        "delta_D_percent": float(delta_D),
        "failed_seeds": failed,
    }

# ============================================================
# 10. Result formatting
# ============================================================

SUMMARY_COLUMNS = [
    "experiment",
    "scenario",
    "family",
    "train_seed",
    "n_episodes",
    "n_eval",
    "n_bins",
    "wind_edges",
    "reward_parameter",
    "multiplier",
    "parameter_value",
    "q_entries",
    "q_memory_mb",
    "n_success",
    "success_rate",
    "E_dij_mean_J",
    "E_rl_mean_J",
    "E_rl_std_J",
    "L_rl_mean_m",
    "delta_D_percent",
    "train_seconds",
]


def make_result_row(
    experiment,
    scenario_key,
    train_seed,
    train_seconds,
    eval_result,
    reward_parameter="",
    multiplier="",
    parameter_value="",
):
    sc = SCENARIOS[scenario_key]
    return {
        "experiment": experiment,
        "scenario": scenario_key,
        "family": sc.family,
        "train_seed": train_seed,
        "n_episodes": N_EPISODES,
        "n_eval": N_EVAL_SEEDS,
        "n_bins": N_BIN,
        "wind_edges": "[" + ";".join(f"{v:g}" for v in wind_edges) + "]",
        "reward_parameter": reward_parameter,
        "multiplier": multiplier,
        "parameter_value": parameter_value,
        "q_entries": int(Q.size),
        "q_memory_mb": Q.nbytes / 1_000_000.0,
        "n_success": eval_result["n_success"],
        "success_rate": eval_result["success_rate"],
        "E_dij_mean_J": eval_result["E_dij_mean_J"],
        "E_rl_mean_J": eval_result["E_rl_mean_J"],
        "E_rl_std_J": eval_result["E_rl_std_J"],
        "L_rl_mean_m": eval_result["L_rl_mean_m"],
        "delta_D_percent": eval_result["delta_D_percent"],
        "train_seconds": train_seconds,
    }


def print_one_result(row):
    print(
        f"[result] scenario={row['scenario']} | bins={row['n_bins']} | "
        f"param={row['reward_parameter'] or '-'} | mult={row['multiplier'] or '-'} | "
        f"success={row['n_success']}/{row['n_eval']} | "
        f"E_D={fmt(row['E_dij_mean_J'], 3)} J | E_RL={fmt(row['E_rl_mean_J'], 3)} J | "
        f"L_RL={fmt(row['L_rl_mean_m'], 2)} m | Delta_D={fmt(row['delta_D_percent'], 3)}%"
    )


def fmt(value, digits=6):
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "nan"
        if math.isinf(float(value)):
            return "inf" if value > 0 else "-inf"
        return f"{float(value):.{digits}f}"
    return str(value)


def print_copyable_summary(rows):
    print("\n" + "=" * 110)
    print("COPYABLE FINAL SUMMARY — COPY EVERYTHING FROM THE NEXT LINE")
    print("=" * 110)
    print("\t".join(SUMMARY_COLUMNS))
    for row in rows:
        values = []
        for c in SUMMARY_COLUMNS:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                values.append(fmt(v, 6))
            else:
                values.append(str(v))
        print("\t".join(values))
    print("=" * 110)
    print("END OF COPYABLE SUMMARY")
    print("=" * 110)


def save_csv(rows, filename):
    path = Path.cwd() / filename
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {path}")

# ============================================================
# 11. Reviewer experiment 4 — wind quantization
# ============================================================


def run_quantization_experiment():
    scenarios = QUICK_SCENARIOS if QUICK_TEST else QUANTIZATION_SCENARIOS
    seeds_eval = list(range(EVAL_SEED_START, EVAL_SEED_START + N_EVAL_SEEDS))
    rows = []

    print("\n### REVIEWER COMMENT 4: QUANTIZATION EXPERIMENT ###")
    print(f"QUICK_TEST={QUICK_TEST} | N_EPISODES={N_EPISODES} | N_EVAL_SEEDS={N_EVAL_SEEDS}")
    print(f"Scenarios: {scenarios}")
    print(f"Training seeds: {TRAINING_SEEDS}")

    for scenario_key in scenarios:
        family = SCENARIOS[scenario_key].family
        base_cfg = make_reward_cfg(family)

        for edges in QUANTIZATION_SCHEMES:
            set_quantization(edges)
            n_bins = len(edges) + 1

            for train_seed in TRAINING_SEEDS:
                print("\n" + "-" * 90)
                print(
                    f"START quantization | scenario={scenario_key} | bins={n_bins} | "
                    f"edges={edges.tolist()} | train_seed={train_seed}"
                )
                train_seconds = train_q_single(
                    scenario_key=scenario_key,
                    reward_cfg_local=base_cfg,
                    random_seed=train_seed,
                    n_episodes=N_EPISODES,
                    verbose_every=VERBOSE_EVERY,
                )
                ev = evaluate_policy_fixed_seeds(scenario_key, seeds_eval)
                row = make_result_row(
                    experiment="quantization",
                    scenario_key=scenario_key,
                    train_seed=train_seed,
                    train_seconds=train_seconds,
                    eval_result=ev,
                )
                rows.append(row)
                print_one_result(row)
                if ev["failed_seeds"]:
                    print(f"Failed RL seeds: {ev['failed_seeds']}")

    print_copyable_summary(rows)
    if SAVE_CSV:
        save_csv(rows, "review_quantization_thresholds_results.csv")
    return rows

# ============================================================
# 12. Reviewer experiment 3 — reward sensitivity
# ============================================================


def run_reward_sensitivity_experiment():
    scenarios = QUICK_SCENARIOS if QUICK_TEST else SENSITIVITY_SCENARIOS
    seeds_eval = list(range(EVAL_SEED_START, EVAL_SEED_START + N_EVAL_SEEDS))
    rows = []

    # Reward sensitivity is evaluated using the final 3-bin discretization
    # with thresholds [-2, 2] m/s.
    set_quantization(BASE_WIND_EDGES)

    print("\n### REVIEWER COMMENT 3: REWARD SENSITIVITY EXPERIMENT ###")
    print(f"QUICK_TEST={QUICK_TEST} | N_EPISODES={N_EPISODES} | N_EVAL_SEEDS={N_EVAL_SEEDS}")
    print(f"Scenarios: {scenarios}")
    print(f"Parameters: {SENSITIVITY_PARAMETERS}")
    print(f"Multipliers: baseline, {SENSITIVITY_MULTIPLIERS}")
    print(f"Training seeds: {TRAINING_SEEDS}")

    for scenario_key in scenarios:
        family = SCENARIOS[scenario_key].family

        # Baseline is trained only once per scenario and training seed.
        for train_seed in TRAINING_SEEDS:
            print("\n" + "-" * 90)
            print(f"START sensitivity BASELINE | scenario={scenario_key} | train_seed={train_seed}")
            cfg = make_reward_cfg(family)
            train_seconds = train_q_single(
                scenario_key=scenario_key,
                reward_cfg_local=cfg,
                random_seed=train_seed,
                n_episodes=N_EPISODES,
                verbose_every=VERBOSE_EVERY,
            )
            ev = evaluate_policy_fixed_seeds(scenario_key, seeds_eval)
            row = make_result_row(
                experiment="reward_sensitivity",
                scenario_key=scenario_key,
                train_seed=train_seed,
                train_seconds=train_seconds,
                eval_result=ev,
                reward_parameter="baseline",
                multiplier=1.0,
                parameter_value="",
            )
            rows.append(row)
            print_one_result(row)
            if ev["failed_seeds"]:
                print(f"Failed RL seeds: {ev['failed_seeds']}")

        # One-at-a-time ±20% changes.
        for parameter in SENSITIVITY_PARAMETERS:
            for multiplier in SENSITIVITY_MULTIPLIERS:
                cfg = make_reward_cfg(family, changed_parameter=parameter, multiplier=multiplier)
                actual_value = cfg[parameter]

                for train_seed in TRAINING_SEEDS:
                    print("\n" + "-" * 90)
                    print(
                        f"START sensitivity | scenario={scenario_key} | parameter={parameter} | "
                        f"multiplier={multiplier:.1f} | value={actual_value} | train_seed={train_seed}"
                    )
                    train_seconds = train_q_single(
                        scenario_key=scenario_key,
                        reward_cfg_local=cfg,
                        random_seed=train_seed,
                        n_episodes=N_EPISODES,
                        verbose_every=VERBOSE_EVERY,
                    )
                    ev = evaluate_policy_fixed_seeds(scenario_key, seeds_eval)
                    row = make_result_row(
                        experiment="reward_sensitivity",
                        scenario_key=scenario_key,
                        train_seed=train_seed,
                        train_seconds=train_seconds,
                        eval_result=ev,
                        reward_parameter=parameter,
                        multiplier=multiplier,
                        parameter_value=actual_value,
                    )
                    rows.append(row)
                    print_one_result(row)
                    if ev["failed_seeds"]:
                        print(f"Failed RL seeds: {ev['failed_seeds']}")

    print_copyable_summary(rows)
    if SAVE_CSV:
        save_csv(rows, "review_reward_sensitivity_results.csv")
    return rows

# ============================================================
# 13. Main
# ============================================================


def print_settings():
    mode = "QUICK CHECK" if QUICK_TEST else "FINAL PAPER RUN"
    print("=" * 90)
    print(f"MODE: {mode}")
    print(f"EXPERIMENT: {EXPERIMENT}")
    print(f"N_EPISODES: {N_EPISODES}")
    print(f"N_EVAL_SEEDS: {N_EVAL_SEEDS}")
    print(f"TRAINING_SEEDS: {TRAINING_SEEDS}")
    print(f"Evaluation starts: (0, y), y deterministically derived from seeds {EVAL_SEED_START}...")
    print("Training starts: random grid nodes excluding the goal")
    print("=" * 90)


if __name__ == "__main__":
    print_settings()

    if EXPERIMENT == "quantization":
        run_quantization_experiment()
    elif EXPERIMENT == "reward_sensitivity":
        run_reward_sensitivity_experiment()
    else:
        raise ValueError(
            f"Unknown EXPERIMENT={EXPERIMENT!r}. Use 'quantization' or 'reward_sensitivity'."
        )
