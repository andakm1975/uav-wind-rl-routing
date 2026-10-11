# -*- coding: utf-8 -*-
"""
Финальный вычислительный эксперимент для статьи
«Энергоэффективное планирование маршрута мультикоптера в ветровом поле
на основе обучения с подкреплением».

Что делает программа
--------------------
1. Обучает отдельную Q-таблицу для каждого из 7 рассматриваемых сценариев ветра:
   - почти однородное поле;
   - вихревое поле, вращение против часовой стрелки;
   - вихревое поле, вращение по часовой стрелке;
   - сходящееся радиальное поле;
   - расходящееся радиальное поле;
   - горизонтальное сдвиговое поле;
   - вертикальное сдвиговое поле.
2. Для каждого сценария выполняет 3 независимых обучения (seed 0, 1, 2),
   по 250 000 эпизодов каждое.
3. Использует окончательное трехуровневое квантование компонент ветра
   с порогами -2 и +2 м/с.
4. После обучения проверяет каждую политику на 50 НОВЫХ тестовых реализациях
   seed 2000...2049. Тестовые начальные положения лежат на левой границе.
5. Для тех же тестовых реализаций независимо рассчитывает эталон Дейкстры
   и локальный жадный алгоритм.
6. Сохраняет подробные CSV, итоговые сводки, Q-таблицы и рисунки для статьи.
7. Поддерживает точное продолжение обучения после остановки программы:
   каждые CHECKPOINT_EVERY эпизодов сохраняются Q-таблица и состояния ГПСЧ.
8. Все надписи на рисунках выполнены на русском языке. Координаты показаны
   в метрах, цветовая шкала скорости ветра — в м/с, энергозатраты — в Дж,
   длина маршрута — в м.

Примечание по антициклическому правилу
--------------------------------------
При построении RL-маршрута сначала рассматриваются только непосещенные допустимые
соседи. Если таких соседей нет, разрешается резервный выбор среди всех допустимых
соседей. Программа отдельно считает число таких срабатываний и фактических повторных
посещений, чтобы это можно было корректно описать в статье.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import os
import pickle
import platform
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# =============================================================================
# 0. НАСТРОЙКИ ФИНАЛЬНОГО ПРОГОНА
# =============================================================================

QUICK_TEST = False

# Окончательные параметры статьи
# Генерация ветровых полей оптимизирована NumPy-векторизацией без изменения математической модели.
W, H = 40, 40
CELL_SIZE_M = 10.0
goal = (W - 1, H - 1)

DRONE_AIRSPEED_CRUISE = 10.0  # м/с
P_HOVER_W = 220.0             # Вт
K_DRAG_W_PER_MPS2 = 1.2       # Вт/(м/с)^2

# При Va = 10 м/с эффективная мощность равна 340 Вт.
# В коде она вычисляется по формуле P = P_HOVER_W + K_DRAG * Va^2.

# Окончательное трехуровневое квантование, выбранное по результатам предварительных экспериментов
WIND_EDGES = np.array([-2.0, 2.0], dtype=float)
N_BIN = len(WIND_EDGES) + 1

# Обучение
ALPHA = 0.1
GAMMA = 0.99
FINAL_N_EPISODES = 250_000
FINAL_MAX_STEPS = 200
EPS_START = 1.0
EPS_END = 0.05
TRAINING_SEEDS = [0, 1, 2]

# Финальная независимая тестовая выборка
EVAL_SEED_START = 2000
N_EVAL_SEEDS = 50

# Контроль сохранения и возобновления
CHECKPOINT_EVERY = 25_000
VERBOSE_EVERY = 25_000
RESUME = True
SAVE_Q_TABLES = True

# Рисунки
SAVE_ROUTE_FIGURES = True
SAVE_SCENARIO_BOXPLOTS = True
SAVE_SUMMARY_FIGURE = True
FIGURE_FORMATS = ("pdf", "jpg", "eps")
FIGURE_JPG_DPI = 300
FIGURE_TRAIN_SEED = 0
FIGURE_EVAL_SEED = 2000

# Если нужно временно прогнать только часть сценариев, сократите этот список.
SCENARIOS_TO_RUN = [
    "uniform",
    "cyclone_left",
    "cyclone_right",
    "radial_in",
    "radial_out",
    "shear_horizontal",
    "shear_vertical",
]

# Быстрая проверка кода без долгого расчета
if QUICK_TEST:
    N_EPISODES = 20
    MAX_STEPS = 80
    EVAL_SEEDS = list(range(EVAL_SEED_START, EVAL_SEED_START + 2))
    TRAIN_SEEDS_ACTIVE = TRAINING_SEEDS
    CHECKPOINT_EVERY_ACTIVE = 100
    VERBOSE_EVERY_ACTIVE = 100
else:
    N_EPISODES = FINAL_N_EPISODES
    MAX_STEPS = FINAL_MAX_STEPS
    EVAL_SEEDS = list(range(EVAL_SEED_START, EVAL_SEED_START + N_EVAL_SEEDS))
    TRAIN_SEEDS_ACTIVE = TRAINING_SEEDS
    CHECKPOINT_EVERY_ACTIVE = CHECKPOINT_EVERY
    VERBOSE_EVERY_ACTIVE = VERBOSE_EVERY


# =============================================================================
# 1. ВЫХОДНЫЕ КАТАЛОГИ И НАСТРОЙКА ГРАФИКОВ
# =============================================================================

OUTPUT_DIR = Path.cwd() / "main_experiment_results"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
Q_DIR = OUTPUT_DIR / "q_tables"
DETAIL_DIR = OUTPUT_DIR / "details"
FIGURE_DIR = OUTPUT_DIR / "figures"
SUMMARY_DIR = OUTPUT_DIR / "summary"

for _d in [OUTPUT_DIR, CHECKPOINT_DIR, Q_DIR, DETAIL_DIR, FIGURE_DIR, SUMMARY_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# PDF сохраняется с TrueType-шрифтами, текст остается редактируемым.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"


# =============================================================================
# 2. СЦЕНАРИИ ВЕТРОВЫХ ПОЛЕЙ
# =============================================================================

@dataclass(frozen=True)
class Scenario:
    key: str
    family: str
    ru_name: str
    cyclone_sign: int = 0       # +1: против часовой, -1: по часовой
    radial_outward: Optional[bool] = None
    shear_vertical: Optional[bool] = None


SCENARIOS: Dict[str, Scenario] = {
    "uniform": Scenario(
        "uniform", "type1", "почти однородное поле"
    ),
    "cyclone_left": Scenario(
        "cyclone_left", "cyclone", "вихревое поле, против часовой стрелки",
        cyclone_sign=+1,
    ),
    "cyclone_right": Scenario(
        "cyclone_right", "cyclone", "вихревое поле, по часовой стрелке",
        cyclone_sign=-1,
    ),
    "radial_in": Scenario(
        "radial_in", "radial", "сходящееся радиальное поле",
        radial_outward=False,
    ),
    "radial_out": Scenario(
        "radial_out", "radial", "расходящееся радиальное поле",
        radial_outward=True,
    ),
    "shear_horizontal": Scenario(
        "shear_horizontal", "shear", "горизонтальное сдвиговое поле",
        shear_vertical=False,
    ),
    "shear_vertical": Scenario(
        "shear_vertical", "shear", "вертикальное сдвиговое поле",
        shear_vertical=True,
    ),
}


# =============================================================================
# 3. РЕШЕТКА И МНОЖЕСТВО ДЕЙСТВИЙ
# =============================================================================

# Единичная окрестность Мура: 8 возможных перемещений.
actions = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
]
n_actions = len(actions)


def is_valid(x: int, y: int) -> bool:
    return 0 <= x < W and 0 <= y < H


def sample_train_start_anywhere(rng: random.Random) -> Tuple[int, int]:
    """Случайный старт обучения почти из любой клетки, кроме цели."""
    while True:
        x = rng.randrange(W)
        y = rng.randrange(H)
        if (x, y) != goal:
            return x, y


def eval_left_edge_start_from_seed(seed: int) -> Tuple[int, int]:
    """Воспроизводимый старт (0, y) на левой границе для тестовой реализации."""
    rng = np.random.RandomState(seed + 271828)
    return 0, int(rng.randint(0, H))


# =============================================================================
# 4. МОДЕЛИ ВЕТРОВЫХ ПОЛЕЙ
# =============================================================================

u_field = np.zeros((H, W), dtype=float)
v_field = np.zeros((H, W), dtype=float)


def smooth_field(field: np.ndarray, iters: int = 3) -> np.ndarray:
    """То же среднее по обрезанной окрестности 3x3, но без медленных циклов Python.

    Математически эквивалентно исходной реализации: у внутренних узлов усредняются
    9 значений, на границе 6, в углах 4.
    """
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


def generate_uniform_wind(seed: int):
    rng = np.random.RandomState(seed)
    U0 = rng.uniform(-5.0, 5.0)
    V0 = rng.uniform(-5.0, 5.0)
    u = np.full((H, W), U0, dtype=float)
    v = np.full((H, W), V0, dtype=float)
    a = 0.05
    u += smooth_field(rng.uniform(-a, a, size=(H, W)), iters=4)
    v += smooth_field(rng.uniform(-a, a, size=(H, W)), iters=4)
    return u, v


def generate_cyclone_wind(seed: int, rotation_sign: int):
    rng = np.random.RandomState(seed)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    dx = xx - cx
    dy = yy - cy
    r2 = dx * dx + dy * dy

    omega_abs = rng.uniform(0.6, 0.95)
    Omega = float(rotation_sign) * omega_abs
    R = rng.uniform(20.0, 30.0)
    f = np.exp(-r2 / (R * R))

    u = -Omega * dy * f
    v = +Omega * dx * f

    u += smooth_field(rng.uniform(-0.05, 0.05, size=(H, W)), iters=2)
    v += smooth_field(rng.uniform(-0.05, 0.05, size=(H, W)), iters=2)
    return u, v


def generate_radial_wind(seed: int, outward: bool):
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


def generate_shear_wind(seed: int, vertical: bool):
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


def set_wind_field(scenario: Scenario, seed: int) -> None:
    global u_field, v_field
    if scenario.family == "type1":
        u_field, v_field = generate_uniform_wind(seed)
    elif scenario.family == "cyclone":
        u_field, v_field = generate_cyclone_wind(seed, scenario.cyclone_sign)
    elif scenario.family == "radial":
        u_field, v_field = generate_radial_wind(seed, bool(scenario.radial_outward))
    elif scenario.family == "shear":
        u_field, v_field = generate_shear_wind(seed, bool(scenario.shear_vertical))
    else:
        raise ValueError(f"Неизвестное семейство поля: {scenario.family}")


# =============================================================================
# 5. ЭНЕРГЕТИЧЕСКАЯ МОДЕЛЬ ПЕРЕХОДА
# =============================================================================

lambda_L = {
    "type1": 0.0,
    "cyclone": 0.0,
    "radial": 0.0,
    "shear": 0.0,
}


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

    return StepInfo(
        True, energy_j, distance_m, J_step, airspeed_req, time_s,
        ground_speed_mps, wind_parallel, wind_cross,
    )


# =============================================================================
# 6. КВАНТОВАНИЕ И Q-ТАБЛИЦА
# =============================================================================


def wind_bin(value: float) -> int:
    idx = int(np.digitize(value, WIND_EDGES))
    return max(0, min(N_BIN - 1, idx))


def get_wind_bins_at(x: int, y: int) -> Tuple[int, int]:
    return wind_bin(u_field[y, x]), wind_bin(v_field[y, x])


Q = np.zeros((H, W, N_BIN, N_BIN, n_actions), dtype=np.float64)


# =============================================================================
# 7. ФУНКЦИЯ ВОЗНАГРАЖДЕНИЯ
# =============================================================================

reward_cfg = {
    "type1": {
        "gamma": GAMMA,
        "energy_scale_j": 55.0,
        "alpha_d": 8.0,
        "beta_L": 0.80,
        "delta_dnext": -0.040,
        "goal_bonus": 120.0,
        "fail_penalty": 180.0,
    },
    "cyclone": {
        "gamma": GAMMA,
        "energy_scale_j": 50.0,
        "alpha_d": 6.0,
        "beta_L": 0.90,
        "delta_dnext": -0.045,
        "goal_bonus": 130.0,
        "fail_penalty": 200.0,
    },
    "radial": {
        "gamma": GAMMA,
        "energy_scale_j": 50.0,
        "alpha_d": 5.0,
        "beta_L": 0.90,
        "delta_dnext": -0.055,
        "goal_bonus": 145.0,
        "fail_penalty": 220.0,
    },
    "shear": {
        "gamma": GAMMA,
        "energy_scale_j": 43.0,
        "alpha_d": 6.0,
        "beta_L": 0.90,
        "delta_dnext": -0.040,
        "goal_bonus": 125.0,
        "fail_penalty": 190.0,
    },
}


# =============================================================================
# 8. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ Q-ОБУЧЕНИЯ
# =============================================================================


def feasible_action_indices(x: int, y: int, family: str) -> List[int]:
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


def epsilon_by_episode(ep: int) -> float:
    frac = ep / max(1, N_EPISODES - 1)
    return EPS_START + (EPS_END - EPS_START) * frac


def _rngs_for_training(train_seed: int):
    """
    Раздельные потоки случайных чисел для полей, стартов и выбора действий.
    Это исключает случайное изменение последовательности полей из-за различий
    в длине эпизодов и делает независимые обучения полностью воспроизводимыми.
    """
    return (
        random.Random(100_000 + train_seed),
        random.Random(200_000 + train_seed),
        random.Random(300_000 + train_seed),
    )


def checkpoint_path(scenario_key: str, train_seed: int) -> Path:
    return CHECKPOINT_DIR / f"checkpoint_{scenario_key}_train{train_seed}.pkl"


def q_path(scenario_key: str, train_seed: int) -> Path:
    return Q_DIR / f"Q_{scenario_key}_train{train_seed}.npy"


def train_q(scenario: Scenario, train_seed: int) -> Dict[str, float]:
    """Обучение одной политики с возможностью точного продолжения после остановки."""
    global Q

    cp_path = checkpoint_path(scenario.key, train_seed)
    final_q_path = q_path(scenario.key, train_seed)

    env_rng, start_rng, action_rng = _rngs_for_training(train_seed)
    success_counter = 0
    start_episode = 0
    elapsed_before = 0.0

    if RESUME and cp_path.exists():
        with open(cp_path, "rb") as f:
            cp = pickle.load(f)
        Q = cp["Q"]
        start_episode = int(cp["next_episode"])
        success_counter = int(cp["success_counter"])
        elapsed_before = float(cp.get("elapsed_seconds", 0.0))
        env_rng.setstate(cp["env_rng_state"])
        start_rng.setstate(cp["start_rng_state"])
        action_rng.setstate(cp["action_rng_state"])
        print(f"[возобновление] {scenario.key}, seed={train_seed}: эпизод {start_episode}/{N_EPISODES}")
    else:
        Q = np.zeros((H, W, N_BIN, N_BIN, n_actions), dtype=np.float64)

    t0 = time.time()
    gx, gy = goal
    cfg = reward_cfg[scenario.family]

    for ep in range(start_episode, N_EPISODES):
        wind_seed = env_rng.randrange(10**9)
        set_wind_field(scenario, wind_seed)
        x, y = sample_train_start_anywhere(start_rng)
        bwx, bwy = get_wind_bins_at(x, y)
        reached_goal = False
        eps = epsilon_by_episode(ep)

        for step in range(MAX_STEPS):
            if (x, y) == (gx, gy):
                reached_goal = True
                break

            valid_idxs = feasible_action_indices(x, y, scenario.family)
            if not valid_idxs:
                break

            if action_rng.random() < eps:
                a_idx = action_rng.choice(valid_idxs)
            else:
                qvals = Q[y, x, bwx, bwy, :]
                best_val = max(float(qvals[idx]) for idx in valid_idxs)
                best_idxs = [idx for idx in valid_idxs if abs(float(qvals[idx]) - best_val) < 1e-12]
                a_idx = action_rng.choice(best_idxs)

            dx, dy = actions[a_idx]
            nx, ny = x + dx, y + dy
            s = step_info(x, y, nx, ny, scenario.family)
            if not s.feasible:
                break

            d_t = math.hypot(gx - x, gy - y)
            d_t1 = math.hypot(gx - nx, gy - ny)
            J_norm = s.J_step / cfg["energy_scale_j"]

            r = (
                -J_norm
                + cfg["alpha_d"] * (d_t - d_t1)
                - cfg["beta_L"] * math.hypot(dx, dy)
                + cfg["delta_dnext"] * d_t1
            )

            reached = (nx, ny) == (gx, gy)
            last_step = (step == MAX_STEPS - 1)
            if reached:
                r += cfg["goal_bonus"]
            if (not reached) and last_step:
                r -= cfg["fail_penalty"]

            done = reached or last_step
            bwx_next, bwy_next = get_wind_bins_at(nx, ny)
            best_next = 0.0 if done else best_valid_q(nx, ny, bwx_next, bwy_next, scenario.family)
            td_target = r + (0.0 if done else cfg["gamma"] * best_next)
            td_error = td_target - Q[y, x, bwx, bwy, a_idx]
            Q[y, x, bwx, bwy, a_idx] += ALPHA * td_error

            x, y = nx, ny
            bwx, bwy = bwx_next, bwy_next

            if done:
                if reached:
                    reached_goal = True
                break

        if reached_goal:
            success_counter += 1

        episode_done = ep + 1
        elapsed_now = elapsed_before + (time.time() - t0)

        if VERBOSE_EVERY_ACTIVE and episode_done % VERBOSE_EVERY_ACTIVE == 0:
            sr = success_counter / float(episode_done)
            print(
                f"[обучение] {scenario.key} | seed={train_seed} | "
                f"эпизод={episode_done}/{N_EPISODES} | eps={eps:.3f} | "
                f"доля успеха={sr:.3f} | время={elapsed_now:.1f} с"
            )

        if CHECKPOINT_EVERY_ACTIVE and episode_done % CHECKPOINT_EVERY_ACTIVE == 0 and episode_done < N_EPISODES:
            cp = {
                "Q": Q,
                "next_episode": episode_done,
                "success_counter": success_counter,
                "elapsed_seconds": elapsed_now,
                "env_rng_state": env_rng.getstate(),
                "start_rng_state": start_rng.getstate(),
                "action_rng_state": action_rng.getstate(),
            }
            tmp_path = cp_path.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(cp, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cp_path)

    total_elapsed = elapsed_before + (time.time() - t0)
    if SAVE_Q_TABLES:
        np.save(final_q_path, Q)
    if cp_path.exists():
        cp_path.unlink()

    return {
        "train_seconds": float(total_elapsed),
        "train_success_rate": float(success_counter / N_EPISODES),
    }


# =============================================================================
# 9. ТРИ МЕТОДА ПОСТРОЕНИЯ МАРШРУТА
# =============================================================================


def dijkstra_shortest_path(scenario: Scenario, start_point: Tuple[int, int]):
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
            s = step_info(x, y, nx, ny, scenario.family)
            if not s.feasible:
                continue
            new_cost = cost + s.J_step
            if new_cost + 1e-12 < dist[ny, nx]:
                dist[ny, nx] = new_cost
                prev[ny, nx] = (x, y)
                heapq.heappush(heap, (new_cost, nx, ny))

    if not np.isfinite(dist[gy, gx]):
        return [], math.inf, math.inf, math.inf, False

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
        s = step_info(x0, y0, x1, y1, scenario.family)
        total_E += s.energy_j
        total_L += s.distance_m
    total_J = total_E + lambda_L[scenario.family] * total_L
    return path, total_E, total_L, total_J, True


def greedy_path(scenario: Scenario, start_point: Tuple[int, int]):
    sx, sy = start_point
    gx, gy = goal
    x, y = sx, sy
    path = [(x, y)]
    visited = {(x, y)}
    max_steps_greedy = 5 * (W + H)

    for _ in range(max_steps_greedy):
        if (x, y) == (gx, gy):
            break

        best = None
        best_key = None
        for dx, dy in actions:
            nx, ny = x + dx, y + dy
            if not is_valid(nx, ny) or (nx, ny) in visited:
                continue
            s = step_info(x, y, nx, ny, scenario.family)
            if not s.feasible:
                continue
            d_new = math.hypot(gx - nx, gy - ny)
            key = (s.energy_j, d_new)
            if best is None or key < best_key:
                best = (nx, ny)
                best_key = key

        if best is None:
            break

        x, y = best
        path.append((x, y))
        visited.add((x, y))

    reached_goal = ((x, y) == (gx, gy))
    total_E = 0.0
    total_L = 0.0
    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]
        s = step_info(x0, y0, x1, y1, scenario.family)
        total_E += s.energy_j
        total_L += s.distance_m
    total_J = total_E + lambda_L[scenario.family] * total_L
    return path, total_E, total_L, total_J, reached_goal


def rl_policy_path(scenario: Scenario, start_point: Tuple[int, int]):
    """
    Сначала исключает посещенные узлы. Если непосещенных допустимых соседей нет,
    используется резервный выбор среди всех допустимых соседей. Возвращаются счетчики
    срабатываний резерва и фактических повторных посещений.
    """
    sx, sy = start_point
    gx, gy = goal
    x, y = sx, sy
    path = [(x, y)]
    visited = {(x, y)}
    bwx, bwy = get_wind_bins_at(x, y)
    fallback_count = 0
    revisit_count = 0

    for _ in range(MAX_STEPS):
        if (x, y) == (gx, gy):
            break

        qvals = Q[y, x, bwx, bwy, :]
        best_val = -1e18
        best_idxs: List[int] = []

        for a_idx, (dx, dy) in enumerate(actions):
            nx, ny = x + dx, y + dy
            if not is_valid(nx, ny):
                continue
            if not step_info(x, y, nx, ny, scenario.family).feasible:
                continue
            if (nx, ny) in visited:
                continue
            val = float(qvals[a_idx])
            if val > best_val + 1e-12:
                best_val = val
                best_idxs = [a_idx]
            elif abs(val - best_val) < 1e-12:
                best_idxs.append(a_idx)

        if not best_idxs:
            fallback_count += 1
            best_val = -1e18
            for a_idx, (dx, dy) in enumerate(actions):
                nx, ny = x + dx, y + dy
                if not is_valid(nx, ny):
                    continue
                if not step_info(x, y, nx, ny, scenario.family).feasible:
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
                s2 = step_info(x, y, nx2, ny2, scenario.family)
                key = (d_new, s2.energy_j, idx)
                if best_choice is None or key < best_key:
                    best_choice = idx
                    best_key = key
            a_idx = int(best_choice)

        dx, dy = actions[a_idx]
        nx, ny = x + dx, y + dy
        if (nx, ny) in visited:
            revisit_count += 1
        x, y = nx, ny
        path.append((x, y))
        visited.add((x, y))
        bwx, bwy = get_wind_bins_at(x, y)

    reached_goal = ((x, y) == (gx, gy))
    total_E = 0.0
    total_L = 0.0
    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]
        s = step_info(x0, y0, x1, y1, scenario.family)
        if not s.feasible:
            total_E = math.inf
            total_L = math.inf
            break
        total_E += s.energy_j
        total_L += s.distance_m
    total_J = total_E + lambda_L[scenario.family] * total_L if np.isfinite(total_E) else math.inf

    return path, total_E, total_L, total_J, reached_goal, fallback_count, revisit_count


# =============================================================================
# 10. CSV/JSON
# =============================================================================


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: Path) -> List[dict]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_float(v, default=math.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def as_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


# =============================================================================
# 11. ЭТАЛОН И ЖАДНЫЙ АЛГОРИТМ: ОДИН РАЗ НА СЦЕНАРИЙ
# =============================================================================


def baseline_csv_path(scenario: Scenario) -> Path:
    return DETAIL_DIR / f"baseline_{scenario.key}.csv"


def evaluate_baselines(scenario: Scenario) -> List[dict]:
    path_csv = baseline_csv_path(scenario)
    if RESUME and path_csv.exists():
        print(f"[готово] эталон и жадный алгоритм: {scenario.key}")
        return read_csv(path_csv)

    rows = []
    for idx, eval_seed in enumerate(EVAL_SEEDS, start=1):
        set_wind_field(scenario, eval_seed)
        start_point = eval_left_edge_start_from_seed(eval_seed)

        _, E_dij, L_dij, J_dij, dij_ok = dijkstra_shortest_path(scenario, start_point)
        _, E_gr, L_gr, J_gr, gr_ok = greedy_path(scenario, start_point)

        rows.append({
            "scenario": scenario.key,
            "eval_seed": eval_seed,
            "start_x": start_point[0],
            "start_y": start_point[1],
            "dijkstra_success": int(dij_ok),
            "E_dijkstra_J": E_dij if dij_ok else "",
            "L_dijkstra_m": L_dij if dij_ok else "",
            "J_dijkstra": J_dij if dij_ok else "",
            "greedy_success": int(gr_ok),
            "E_greedy_J": E_gr if gr_ok else "",
            "L_greedy_m": L_gr if gr_ok else "",
            "J_greedy": J_gr if gr_ok else "",
        })

        if idx % 10 == 0 or idx == len(EVAL_SEEDS):
            print(f"[базовые методы] {scenario.key}: {idx}/{len(EVAL_SEEDS)}")

    write_csv(path_csv, rows)
    return rows


# =============================================================================
# 12. ОЦЕНКА RL-ПОЛИТИКИ
# =============================================================================


def rl_detail_path(scenario: Scenario, train_seed: int) -> Path:
    return DETAIL_DIR / f"rl_{scenario.key}_train{train_seed}.csv"


def run_summary_path(scenario: Scenario, train_seed: int) -> Path:
    return SUMMARY_DIR / f"run_{scenario.key}_train{train_seed}.json"


def evaluate_rl_policy(scenario: Scenario, train_seed: int, baseline_rows: List[dict], train_info: dict) -> dict:
    detail_path = rl_detail_path(scenario, train_seed)
    summary_path = run_summary_path(scenario, train_seed)

    if RESUME and detail_path.exists() and summary_path.exists():
        print(f"[готово] оценка RL: {scenario.key}, train_seed={train_seed}")
        return load_json(summary_path)

    baseline_by_seed = {as_int(r["eval_seed"]): r for r in baseline_rows}
    rows = []

    for idx, eval_seed in enumerate(EVAL_SEEDS, start=1):
        set_wind_field(scenario, eval_seed)
        start_point = eval_left_edge_start_from_seed(eval_seed)
        _, E_rl, L_rl, J_rl, rl_ok, fallback_count, revisit_count = rl_policy_path(scenario, start_point)
        b = baseline_by_seed[eval_seed]

        rows.append({
            "scenario": scenario.key,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "start_x": start_point[0],
            "start_y": start_point[1],
            "rl_success": int(rl_ok),
            "E_rl_J": E_rl if rl_ok else "",
            "L_rl_m": L_rl if rl_ok else "",
            "J_rl": J_rl if rl_ok else "",
            "rl_fallback_count": fallback_count,
            "rl_revisit_count": revisit_count,
            "dijkstra_success": b["dijkstra_success"],
            "E_dijkstra_J": b["E_dijkstra_J"],
            "L_dijkstra_m": b["L_dijkstra_m"],
            "greedy_success": b["greedy_success"],
            "E_greedy_J": b["E_greedy_J"],
            "L_greedy_m": b["L_greedy_m"],
        })

        if idx % 10 == 0 or idx == len(EVAL_SEEDS):
            print(f"[RL-оценка] {scenario.key}, train_seed={train_seed}: {idx}/{len(EVAL_SEEDS)}")

    write_csv(detail_path, rows)

    rl_success_rows = [r for r in rows if as_int(r["rl_success"]) == 1]
    common_gr_rows = [
        r for r in rl_success_rows
        if as_int(r["greedy_success"]) == 1 and r["E_greedy_J"] not in ("", None)
    ]

    E_rl = np.array([as_float(r["E_rl_J"]) for r in rl_success_rows], dtype=float)
    L_rl = np.array([as_float(r["L_rl_m"]) for r in rl_success_rows], dtype=float)
    E_dij_pair = np.array([as_float(r["E_dijkstra_J"]) for r in rl_success_rows], dtype=float)

    if len(E_rl) > 0:
        delta_D = 100.0 * (float(np.mean(E_rl)) - float(np.mean(E_dij_pair))) / float(np.mean(E_dij_pair))
    else:
        delta_D = math.inf

    if common_gr_rows:
        E_rl_common = np.array([as_float(r["E_rl_J"]) for r in common_gr_rows], dtype=float)
        E_gr_common = np.array([as_float(r["E_greedy_J"]) for r in common_gr_rows], dtype=float)
        improve_gr = 100.0 * (float(np.mean(E_gr_common)) - float(np.mean(E_rl_common))) / float(np.mean(E_gr_common))
    else:
        improve_gr = math.nan

    summary = {
        "scenario": scenario.key,
        "scenario_ru": scenario.ru_name,
        "train_seed": train_seed,
        "n_episodes": N_EPISODES,
        "n_eval": len(EVAL_SEEDS),
        "eval_seed_start": EVAL_SEED_START,
        "n_bins": N_BIN,
        "wind_edges_mps": WIND_EDGES.tolist(),
        "q_entries": int(Q.size),
        "q_memory_mb": float(Q.nbytes / 1_000_000.0),
        "train_seconds": float(train_info.get("train_seconds", math.nan)),
        "train_success_rate": float(train_info.get("train_success_rate", math.nan)),
        "rl_successes": len(rl_success_rows),
        "rl_success_rate": float(len(rl_success_rows) / len(EVAL_SEEDS)),
        "greedy_successes": int(sum(as_int(r["greedy_success"]) for r in baseline_rows)),
        "greedy_success_rate": float(sum(as_int(r["greedy_success"]) for r in baseline_rows) / len(EVAL_SEEDS)),
        "E_dijkstra_on_rl_success_mean_J": float(np.mean(E_dij_pair)) if len(E_dij_pair) else math.nan,
        "E_rl_mean_J": float(np.mean(E_rl)) if len(E_rl) else math.nan,
        "E_rl_std_J": float(np.std(E_rl, ddof=1)) if len(E_rl) > 1 else 0.0,
        "L_rl_mean_m": float(np.mean(L_rl)) if len(L_rl) else math.nan,
        "L_rl_std_m": float(np.std(L_rl, ddof=1)) if len(L_rl) > 1 else 0.0,
        "delta_D_percent": float(delta_D),
        "common_rl_greedy_successes": len(common_gr_rows),
        "improve_rl_vs_greedy_percent": float(improve_gr),
        "rl_fallback_total": int(sum(as_int(r["rl_fallback_count"]) for r in rows)),
        "rl_revisit_total": int(sum(as_int(r["rl_revisit_count"]) for r in rows)),
    }
    save_json(summary_path, summary)
    return summary


# =============================================================================
# 13. РИСУНКИ ДЛЯ СТАТЬИ — ВСЕ НАДПИСИ НА РУССКОМ, ОСИ С ЕДИНИЦАМИ
# =============================================================================


def save_figure_all_formats(fig, filename_pdf: Path):
    """Сохраняет рисунок сразу в PDF, JPG и EPS.
    filename_pdf передаётся с расширением .pdf и используется как базовое имя.
    """
    base = filename_pdf.with_suffix("")
    if "pdf" in FIGURE_FORMATS:
        fig.savefig(base.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    if "jpg" in FIGURE_FORMATS:
        fig.savefig(base.with_suffix(".jpg"), format="jpg", dpi=FIGURE_JPG_DPI, bbox_inches="tight")
    if "eps" in FIGURE_FORMATS:
        fig.savefig(base.with_suffix(".eps"), format="eps", bbox_inches="tight")


def meters_xy(path: List[Tuple[int, int]]):
    xs = [p[0] * CELL_SIZE_M for p in path]
    ys = [p[1] * CELL_SIZE_M for p in path]
    return xs, ys


def plot_wind_and_paths_ru(
    scenario: Scenario,
    path_dij,
    path_gr,
    path_rl,
    start_point,
    greedy_success: bool,
    filename_pdf: Path,
):
    speed_mag = np.sqrt(u_field ** 2 + v_field ** 2)
    fig, ax = plt.subplots(figsize=(7.4, 7.2))

    h = CELL_SIZE_M
    extent = [-0.5 * h, (W - 0.5) * h, -0.5 * h, (H - 0.5) * h]
    im = ax.imshow(speed_mag, origin="lower", cmap="viridis", extent=extent, aspect="equal")

    skip = 3
    xs_idx = np.arange(0, W, skip)
    ys_idx = np.arange(0, H, skip)
    X_idx, Y_idx = np.meshgrid(xs_idx, ys_idx)
    X = X_idx * h
    Y = Y_idx * h
    U = u_field[ys_idx[:, None], xs_idx]
    V = v_field[ys_idx[:, None], xs_idx]
    ax.quiver(X, Y, U, V, color="white", scale=35)

    if path_dij:
        xd, yd = meters_xy(path_dij)
        ax.plot(xd, yd, "-o", color="#D62728", label="Алгоритм Дейкстры", linewidth=2.2, markersize=3.2, zorder=5)
    if path_gr:
        xg, yg = meters_xy(path_gr)
        ax.plot(xg, yg, "--s", color="#FF7F0E", label="Жадный алгоритм", linewidth=2.0, markersize=3.2, zorder=6)
        if not greedy_success:
            ax.scatter([xg[-1]], [yg[-1]], s=60, color="#FF7F0E", marker="x", zorder=9, label="Остановка жадного алгоритма")
    if path_rl:
        xr, yr = meters_xy(path_rl)
        ax.plot(xr, yr, "-.^", color="#C000C0", label="Обученная политика", linewidth=2.2, markersize=3.4, zorder=7)

    sx_m, sy_m = start_point[0] * h, start_point[1] * h
    gx_m, gy_m = goal[0] * h, goal[1] * h
    ax.scatter([sx_m], [sy_m], s=90, marker="*", color="black", zorder=10, label="Старт")
    ax.scatter([gx_m], [gy_m], s=90, marker="*", color="white", edgecolor="black", linewidth=0.8, zorder=10, label="Цель")

    ax.set_title(f"Маршруты в ветровом поле: {scenario.ru_name}")
    ax.set_xlabel("Координата x, м")
    ax.set_ylabel("Координата y, м")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.legend(loc="best", fontsize=8)

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.10)
    cbar.set_label("Модуль скорости ветра, м/с")

    plt.tight_layout()
    save_figure_all_formats(fig, filename_pdf)
    plt.close(fig)


def plot_energy_boxplot_ru(scenario: Scenario, baseline_rows: List[dict], all_rl_rows: List[dict], filename_pdf: Path):
    E_dij = [as_float(r["E_dijkstra_J"]) for r in baseline_rows if as_int(r["dijkstra_success"]) == 1]
    E_gr = [as_float(r["E_greedy_J"]) for r in baseline_rows if as_int(r["greedy_success"]) == 1 and r["E_greedy_J"] not in ("", None)]
    E_rl = [as_float(r["E_rl_J"]) for r in all_rl_rows if as_int(r["rl_success"]) == 1 and r["E_rl_J"] not in ("", None)]

    data = []
    labels = []
    for arr, label in [
        (E_dij, "Дейкстра"),
        (E_gr, "Жадный"),
        (E_rl, "Обученная\nполитика"),
    ]:
        if arr:
            data.append(arr)
            labels.append(label)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title(f"Распределение энергозатрат: {scenario.ru_name}")
    ax.set_ylabel("Суммарные энергозатраты E, Дж")
    ax.set_xlabel("Метод построения маршрута")
    ax.grid(True, axis="y")
    plt.tight_layout()
    save_figure_all_formats(fig, filename_pdf)
    plt.close(fig)


def plot_summary_delta_ru(scenario_summaries: List[dict], filename_pdf: Path):
    if not scenario_summaries:
        return
    names = [s["scenario_ru"] for s in scenario_summaries]
    means = [float(s["delta_D_mean_percent"]) for s in scenario_summaries]
    stds = [float(s["delta_D_std_percent"]) for s in scenario_summaries]

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Отклонение энергозатрат от эталона ΔD, %")
    ax.set_xlabel("Ветровое поле")
    ax.set_title("Отклонение энергозатрат обученной политики от решения Дейкстры")
    ax.grid(True, axis="y")
    plt.tight_layout()
    save_figure_all_formats(fig, filename_pdf)
    plt.close(fig)


# =============================================================================
# 14. СВОДНЫЕ ТАБЛИЦЫ
# =============================================================================


def collect_run_summaries() -> List[dict]:
    rows = []
    for scenario_key in SCENARIOS_TO_RUN:
        scenario = SCENARIOS[scenario_key]
        for train_seed in TRAIN_SEEDS_ACTIVE:
            p = run_summary_path(scenario, train_seed)
            if p.exists():
                rows.append(load_json(p))
    rows.sort(key=lambda r: (SCENARIOS_TO_RUN.index(r["scenario"]), int(r["train_seed"])))
    return rows


def build_scenario_summaries(run_rows: List[dict]) -> List[dict]:
    out = []
    for scenario_key in SCENARIOS_TO_RUN:
        scenario = SCENARIOS[scenario_key]
        rows = [r for r in run_rows if r["scenario"] == scenario_key]
        if not rows:
            continue

        delta = np.array([float(r["delta_D_percent"]) for r in rows], dtype=float)
        rl_success_rates = np.array([float(r["rl_success_rate"]) for r in rows], dtype=float)
        E_rl_means = np.array([float(r["E_rl_mean_J"]) for r in rows], dtype=float)
        L_rl_means = np.array([float(r["L_rl_mean_m"]) for r in rows], dtype=float)
        improve = np.array([float(r["improve_rl_vs_greedy_percent"]) for r in rows], dtype=float)

        out.append({
            "scenario": scenario_key,
            "scenario_ru": scenario.ru_name,
            "n_training_runs": len(rows),
            "rl_success_rate_mean": float(np.mean(rl_success_rates)),
            "rl_success_rate_min": float(np.min(rl_success_rates)),
            "greedy_success_rate": float(rows[0]["greedy_success_rate"]),
            "E_rl_mean_over_training_runs_J": float(np.mean(E_rl_means)),
            "E_rl_std_between_training_runs_J": float(np.std(E_rl_means, ddof=1)) if len(rows) > 1 else 0.0,
            "L_rl_mean_over_training_runs_m": float(np.mean(L_rl_means)),
            "L_rl_std_between_training_runs_m": float(np.std(L_rl_means, ddof=1)) if len(rows) > 1 else 0.0,
            "delta_D_mean_percent": float(np.mean(delta)),
            "delta_D_std_percent": float(np.std(delta, ddof=1)) if len(rows) > 1 else 0.0,
            "delta_D_min_percent": float(np.min(delta)),
            "delta_D_max_percent": float(np.max(delta)),
            "improve_rl_vs_greedy_mean_percent": float(np.nanmean(improve)) if np.any(np.isfinite(improve)) else math.nan,
            "rl_fallback_total_all_runs": int(sum(int(r["rl_fallback_total"]) for r in rows)),
            "rl_revisit_total_all_runs": int(sum(int(r["rl_revisit_total"]) for r in rows)),
        })
    return out


# =============================================================================
# 15. СЛУЖЕБНАЯ ИНФОРМАЦИЯ О ВОСПРОИЗВОДИМОСТИ
# =============================================================================


def current_script_sha256() -> str:
    try:
        script_path = Path(__file__)
        data = script_path.read_bytes()
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return "недоступно"


def save_run_configuration() -> None:
    cfg = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": mpl.__version__,
        "script_sha256": current_script_sha256(),
        "quick_test": QUICK_TEST,
        "grid_W": W,
        "grid_H": H,
        "cell_size_m": CELL_SIZE_M,
        "goal": list(goal),
        "airspeed_mps": DRONE_AIRSPEED_CRUISE,
        "P_hover_W": P_HOVER_W,
        "K_drag_W_per_mps2": K_DRAG_W_PER_MPS2,
        "effective_power_W": P_HOVER_W + K_DRAG_W_PER_MPS2 * DRONE_AIRSPEED_CRUISE ** 2,
        "wind_edges_mps": WIND_EDGES.tolist(),
        "n_bins": N_BIN,
        "q_shape": [H, W, N_BIN, N_BIN, n_actions],
        "q_entries": int(H * W * N_BIN * N_BIN * n_actions),
        "q_memory_mb_float64": float(H * W * N_BIN * N_BIN * n_actions * 8 / 1_000_000.0),
        "alpha": ALPHA,
        "gamma": GAMMA,
        "n_episodes": N_EPISODES,
        "max_steps": MAX_STEPS,
        "eps_start": EPS_START,
        "eps_end": EPS_END,
        "training_seeds": TRAIN_SEEDS_ACTIVE,
        "eval_seed_start": EVAL_SEED_START,
        "n_eval_seeds": len(EVAL_SEEDS),
        "eval_seeds": EVAL_SEEDS,
        "scenarios": [asdict(SCENARIOS[k]) for k in SCENARIOS_TO_RUN],
        "reward_cfg": reward_cfg,
    }
    save_json(OUTPUT_DIR / "run_configuration.json", cfg)


# =============================================================================
# 16. ГЛАВНЫЙ ЦИКЛ
# =============================================================================


def ensure_q_loaded(scenario: Scenario, train_seed: int) -> None:
    global Q
    p = q_path(scenario.key, train_seed)
    if not p.exists():
        raise FileNotFoundError(f"Не найдена Q-таблица: {p}")
    Q = np.load(p)


def make_representative_route_figure(scenario: Scenario, train_seed: int, preferred_eval_seed: int) -> None:
    if not SAVE_ROUTE_FIGURES:
        return
    out_pdf = FIGURE_DIR / f"routes_{scenario.key}.pdf"
    if RESUME and out_pdf.exists():
        return

    ensure_q_loaded(scenario, train_seed)
    candidate_seeds = [preferred_eval_seed] + [s for s in EVAL_SEEDS if s != preferred_eval_seed]

    chosen = None
    for eval_seed in candidate_seeds:
        set_wind_field(scenario, eval_seed)
        start_point = eval_left_edge_start_from_seed(eval_seed)
        path_rl, _, _, _, rl_ok, _, _ = rl_policy_path(scenario, start_point)
        if rl_ok:
            path_dij, _, _, _, _ = dijkstra_shortest_path(scenario, start_point)
            path_gr, _, _, _, gr_ok = greedy_path(scenario, start_point)
            chosen = (eval_seed, start_point, path_dij, path_gr, path_rl, gr_ok)
            break

    if chosen is None:
        print(f"[рисунок пропущен] {scenario.key}: нет успешной RL-траектории")
        return

    eval_seed, start_point, path_dij, path_gr, path_rl, gr_ok = chosen
    # set_wind_field уже вызван для chosen seed в цикле; повторяем явно для надежности.
    set_wind_field(scenario, eval_seed)
    plot_wind_and_paths_ru(
        scenario, path_dij, path_gr, path_rl, start_point, gr_ok, out_pdf
    )
    print(f"[рисунок] {out_pdf.name} | тестовая реализация seed={eval_seed}")


def main():
    save_run_configuration()

    effective_power = P_HOVER_W + K_DRAG_W_PER_MPS2 * DRONE_AIRSPEED_CRUISE ** 2
    print("=" * 96)
    print("ФИНАЛЬНЫЙ ПРОГОН ДЛЯ СТАТЬИ")
    print(f"Режим быстрой проверки: {QUICK_TEST}")
    print(f"Сетка: {W} x {H}, шаг сетки: {CELL_SIZE_M:.1f} м")
    print(f"Собственная воздушная скорость: {DRONE_AIRSPEED_CRUISE:.1f} м/с")
    print(f"Эффективная мощность при этой скорости: {effective_power:.1f} Вт")
    print(f"Квантование ветра: N_BIN={N_BIN}, пороги={WIND_EDGES.tolist()} м/с")
    print(f"Размер Q-таблицы: {Q.size} значений, {Q.nbytes / 1_000_000.0:.4f} Мбайт")
    print(f"Эпизодов обучения: {N_EPISODES}")
    print(f"Training seeds: {TRAIN_SEEDS_ACTIVE}")
    print(f"Независимые тестовые seed: {EVAL_SEEDS[0]}...{EVAL_SEEDS[-1]} ({len(EVAL_SEEDS)} шт.)")
    print(f"Каталог результатов: {OUTPUT_DIR}")
    print("=" * 96)

    for scenario_key in SCENARIOS_TO_RUN:
        scenario = SCENARIOS[scenario_key]
        print("\n" + "#" * 96)
        print(f"СЦЕНАРИЙ: {scenario.ru_name} [{scenario.key}]")
        print("#" * 96)

        baseline_rows = evaluate_baselines(scenario)

        for train_seed in TRAIN_SEEDS_ACTIVE:
            summary_p = run_summary_path(scenario, train_seed)
            detail_p = rl_detail_path(scenario, train_seed)
            final_q = q_path(scenario.key, train_seed)

            if RESUME and summary_p.exists() and detail_p.exists() and final_q.exists():
                print(f"[пропуск] полностью готово: {scenario.key}, train_seed={train_seed}")
                continue

            print("\n" + "-" * 96)
            print(f"Обучение: {scenario.key}, train_seed={train_seed}")

            # Если финальная Q уже есть, повторно не обучаем.
            if RESUME and final_q.exists():
                ensure_q_loaded(scenario, train_seed)
                train_info = {"train_seconds": math.nan, "train_success_rate": math.nan}
                print(f"[загрузка Q] {final_q.name}")
            else:
                train_info = train_q(scenario, train_seed)

            ensure_q_loaded(scenario, train_seed)
            summary = evaluate_rl_policy(scenario, train_seed, baseline_rows, train_info)
            print(
                f"[результат] {scenario.key} | train_seed={train_seed} | "
                f"RL={summary['rl_successes']}/{summary['n_eval']} | "
                f"ΔD={summary['delta_D_percent']:.3f}% | "
                f"жадный={summary['greedy_successes']}/{summary['n_eval']} | "
                f"повторных посещений={summary['rl_revisit_total']}"
            )

        # Один маршрутный рисунок на сценарий строим по политике train_seed=0.
        if FIGURE_TRAIN_SEED in TRAIN_SEEDS_ACTIVE:
            make_representative_route_figure(scenario, FIGURE_TRAIN_SEED, FIGURE_EVAL_SEED)

        # Коробчатая диаграмма после завершения всех активных training seed.
        if SAVE_SCENARIO_BOXPLOTS:
            all_rl_rows = []
            for ts in TRAIN_SEEDS_ACTIVE:
                p = rl_detail_path(scenario, ts)
                if p.exists():
                    all_rl_rows.extend(read_csv(p))
            if all_rl_rows:
                box_pdf = FIGURE_DIR / f"energy_boxplot_{scenario.key}.pdf"
                plot_energy_boxplot_ru(scenario, baseline_rows, all_rl_rows, box_pdf)

    run_rows = collect_run_summaries()
    if run_rows:
        write_csv(SUMMARY_DIR / "training_runs_summary.csv", run_rows)

    scenario_rows = build_scenario_summaries(run_rows)
    if scenario_rows:
        write_csv(SUMMARY_DIR / "scenario_summary.csv", scenario_rows)

    if SAVE_SUMMARY_FIGURE and scenario_rows:
        plot_summary_delta_ru(scenario_rows, FIGURE_DIR / "delta_D_summary.pdf")

    print("\n" + "=" * 120)
    print("ИТОГОВАЯ СВОДКА ДЛЯ КОПИРОВАНИЯ")
    print("=" * 120)
    if scenario_rows:
        headers = [
            "scenario", "n_training_runs", "rl_success_rate_mean", "greedy_success_rate",
            "delta_D_mean_percent", "delta_D_std_percent", "E_rl_mean_over_training_runs_J",
            "L_rl_mean_over_training_runs_m", "improve_rl_vs_greedy_mean_percent",
            "rl_revisit_total_all_runs",
        ]
        print("\t".join(headers))
        for r in scenario_rows:
            print("\t".join(str(r.get(h, "")) for h in headers))
    print("=" * 120)
    print(f"Подробные результаты сохранены в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
