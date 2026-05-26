import sys
import random
import copy
import time
from collections import deque

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("pip install torch numpy")
    sys.exit(1)

GRID_WIDTH, GRID_HEIGHT = 5, 6
NUM_STONES = 6
CELL_COUNT = GRID_WIDTH * GRID_HEIGHT
NUM_ACTIONS = CELL_COUNT + CELL_COUNT * 4
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MODEL_PATH = "bolotudu_strong.pth"
BUFFER_SIZE = 150_000
BATCH_SIZE = 128
GAMMA = 0.99
LR = 5e-4
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 40_000
TARGET_UPDATE = 1000
TRAIN_EVERY = 2
SEARCH_DEPTH = 4  # глубина при игре с человеком


# ===================== Среда =====================
class BolotuduEnv:
    def __init__(self):
        self.reset()

    def copy(self):
        e = BolotuduEnv()
        e.board = [row[:] for row in self.board]
        e.current_player = self.current_player
        e.stage = self.stage
        e.stones_count = self.stones_count[:]
        e.remaining_pairs = self.remaining_pairs[:]
        e.stones_to_place = self.stones_to_place
        return e

    def reset(self, start_player=0):
        self.board = [[None] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
        self.current_player = start_player
        self.stage = 1
        self.stones_count = [0, 0]
        self.remaining_pairs = [NUM_STONES, NUM_STONES]
        self.stones_to_place = 2
        return self.get_state()

    def get_state(self):
        flat = []
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                v = self.board[r][c]
                flat.append(0.0 if v is None else (1.0 if v == 0 else -1.0))
        flat += [
            float(self.stage),
            float(self.current_player),
            self.stones_to_place / 2.0,
            self.remaining_pairs[0] / NUM_STONES,
            self.remaining_pairs[1] / NUM_STONES,
            self.stones_count[0] / 12.0,
            self.stones_count[1] / 12.0,
        ]
        return np.array(flat, dtype=np.float32)

    @staticmethod
    def rc_to_idx(r, c):
        return r * GRID_WIDTH + c

    @staticmethod
    def idx_to_rc(i):
        return i // GRID_WIDTH, i % GRID_WIDTH

    def winner(self):
        if self.stones_count[0] <= 2:
            return 1
        if self.stones_count[1] <= 2:
            return 0
        return None

    def legal_actions(self):
        acts = []
        if self.stage == 1:
            if self.remaining_pairs[self.current_player] <= 0:
                return acts
            for r in range(GRID_HEIGHT):
                for c in range(GRID_WIDTH):
                    if self.board[r][c] is None and self._can_place(r, c):
                        acts.append(self.rc_to_idx(r, c))
        else:
            for fr in range(GRID_HEIGHT):
                for fc in range(GRID_WIDTH):
                    if self.board[fr][fc] != self.current_player:
                        continue
                    for d, (dr, dc) in enumerate(DIRS):
                        tr, tc = fr + dr, fc + dc
                        if 0 <= tr < GRID_HEIGHT and 0 <= tc < GRID_WIDTH and self.board[tr][tc] is None:
                            acts.append(CELL_COUNT + self.rc_to_idx(fr, fc) * 4 + d)
        return acts

    def _can_place(self, row, col):
        self.board[row][col] = self.current_player
        ok = not self._line_at(row, col, self.board)
        self.board[row][col] = None
        return ok

    def _line_at(self, row, col, board):
        p = board[row][col]
        if p is None:
            return False
        if self._count_h(row, col, board) >= 3:
            return True
        return self._count_v(row, col, board) >= 3

    def _count_h(self, row, col, board):
        p = board[row][col]
        cnt = 1
        c = col - 1
        while c >= 0 and board[row][c] == p:
            cnt += 1
            c -= 1
        c = col + 1
        while c < GRID_WIDTH and board[row][c] == p:
            cnt += 1
            c += 1
        return cnt

    def _count_v(self, row, col, board):
        p = board[row][col]
        cnt = 1
        r = row - 1
        while r >= 0 and board[r][col] == p:
            cnt += 1
            r -= 1
        r = row + 1
        while r < GRID_HEIGHT and board[r][col] == p:
            cnt += 1
            r += 1
        return cnt

    def _line_dir(self, row, col):
        if self._count_h(row, col, self.board) >= 3:
            return "h"
        if self._count_v(row, col, self.board) >= 3:
            return "v"
        return None

    def _adjacent_opp(self, row, col, direc):
        p = self.board[row][col]
        opp = 1 - p
        out = []
        if direc == "h":
            left = col
            while left > 0 and self.board[row][left - 1] == p:
                left -= 1
            right = col
            while right < GRID_WIDTH - 1 and self.board[row][right + 1] == p:
                right += 1
            if left > 0 and self.board[row][left - 1] == opp:
                out.append((row, left - 1))
            if right < GRID_WIDTH - 1 and self.board[row][right + 1] == opp:
                out.append((row, right + 1))
        else:
            top = row
            while top > 0 and self.board[top - 1][col] == p:
                top -= 1
            bottom = row
            while bottom < GRID_HEIGHT - 1 and self.board[bottom + 1][col] == p:
                bottom += 1
            if top > 0 and self.board[top - 1][col] == opp:
                out.append((top - 1, col))
            if bottom < GRID_HEIGHT - 1 and self.board[bottom + 1][col] == opp:
                out.append((bottom + 1, col))
        return out

    def step(self, action):
        w = self.winner()
        if w is not None:
            return self.get_state(), 0.0, True

        old_board = [row[:] for row in self.board]
        reward = -0.02

        if self.stage == 1:
            if action >= CELL_COUNT:
                return self.get_state(), -2.0, False
            r, c = self.idx_to_rc(action)
            if self.board[r][c] is not None or not self._can_place(r, c):
                return self.get_state(), -2.0, False
            self.board[r][c] = self.current_player
            self.stones_count[self.current_player] += 1
            self.stones_to_place -= 1
            if self.stones_to_place == 0:
                self.remaining_pairs[self.current_player] -= 1
                self.stones_to_place = 2
                self.current_player = 1 - self.current_player
            if sum(self.remaining_pairs) == 0:
                self.stage = 2
        else:
            if action < CELL_COUNT:
                return self.get_state(), -2.0, False
            mid = action - CELL_COUNT
            fi = mid // 4
            d = mid % 4
            fr, fc = self.idx_to_rc(fi)
            dr, dc = DIRS[d]
            tr, tc = fr + dr, fc + dc
            if self.board[fr][fc] != self.current_player:
                return self.get_state(), -2.0, False
            if not (0 <= tr < GRID_HEIGHT and 0 <= tc < GRID_WIDTH) or self.board[tr][tc] is not None:
                return self.get_state(), -2.0, False

            self.board[tr][tc] = self.board[fr][fc]
            self.board[fr][fc] = None

            if self._line_at(tr, tc, self.board) and not self._line_at(tr, tc, old_board):
                adj = self._adjacent_opp(tr, tc, self._line_dir(tr, tc))
                if adj:
                    rr, cc = adj[0]
                    self.board[rr][cc] = None
                    self.stones_count[1 - self.current_player] -= 1
                    reward += 3.0

            self.current_player = 1 - self.current_player

        w = self.winner()
        done = w is not None
        if done:
            if w == self.current_player:
                reward += 15.0
            else:
                reward -= 15.0
        return self.get_state(), reward, done


def evaluate_board(env, player):
    """Оценка позиции для поиска (сильная игра с человеком)."""
    w = env.winner()
    if w == player:
        return 10000
    if w == 1 - player:
        return -10000
    opp = 1 - player
    score = (env.stones_count[player] - env.stones_count[opp]) * 8.0
    for r in range(GRID_HEIGHT):
        for c in range(GRID_WIDTH):
            if env.board[r][c] == player:
                if env._count_h(r, c, env.board) == 2:
                    score += 4
                if env._count_v(r, c, env.board) == 2:
                    score += 4
            elif env.board[r][c] == opp:
                if env._count_h(r, c, env.board) == 2:
                    score -= 5
                if env._count_v(r, c, env.board) == 2:
                    score -= 5
    if env.stage == 1:
        score += env.remaining_pairs[player] * 0.5
    return score


def negamax(env, depth, player, ai_player, alpha=-1e9, beta=1e9):
    w = env.winner()
    if w is not None:
        return 10000 if w == ai_player else -10000
    if depth == 0:
        cur = env.current_player
        val = evaluate_board(env, ai_player)
        return val if cur == ai_player else -val

    legal = env.legal_actions()
    if not legal:
        return evaluate_board(env, ai_player)

    best = -1e9
    for a in legal:
        child = env.copy()
        child.step(a)
        val = -negamax(child, depth - 1, player, ai_player, -beta, -alpha)
        best = max(best, val)
        alpha = max(alpha, val)
        if alpha >= beta:
            break
    return best


# ===================== DQN + буфер =====================
class ReplayBuffer:
    def __init__(self, cap=BUFFER_SIZE):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, n):
        b = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*b)
        return (
            np.stack(s),
            np.array(a, np.int64),
            np.array(r, np.float32),
            np.stack(ns),
            np.array(d, np.float32),
        )

    def __len__(self):
        return len(self.buf)


class DQN(nn.Module):
    def __init__(self, dim, n_act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_act),
        )

    def forward(self, x):
        return self.net(x)


class RLAgent:
    def __init__(self, state_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = DQN(state_dim, NUM_ACTIONS).to(self.device)
        self.target = DQN(state_dim, NUM_ACTIONS).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.opt = optim.Adam(self.policy.parameters(), lr=LR)
        self.buffer = ReplayBuffer()
        self.steps = 0

    def eps(self):
        t = min(1.0, self.steps / EPS_DECAY)
        return EPS_START + (EPS_END - EPS_START) * t

    def q_values(self, state, legal):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy(s).squeeze(0).cpu().numpy()
        return {a: q[a] for a in legal}

    def pick_train(self, state, legal):
        if random.random() < self.eps():
            return random.choice(legal)
        q = self.q_values(state, legal)
        return max(q, key=q.get)

    def learn(self):
        if len(self.buffer) < BATCH_SIZE * 20:
            return
        s, a, r, ns, d = self.buffer.sample(BATCH_SIZE)
        s = torch.tensor(s, device=self.device)
        a = torch.tensor(a, device=self.device)
        r = torch.tensor(r, device=self.device)
        ns = torch.tensor(ns, device=self.device)
        d = torch.tensor(d, device=self.device)
        q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.target(ns).max(1)[0]
            tgt = r + GAMMA * nq * (1 - d)
        loss = nn.functional.smooth_l1_loss(q, tgt)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def save(self):
        torch.save(self.policy.state_dict(), MODEL_PATH)

    def load(self):
        self.policy.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.target.load_state_dict(self.policy.state_dict())


def strong_pick(env, rl_agent, ai_player=1, depth=SEARCH_DEPTH):
    """Лучший ход: перебор + подсказка нейросети."""
    legal = env.legal_actions()
    if not legal:
        return None
    if len(legal) == 1:
        return legal[0]

    q = rl_agent.q_values(env.get_state(), legal)
    ranked = sorted(legal, key=lambda a: q.get(a, 0), reverse=True)[: min(12, len(legal))]

    best_a, best_v = None, -1e9
    for a in ranked:
        child = env.copy()
        child.step(a)
        v = negamax(child, depth - 1, ai_player, ai_player)
        if v > best_v:
            best_v, best_a = v, a
    return best_a


def train(episodes=40_000):
    env = BolotuduEnv()
    dim = len(env.reset())
    agent = RLAgent(dim)
    t0 = time.time()
    print("=== Обучение DQN (опыт в ReplayBuffer) ===")
    print("Эпизодов:", episodes)
    print("Устройство:", agent.device)
    print("Ориентир по времени: 40k эпизодов ~ 1-3 часа")
    print("Можно оставить ПК включённым и уйти.\n")

    for ep in range(1, episodes + 1):
        env.reset()
        done = False
        moves = 0
        while not done and moves < 250:
            legal = env.legal_actions()
            if not legal:
                break
            s = env.get_state()
            a = agent.pick_train(s, legal)
            ns, r, done = env.step(a)
            agent.buffer.push(s, a, r, ns, done)
            agent.steps += 1
            moves += 1
            if agent.steps % TRAIN_EVERY == 0:
                agent.learn()
            if agent.steps % TARGET_UPDATE == 0:
                agent.target.load_state_dict(agent.policy.state_dict())

        if ep % 2000 == 0:
            elapsed = time.time() - t0
            eps = agent.eps()
            per_ep = elapsed / ep
            left = per_ep * (episodes - ep)
            print(
                f"Эпизод {ep}/{episodes} | буфер {len(agent.buffer)} | "
                f"eps {eps:.2f} | прошло {elapsed/60:.1f} мин | осталось ~{left/60:.1f} мин"
            )

    agent.save()
    print(f"\nГотово за {(time.time()-t0)/60:.1f} мин. Модель: {MODEL_PATH}")
    print("Запуск игры: python bolotudu_strong.py play")


# ===================== GUI: ты синий (0), ИИ красный (1) =====================
def play():
    import tkinter as tk
    from tkinter import messagebox

    CELL = 60
    COLORS = ["#4287f5", "#f54242"]
    HUMAN, AI = 0, 1

    env = BolotuduEnv()
    dim = len(env.reset())
    agent = RLAgent(dim)
    try:
        agent.load()
    except FileNotFoundError:
        print("Сначала: python bolotudu_strong.py train")
        return

    env.reset(0)
    selected = None

    root = tk.Tk()
    root.title("Болотуду — ты vs сильный ИИ")
    canvas = tk.Canvas(root, width=GRID_WIDTH * CELL, height=GRID_HEIGHT * CELL, bg="#8B4513")
    canvas.pack(pady=8)
    lbl = tk.Label(root, text="", font=("Arial", 11))
    lbl.pack()

    def draw():
        canvas.delete("all")
        for i in range(GRID_WIDTH + 1):
            x = i * CELL
            canvas.create_line(x, 0, x, GRID_HEIGHT * CELL, fill="#D2B48C", width=2)
        for i in range(GRID_HEIGHT + 1):
            y = i * CELL
            canvas.create_line(0, y, GRID_WIDTH * CELL, y, fill="#D2B48C", width=2)
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                if env.board[r][c] is not None:
                    x, y = c * CELL + CELL // 2, r * CELL + CELL // 2
                    col = COLORS[env.board[r][c]]
                    w = 4 if selected == (r, c) else 1
                    canvas.create_oval(x - 20, y - 20, x + 20, y + 20, fill=col, outline="yellow" if w == 4 else "black", width=w)
        w = env.winner()
        if w is not None:
            lbl.config(text="Конец игры")
        else:
            who = "Ты" if env.current_player == HUMAN else "ИИ думает..."
            st = "расстановка" if env.stage == 1 else "ходы"
            lbl.config(text=f"{who} | фаза: {st} | камни: {env.stones_count}")

    def end_check():
        w = env.winner()
        if w == HUMAN:
            messagebox.showinfo("Итог", "Ты победил!")
            return True
        if w == AI:
            messagebox.showinfo("Итог", "Победил ИИ")
            return True
        return False

    def ai_move():
        if env.current_player != AI or end_check():
            return
        lbl.config(text="ИИ думает...")
        root.update()
        a = strong_pick(env, agent, AI, depth=SEARCH_DEPTH)
        if a is not None:
            env.step(a)
        draw()
        end_check()

    def human_click(r, c):
        nonlocal selected
        if env.current_player != HUMAN:
            return
        legal = env.legal_actions()

        if env.stage == 1:
            a = env.rc_to_idx(r, c)
            if a in legal:
                env.step(a)
                selected = None
                draw()
                if end_check():
                    return
                ai_move()
            return

        if env.board[r][c] == HUMAN:
            selected = (r, c)
            draw()
            return

        if selected is None:
            return
        fr, fc = selected
        action = None
        for a in legal:
            if a < CELL_COUNT:
                continue
            mid = a - CELL_COUNT
            fi, d = mid // 4, mid % 4
            frr, fcc = env.idx_to_rc(fi)
            if frr != fr or fcc != fc:
                continue
            dr, dc = DIRS[d]
            if fr + dr == r and fc + dc == c:
                action = a
                break
        if action is not None:
            env.step(action)
            selected = None
            draw()
            if end_check():
                return
            ai_move()

    def on_click(ev):
        col, row = ev.x // CELL, ev.y // CELL
        if 0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH:
            human_click(row, col)

    canvas.bind("<Button-1>", on_click)
    draw()
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("train [эпизоды]  |  play")
        sys.exit(0)
    cmd = sys.argv[1].lower()
    if cmd == "train":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 40_000
        train(n)
    elif cmd == "play":
        play()
    else:
        print("Команды: train, play")