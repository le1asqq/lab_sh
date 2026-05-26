import tkinter as tk
from tkinter import messagebox
import sqlite3
import os
import math

from bolotudu_strong import (
    BolotuduEnv,
    RLAgent,
    smart_pick,
    MODEL_PATH,
    CELL_COUNT,
    DIRS,
    GRID_WIDTH,
    GRID_HEIGHT,
    NUM_STONES,
)

CELL_SIZE = 60
PLAYER_COLORS = ["#4287f5", "#f54242"]
DATABASE_FILE = "bolotudu.db"
BOARD_COLOR = "#8B4513"
GRID_COLOR = "#D2B48C"


class BolotuduGame:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Болотуду")
        self.window.geometry("800x600")
        self.window.configure(bg="#2C3E50")

        self.create_database()

        try:
            self.window.iconbitmap("icon.ico")
        except Exception:
            pass

        self.board = [[None] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
        self.current_player = 0
        self.stage = 1
        self.stones_count = [0, 0]
        self.remaining_pairs = [NUM_STONES, NUM_STONES]
        self.stones_to_place = 2
        self.selected_stone = None
        self.current_user = None

        self.vs_ai = False
        self.env = None
        self.rl_agent = None
        self.human_player = 0
        self.ai_player = 1
        self.ai_selected = None

        self.canvas = None
        self.info_label = None

        self.show_login_screen()
        self.window.mainloop()

    def create_database(self):
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            games_played INTEGER DEFAULT 0,
            games_won INTEGER DEFAULT 0
        )
        """
        )
        conn.commit()
        conn.close()

    def show_login_screen(self):
        for widget in self.window.winfo_children():
            widget.destroy()

        login_frame = tk.Frame(self.window, bg="#ffffff", padx=40, pady=40)
        login_frame.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(login_frame, text="Вход в игру", font=("Arial", 24, "bold"), bg="#ffffff").pack(
            pady=(0, 30)
        )

        tk.Label(login_frame, text="Имя пользователя:", bg="#ffffff", font=("Arial", 12)).pack()
        username_entry = tk.Entry(login_frame, font=("Arial", 12))
        username_entry.pack(pady=(0, 10))

        tk.Label(login_frame, text="Пароль:", bg="#ffffff", font=("Arial", 12)).pack()
        password_entry = tk.Entry(login_frame, show="*", font=("Arial", 12))
        password_entry.pack(pady=(0, 20))

        button_style = {"font": ("Arial", 12), "width": 20, "pady": 8}

        def login():
            username = username_entry.get()
            password = password_entry.get()
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE username=? AND password=?", (username, password)
            )
            user = cursor.fetchone()
            conn.close()
            if user:
                self.current_user = username
                messagebox.showinfo("Успех", f"Добро пожаловать, {username}!")
                self.show_main_menu()
            else:
                messagebox.showerror("Ошибка", "Неверное имя пользователя или пароль")

        tk.Button(login_frame, text="Войти", command=login, bg="#4CAF50", fg="white", **button_style).pack(
            pady=5
        )
        tk.Button(
            login_frame,
            text="Регистрация",
            command=self.show_register_screen,
            bg="#2196F3",
            fg="white",
            **button_style,
        ).pack(pady=5)

    def show_register_screen(self):
        for widget in self.window.winfo_children():
            widget.destroy()

        register_frame = tk.Frame(self.window, bg="#ffffff", padx=40, pady=40)
        register_frame.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(register_frame, text="Регистрация", font=("Arial", 24, "bold"), bg="#ffffff").pack(
            pady=(0, 30)
        )

        tk.Label(
            register_frame, text="Придумайте имя пользователя:", bg="#ffffff", font=("Arial", 12)
        ).pack()
        username_entry = tk.Entry(register_frame, font=("Arial", 12))
        username_entry.pack(pady=(0, 10))

        tk.Label(register_frame, text="Придумайте пароль:", bg="#ffffff", font=("Arial", 12)).pack()
        password_entry = tk.Entry(register_frame, show="*", font=("Arial", 12))
        password_entry.pack(pady=(0, 10))

        tk.Label(register_frame, text="Подтвердите пароль:", bg="#ffffff", font=("Arial", 12)).pack()
        confirm_entry = tk.Entry(register_frame, show="*", font=("Arial", 12))
        confirm_entry.pack(pady=(0, 20))

        button_style = {"font": ("Arial", 12), "width": 20, "pady": 8}

        def register():
            username = username_entry.get()
            password = password_entry.get()
            confirm = confirm_entry.get()
            if not username or not password:
                messagebox.showerror("Ошибка", "Все поля должны быть заполнены")
                return
            if password != confirm:
                messagebox.showerror("Ошибка", "Пароли не совпадают")
                return
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)", (username, password)
                )
                conn.commit()
                messagebox.showinfo("Успех", "Регистрация успешна!")
                self.show_login_screen()
            except sqlite3.IntegrityError:
                messagebox.showerror("Ошибка", "Такое имя пользователя уже существует")
            finally:
                conn.close()

        tk.Button(
            register_frame, text="Зарегистрироваться", command=register, bg="#4CAF50", fg="white", **button_style
        ).pack(pady=5)
        tk.Button(
            register_frame, text="Назад", command=self.show_login_screen, bg="#f44336", fg="white", **button_style
        ).pack(pady=5)

    def reset_game_state(self):
        self.board = [[None] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
        self.current_player = 0
        self.stage = 1
        self.stones_count = [0, 0]
        self.remaining_pairs = [NUM_STONES, NUM_STONES]
        self.stones_to_place = 2
        self.selected_stone = None
        self.ai_selected = None
        self.env = None

    def show_main_menu(self):
        self.vs_ai = False
        self.reset_game_state()

        for widget in self.window.winfo_children():
            widget.destroy()

        main_frame = tk.Frame(self.window, bg="#f0f0f0")
        main_frame.pack(expand=True, fill=tk.BOTH)

        menu_frame = tk.Frame(main_frame, bg="#ffffff", padx=40, pady=40)
        menu_frame.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(menu_frame, text="Болотуду", font=("Arial", 24, "bold"), bg="#ffffff").pack(pady=(0, 30))

        button_style = {"font": ("Arial", 12), "width": 25, "pady": 10}

        tk.Button(
            menu_frame,
            text="Начать игру",
            command=lambda: self.start_game(vs_ai=False),
            bg="#4CAF50",
            fg="white",
            **button_style,
        ).pack(pady=5)

        tk.Button(
            menu_frame,
            text="Играть против ИИ",
            command=lambda: self.start_game(vs_ai=True),
            bg="#2196F3",
            fg="white",
            **button_style,
        ).pack(pady=5)

        tk.Button(menu_frame, text="Выход", command=self.window.quit, bg="#f44336", fg="white", **button_style).pack(
            pady=5
        )

    def start_game(self, vs_ai=False):
        self.vs_ai = vs_ai
        self.reset_game_state()

        if vs_ai:
            model_full = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_PATH)
            if not os.path.isfile(model_full):
                model_full = MODEL_PATH
            if not os.path.isfile(model_full):
                messagebox.showerror(
                    "Ошибка",
                    "Модель ИИ не найдена (bolotudu_strong.pth).\n"
                    "Обучите ИИ отдельно через терминал или кнопку в меню.",
                )
                self.show_main_menu()
                return
            try:
                self.env = BolotuduEnv()
                self.env.reset(0)
                dim = len(self.env.get_state())
                self.rl_agent = RLAgent(dim)
                self.rl_agent.load()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить ИИ:\n{e}")
                self.show_main_menu()
                return

        self.setup_game_board()

    def get_board(self):
        if self.vs_ai and self.env:
            return self.env.board
        return self.board

    def get_current_player(self):
        if self.vs_ai and self.env:
            return self.env.current_player
        return self.current_player

    def get_stage(self):
        if self.vs_ai and self.env:
            return self.env.stage
        return self.stage

    def draw_board(self):
        self.canvas.delete("all")
        board = self.get_board()
        current = self.get_current_player()
        stage = self.get_stage()

        for i in range(GRID_WIDTH + 1):
            x = i * CELL_SIZE
            self.canvas.create_line(x, 0, x, GRID_HEIGHT * CELL_SIZE, fill=GRID_COLOR, width=2)
        for i in range(GRID_HEIGHT + 1):
            y = i * CELL_SIZE
            self.canvas.create_line(0, y, GRID_WIDTH * CELL_SIZE, y, fill=GRID_COLOR, width=2)

        sel = self.ai_selected if self.vs_ai else self.selected_stone

        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                if board[row][col] is not None:
                    x = col * CELL_SIZE + CELL_SIZE // 2
                    y = row * CELL_SIZE + CELL_SIZE // 2
                    color = PLAYER_COLORS[board[row][col]]
                    outline = "yellow" if (row, col) == sel else "black"
                    width = 3 if (row, col) == sel else 1
                    self.canvas.create_oval(
                        x - CELL_SIZE // 3,
                        y - CELL_SIZE // 3,
                        x + CELL_SIZE // 3,
                        y + CELL_SIZE // 3,
                        fill=color,
                        outline=outline,
                        width=width,
                    )

        if self.vs_ai:
            w = self.env.winner()
            sc = self.env.stones_count
            if w is not None:
                txt = "Вы победили!" if w == self.human_player else "Победил ИИ!"
                self.info_label.config(text=f"{txt} | камни: {sc[0]} / {sc[1]}")
                return
            elif current == self.human_player:
                if stage == 1:
                    hint = "клик по пустой клетке"
                else:
                    hint = "1) свой камень  2) соседняя пустая"
                self.info_label.config(
                    text=f"Ваш ход (синий) | {hint} | камни: {sc[0]} / {sc[1]}"
                )
            else:
                self.info_label.config(text=f"Ход ИИ... | камни: {sc[0]} / {sc[1]}")
        else:
            player_text = f"игрока {current + 1}"
            stones_text = f" (камни: {self.stones_count[current]})"
            self.info_label.config(
                text=f"Ход {player_text} ({PLAYER_COLORS[current]}){stones_text}"
            )

    def setup_game_board(self):
        for widget in self.window.winfo_children():
            widget.destroy()

        self.canvas = tk.Canvas(
            self.window,
            width=GRID_WIDTH * CELL_SIZE,
            height=GRID_HEIGHT * CELL_SIZE,
            bg=BOARD_COLOR,
        )
        self.canvas.pack(pady=20)

        self.info_label = tk.Label(self.window, text="", font=("Arial", 12))
        self.info_label.pack(pady=10)

        tk.Button(
            self.window,
            text="Назад в меню",
            command=self.show_main_menu,
            bg="#f44336",
            fg="white",
            font=("Arial", 12),
        ).pack(pady=10)

        self.canvas.bind("<Button-1>", self.handle_click)
        self.draw_board()

        if self.vs_ai and self.env.current_player == self.ai_player:
            self.window.after(300, self.ai_move)

    def record_game_result(self, human_won):
        if not self.current_user:
            return
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        if human_won:
            cursor.execute(
                "UPDATE users SET games_played = games_played + 1, games_won = games_won + 1 WHERE username=?",
                (self.current_user,),
            )
        else:
            cursor.execute(
                "UPDATE users SET games_played = games_played + 1 WHERE username=?",
                (self.current_user,),
            )
        conn.commit()
        conn.close()

    def finish_ai_game(self, winner):
        human_won = winner == self.human_player
        self.record_game_result(human_won)
        if human_won:
            messagebox.showinfo("Конец игры", "Ты победил!")
        else:
            messagebox.showinfo("Конец игры", "Победил ИИ!")
        self.show_main_menu()

    def ai_move(self):
        if not self.vs_ai or not self.env or not self.rl_agent:
            return
        if self.env.current_player != self.ai_player:
            return
        w = self.env.winner()
        if w is not None:
            self.draw_board()
            self.finish_ai_game(w)
            return

        legal = self.env.legal_actions()
        if not legal:
            return

        action = smart_pick(self.env, self.rl_agent, self.ai_player)
        if action is not None:
            self.env.step(action)

        self.ai_selected = None
        self.draw_board()

        w = self.env.winner()
        if w is not None:
            self.finish_ai_game(w)
            return

        # В расстановке ИИ ставит 2 камня подряд — продолжаем ход
        if self.env.current_player == self.ai_player and self.env.legal_actions():
            self.window.after(80, self.ai_move)

    def handle_click(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if not (0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH):
            return

        if self.vs_ai:
            self.handle_click_ai(row, col)
            return

        if self.stage == 1:
            self.place_stone(row, col)
        else:
            self.handle_move(row, col)

    def handle_click_ai(self, row, col):
        if self.env.current_player != self.human_player:
            return
        w = self.env.winner()
        if w is not None:
            self.finish_ai_game(w)
            return

        legal = self.env.legal_actions()

        if self.env.stage == 1:
            a = self.env.rc_to_idx(row, col)
            if a not in legal:
                messagebox.showerror("Ошибка", "Недопустимый ход!")
                return
            self.env.step(a)
            self.ai_selected = None
            self.draw_board()
            w = self.env.winner()
            if w is not None:
                self.finish_ai_game(w)
                return
            if self.env.current_player == self.ai_player:
                self.window.after(80, self.ai_move)
            return

        if self.env.board[row][col] == self.human_player:
            self.ai_selected = (row, col)
            self.draw_board()
            return

        if self.ai_selected is None:
            return

        fr, fc = self.ai_selected
        action = None
        for a in legal:
            if a < CELL_COUNT:
                continue
            mid = a - CELL_COUNT
            fi, d = mid // 4, mid % 4
            frr, fcc = self.env.idx_to_rc(fi)
            if frr != fr or fcc != fc:
                continue
            dr, dc = DIRS[d]
            if fr + dr == row and fc + dc == col:
                action = a
                break

        if action is None:
            messagebox.showerror("Ошибка", "Недопустимый ход!")
            return

        self.env.step(action)
        self.ai_selected = None
        self.draw_board()
        w = self.env.winner()
        if w is not None:
            self.finish_ai_game(w)
            return
        if self.env.current_player == self.ai_player:
            self.window.after(80, self.ai_move)

    # --- Логика для 2 игроков (как в курсовой) ---
    def place_stone(self, row, col):
        if self.board[row][col] is None and self.remaining_pairs[self.current_player] > 0:
            if self.check_no_three_in_row(row, col):
                self.board[row][col] = self.current_player
                self.stones_count[self.current_player] += 1
                self.stones_to_place -= 1
                if self.stones_to_place == 0:
                    self.remaining_pairs[self.current_player] -= 1
                    self.stones_to_place = 2
                    self.next_turn()
                if sum(self.remaining_pairs) == 0:
                    self.stage = 2
                    messagebox.showinfo("Информация", "Начинается фаза перемещения камней!")
                self.draw_board()
            else:
                messagebox.showerror("Ошибка", "Недопустимый ход!")
        else:
            messagebox.showerror("Ошибка", "Недопустимый ход!")

    def handle_move(self, row, col):
        if self.selected_stone is None:
            if self.board[row][col] is not None and self.board[row][col] == self.current_player:
                self.selected_stone = (row, col)
                self.draw_board()
        else:
            if self.is_valid_move(self.selected_stone[0], self.selected_stone[1], row, col):
                old_row, old_col = self.selected_stone
                old_board = [r[:] for r in self.board]
                self.board[row][col] = self.board[old_row][old_col]
                self.board[old_row][old_col] = None
                self.selected_stone = None
                line_info = self.check_for_line(row, col)
                if line_info[0]:
                    adjacent = self.get_adjacent_opponent_stones(row, col, line_info[1])
                    if adjacent:
                        old_line = self.check_for_line_at_position(row, col, old_board)
                        if not old_line[0]:
                            rr, cc = adjacent[0]
                            self.remove_stone(rr, cc)
                if self.stones_count[1 - self.current_player] <= 2:
                    winner = "Первый игрок" if self.current_player == 0 else "Второй игрок"
                    messagebox.showinfo("Конец игры", f"{winner} победил!")
                    self.show_main_menu()
                    return
                self.next_turn()
            else:
                self.selected_stone = None
                self.draw_board()
                messagebox.showerror("Ошибка", "Недопустимый ход!")

    def remove_stone(self, row, col):
        x = col * CELL_SIZE + CELL_SIZE // 2
        y = row * CELL_SIZE + CELL_SIZE // 2
        stone_color = PLAYER_COLORS[self.board[row][col]]
        for _ in range(3):
            self.canvas.create_oval(
                x - CELL_SIZE // 3,
                y - CELL_SIZE // 3,
                x + CELL_SIZE // 3,
                y + CELL_SIZE // 3,
                fill="red",
                outline="yellow",
                width=3,
            )
            self.window.update()
            self.window.after(100)
            self.canvas.create_oval(
                x - CELL_SIZE // 3,
                y - CELL_SIZE // 3,
                x + CELL_SIZE // 3,
                y + CELL_SIZE // 3,
                fill=stone_color,
                outline="black",
                width=1,
            )
            self.window.update()
            self.window.after(100)
        for i in range(10):
            size = CELL_SIZE // 3 * (10 - i) // 10
            angle = i * 36
            offset_x = size * 0.2 * math.cos(math.radians(angle))
            offset_y = size * 0.2 * math.sin(math.radians(angle))
            fade = (10 - i) / 10
            fade_color = self.blend_colors(stone_color, BOARD_COLOR, fade)
            self.canvas.create_oval(
                x - size + offset_x,
                y - size + offset_y,
                x + size + offset_x,
                y + size + offset_y,
                fill=fade_color,
                outline="black",
                width=1,
            )
            self.window.update()
            self.window.after(50)
        self.board[row][col] = None
        self.stones_count[1 - self.current_player] -= 1
        self.draw_board()

    def blend_colors(self, color1, color2, alpha):
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
        r = int(r1 * alpha + r2 * (1 - alpha))
        g = int(g1 * alpha + g2 * (1 - alpha))
        b = int(b1 * alpha + b2 * (1 - alpha))
        return f"#{r:02x}{g:02x}{b:02x}"

    def next_turn(self):
        self.current_player = 1 - self.current_player
        self.draw_board()

    def is_valid_move(self, fr, fc, tr, tc):
        if self.board[tr][tc] is not None:
            return False
        return (abs(tr - fr) == 1 and tc == fc) or (abs(tc - fc) == 1 and tr == fr)

    def check_no_three_in_row(self, row, col):
        self.board[row][col] = self.current_player
        h = self._count_dir(row, col, 0, -1) + self._count_dir(row, col, 0, 1) + 1
        v = self._count_dir(row, col, -1, 0) + self._count_dir(row, col, 1, 0) + 1
        self.board[row][col] = None
        return h < 3 and v < 3

    def _count_dir(self, row, col, dr, dc):
        p = self.current_player
        cnt = 0
        r, c = row + dr, col + dc
        while 0 <= r < GRID_HEIGHT and 0 <= c < GRID_WIDTH and self.board[r][c] == p:
            cnt += 1
            r += dr
            c += dc
        return cnt

    def check_for_line(self, row, col):
        p = self.board[row][col]
        h = self._count_line(row, col, p, 0, -1) + self._count_line(row, col, p, 0, 1) + 1
        if h >= 3:
            return True, "horizontal"
        v = self._count_line(row, col, p, -1, 0) + self._count_line(row, col, p, 1, 0) + 1
        if v >= 3:
            return True, "vertical"
        return False, None

    def check_for_line_at_position(self, row, col, board):
        p = board[row][col]
        if p is None:
            return False, None
        h = self._count_on_board(board, row, col, p, 0, -1) + self._count_on_board(board, row, col, p, 0, 1) + 1
        if h >= 3:
            return True, "horizontal"
        v = self._count_on_board(board, row, col, p, -1, 0) + self._count_on_board(board, row, col, p, 1, 0) + 1
        if v >= 3:
            return True, "vertical"
        return False, None

    def _count_line(self, row, col, p, dr, dc):
        cnt = 0
        r, c = row + dr, col + dc
        while 0 <= r < GRID_HEIGHT and 0 <= c < GRID_WIDTH and self.board[r][c] == p:
            cnt += 1
            r += dr
            c += dc
        return cnt

    def _count_on_board(self, board, row, col, p, dr, dc):
        cnt = 0
        r, c = row + dr, col + dc
        while 0 <= r < GRID_HEIGHT and 0 <= c < GRID_WIDTH and board[r][c] == p:
            cnt += 1
            r += dr
            c += dc
        return cnt

    def get_adjacent_opponent_stones(self, row, col, direction):
        player = self.board[row][col]
        opponent = 1 - player
        out = []
        if direction == "horizontal":
            left = col
            while left > 0 and self.board[row][left - 1] == player:
                left -= 1
            right = col
            while right < GRID_WIDTH - 1 and self.board[row][right + 1] == player:
                right += 1
            if left > 0 and self.board[row][left - 1] == opponent:
                out.append((row, left - 1))
            if right < GRID_WIDTH - 1 and self.board[row][right + 1] == opponent:
                out.append((row, right + 1))
        else:
            top = row
            while top > 0 and self.board[top - 1][col] == player:
                top -= 1
            bottom = row
            while bottom < GRID_HEIGHT - 1 and self.board[bottom + 1][col] == player:
                bottom += 1
            if top > 0 and self.board[top - 1][col] == opponent:
                out.append((top - 1, col))
            if bottom < GRID_HEIGHT - 1 and self.board[bottom + 1][col] == opponent:
                out.append((bottom + 1, col))
        return out


if __name__ == "__main__":
    BolotuduGame()
