
"""
02464 Human Memory Mini Project
Visual-only experiment runner for VS Code / Jupyter workflows.

What is automated:
- Condition order
- Trial generation
- On-screen instructions
- Short breaks between conditions
- Response collection
- Working-memory distractor task
- CSV logging at trial level and item level

Outputs (beside this Python file):
  <participant>_trials.csv
  <participant>_items.csv

Run:
  python memory_experiment.py
"""

from __future__ import annotations
import csv
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import simpledialog, messagebox

# -----------------------------
# CONFIGURATION
# -----------------------------

SEED = None
random.seed(SEED)

# Keep the repository completely flat: results are saved next to this script.
# Using __file__ also makes this work when the script is launched from another
# working directory.
OUTPUT_DIR = Path(__file__).resolve().parent

# Letters are sampled with replacement, so repetitions are allowed.
LETTER_POOL = list("ABCDEFGHJKMNPQRSTUVWXYZ")

FREE_REPS = 5
FREE_N = 16

SERIAL_REPS = 5
SERIAL_N = 12

CAPACITY_LENGTHS = [4, 5, 6, 7, 8, 9]
CAPACITY_REPS_PER_LENGTH = 2


IDIOM_REPS = 3
IDIOMS_PER_TRIAL = 12

REAL_IDIOMS = [
    "At slå en streg i sandet",
    "At have sommerfugle i maven",
    "At tale frit fra leveren",
    "At gå nedenom og hjem",
    "At få en fjer i hatten",
    "At tage tyren ved hornene",
    "At kaste håndklædet i ringen",
    "At slå to fluer med ét smæk",
    "At løbe panden mod en mur",
    "At være på bølgelængde",
    "At tage det sure med det søde",
    "At gå over gevind",
    "At komme op i det røde felt",
    "At bide i det sure æble",
    "At have is i maven",
    "At få blod på tanden",
    "At få kolde fødder",
    "At komme i fedtefadet",
    "At give den gas",
    "At male fanden på væggen",
    "At gå ned med flaget",
    "At skyde papegøjen",
    "At sidde på den høje hest",
    "At have mange bolde i luften",
    "At tabe ansigt",
    "At vende på en tallerken",
    "At have øjne i nakken",
    "At være en strid banan",
    "At gå på vandet",
    "At holde tungen lige i munden",
    "At feje for egen dør",
    "At trække det korteste strå",
    "At have en finger med i spillet",
    "At gå agurk",
    "At have rent mel i posen",
    "At stå med håret i postkassen",
]

FAKE_IDIOMS = [
    "At hænge hatten på den forkerte knage",
    "At have sand mellem tænderne",
    "At gå med vinden i lommen",
    "At slå søm i tågen",
    "At have begge hænder i honningkrukken",
    "At trække gardinet for tidligt",
    "At sætte støvlerne i den forkerte grøft",
    "At have en sten under hatten",
    "At løbe efter sin egen skygge",
    "At vende koppen på hovedet",
    "At få regn i støvlerne",
    "At gå med hovedet under armen",
    "At sætte ild til sin egen stige",
    "At have vinden under skjorten",
    "At kaste brænde på den forkerte båd",
    "At stå med næsen i postkassen",
    "At have grus i maskineriet",
    "At gå rundt om den samme sten",
    "At trække tæppet over hovedet",
    "At sætte sin hat efter vinden",
    "At have en ræv i rygsækken",
    "At slå knude på sin egen hale",
    "At gå med skeen i den forkerte gryde",
    "At få vand over støvlekanten",
    "At holde døren med albuen",
    "At have for mange nøgler i lommen",
    "At sætte sejl i modvind",
    "At stå med begge ben i samme støvle",
    "At få grus under vingerne",
    "At tænde lampen i dagslys",
    "At have månen i baglommen",
    "At kaste hatten før festen",
    "At gå med gaflen i suppen",
    "At slå hul i sin egen paraply",
    "At have begge fødder på samme trin",
    "At trække stigen op før man er på taget",
]

INTERVAL_SLOW = 3.0
INTERVAL_FAST = 1.0
UNFILLED_DELAY_SECONDS = 60
WM_TASK_SECONDS = 30
DOUBLE_ITEM_BLINK = 0.12

# All stimuli are visual.
FREE_MODE = "visual"
SERIAL_MODE = "visual"

# -----------------------------
# HELPERS
# -----------------------------

def clean_participant_id(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s.strip())
    return s or "anonymous"

def random_letters(n: int) -> list[str]:
    return random.choices(LETTER_POOL, k=n)

def random_phrases(pool: list[str], n: int) -> list[str]:
    return random.sample(pool, n)

def normalize_letters(text: str) -> list[str]:
    return [c for c in text.upper() if c in LETTER_POOL]

def normalize_phrase(text: str) -> str:
    text = text.lower().strip()
    replacements = {
        "é": "e", "è": "e", "ê": "e",
        "å": "aa",
        "æ": "ae",
        "ø": "oe",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_multiline_phrases(text: str) -> list[str]:
    """
    Accept one idiom per line, or separated by semicolons.
    """
    raw = text.replace(";", "\n")
    parts = [x.strip() for x in raw.splitlines() if x.strip()]
    return [normalize_phrase(x) for x in parts]

def multiset_free_recall_score(sequence: list[str], response: list[str]):
    seq_counts = Counter(sequence)
    resp_counts = Counter(response)
    correct = sum(min(seq_counts[k], resp_counts[k]) for k in seq_counts)
    return correct, correct / len(sequence)

def free_recall_weights(sequence: list[str], response: list[str]) -> list[float]:
    """Allocate recalled copies fairly across positions containing duplicates.

    Example: if A occurs three times and the participant recalls two A's, each
    A position receives 2/3. The weights therefore sum to the multiset-based
    number of correctly recalled items without favouring an early or late A.
    """
    seq_counts = Counter(sequence)
    resp_counts = Counter(response)
    return [
        min(seq_counts[item], resp_counts[item]) / seq_counts[item]
        for item in sequence
    ]

def split_regions(n: int):
    third = n // 3
    rem = n % 3
    sizes = [third, third, third]
    for i in range(rem):
        sizes[i] += 1

    labels = []
    for name, size in zip(["start", "middle", "end"], sizes):
        labels.extend([name] * size)
    return labels

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def append_csv(path: Path, row: dict):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

# -----------------------------
# GUI
# -----------------------------

class MemoryExperiment:
    def __init__(self, root: tk.Tk, participant: str):
        self.root = root
        self.participant = participant
        self.trials_path = OUTPUT_DIR / f"{participant}_trials.csv"
        self.items_path = OUTPUT_DIR / f"{participant}_items.csv"

        self.root.title("02464 Human Memory Experiment")
        self.root.geometry("1150x760")
        self.root.configure(bg="white")

        self.label = tk.Label(
            root, text="", font=("Helvetica", 52, "bold"),
            bg="white", fg="black", wraplength=1040, justify="center"
        )
        self.label.pack(expand=True)

        self.info = tk.Label(
            root, text="", font=("Helvetica", 18),
            bg="white", fg="black", wraplength=1040, justify="center"
        )
        self.info.pack(pady=20)

        self.tap_count = 0
        self.root.bind("<space>", self._record_tap)
        self.root.update()

    def _record_tap(self, event=None):
        self.tap_count += 1

    def show_text(self, text: str, size=52):
        self.label.config(text=text, font=("Helvetica", size, "bold"))
        self.root.update()

    def show_info(self, text: str):
        self.info.config(text=text)
        self.root.update()

    def clear(self):
        self.label.config(text="")
        self.info.config(text="")
        self.root.update()

    def wait(self, seconds: float, header: str = "", info: str = ""):
        start = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - start
            remaining = max(0, seconds - elapsed)
            if header:
                self.show_text(header, 34)
            if info:
                self.show_info(f"{info}\n\n{remaining:0.0f} s remaining")
            self.root.update()
            if remaining <= 0:
                break
            time.sleep(0.05)

    def show_block_intro(self, title: str, purpose: str, instructions: list[str]):
        self.show_text(title, 34)
        bullet_text = "• " + "\n• ".join(instructions)
        self.show_info(f"{purpose}\n\n{bullet_text}")
        messagebox.showinfo(title, f"{purpose}\n\n" + "\n".join(f"- {x}" for x in instructions), parent=self.root)

    def break_screen(self, title="Short break", text="Take a short break.\nClick OK when you are ready to continue."):
        messagebox.showinfo(title, text, parent=self.root)

    def present_items(
        self,
        items: list[str],
        interval: float,
        instruction: str = "",
        finger_tapping: bool = False,
        articulatory_suppression: bool = False,
    ):
        self.tap_count = 0

        if finger_tapping:
            instruction += "\nTap SPACE repeatedly during presentation."
        if articulatory_suppression:
            instruction += "\nContinuously repeat: TAH-DAH-TAH-DAH..."

        if instruction.strip():
            self.show_text("Get ready", 34)
            self.show_info(instruction)
            self.wait(3)

        previous_item = None
        for item in items:
            if previous_item == item:
                self.show_text("")
                blink_start = time.perf_counter()
                while time.perf_counter() - blink_start < DOUBLE_ITEM_BLINK:
                    self.root.update()
                    time.sleep(0.01)

            onset = time.perf_counter()
            font_size = 72 if len(item) <= 4 else 38 if len(item) <= 18 else 28
            self.show_text(item, font_size)

            while time.perf_counter() - onset < interval:
                self.root.update()
                time.sleep(0.02)

            previous_item = item

        return self.tap_count

    def get_response(self, prompt: str, max_letters: int | None = None) -> str:
        """
        Response window for letter tasks.

        If max_letters is given, the participant cannot enter more valid
        letter symbols than were presented in that trial. Spaces and commas
        do not count toward the limit.
        """
        if max_letters is None:
            self.clear()
            response = simpledialog.askstring("Recall", prompt, parent=self.root)
            return response or ""

        self.clear()
        win = tk.Toplevel(self.root)
        win.title("Recall")
        win.geometry("760x320")
        win.configure(bg="white")
        win.transient(self.root)
        win.grab_set()

        tk.Label(
            win,
            text=prompt,
            bg="white",
            fg="black",
            font=("Helvetica", 16),
            justify="left",
            wraplength=700
        ).pack(padx=20, pady=(24, 10), anchor="w")

        var = tk.StringVar()
        counter = tk.Label(
            win,
            text=f"0 / {max_letters} symbols",
            bg="white",
            fg="black",
            font=("Helvetica", 13)
        )
        counter.pack(pady=(0, 8))

        entry = tk.Entry(win, textvariable=var, font=("Helvetica", 22), width=42)
        entry.pack(padx=20, pady=10)
        entry.focus_set()

        output = {"text": ""}
        previous_valid = {"text": ""}

        def valid_letter_count(s: str) -> int:
            return len(normalize_letters(s))

        def on_change(*_):
            current = var.get()
            n = valid_letter_count(current)

            if n > max_letters:
                # Revert to the last valid state.
                var.set(previous_valid["text"])
                return

            previous_valid["text"] = current
            counter.config(text=f"{n} / {max_letters} symbols")

        var.trace_add("write", on_change)

        def submit(event=None):
            output["text"] = var.get()
            win.destroy()

        tk.Button(
            win, text="Submit", command=submit, font=("Helvetica", 14)
        ).pack(pady=14)

        win.bind("<Return>", submit)
        win.wait_window()
        return output["text"]

    def get_multiline_response(self, title: str, prompt: str, max_items: int | None = None) -> str:
        self.clear()
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("900x600")
        win.configure(bg="white")
        win.transient(self.root)
        win.grab_set()

        tk.Label(
            win, text=prompt, bg="white", fg="black",
            font=("Helvetica", 16), justify="left", wraplength=840
        ).pack(padx=20, pady=20, anchor="w")

        text_widget = tk.Text(win, wrap="word", font=("Helvetica", 14), height=20)
        text_widget.pack(fill="both", expand=True, padx=20, pady=10)

        output = {"text": ""}

        counter = tk.Label(
            win,
            text=(f"0 / {max_items} responses" if max_items is not None else ""),
            bg="white",
            fg="black",
            font=("Helvetica", 12)
        )
        counter.pack(pady=(0, 6))

        def current_items():
            raw = text_widget.get("1.0", "end").strip()
            return [x for x in raw.replace(";", "\n").splitlines() if x.strip()]

        def update_counter(event=None):
            if max_items is not None:
                items_now = current_items()
                if len(items_now) > max_items:
                    # Remove the extra final line(s).
                    trimmed = "\n".join(items_now[:max_items])
                    text_widget.delete("1.0", "end")
                    text_widget.insert("1.0", trimmed)
                    items_now = items_now[:max_items]
                counter.config(text=f"{len(items_now)} / {max_items} responses")

        text_widget.bind("<KeyRelease>", update_counter)

        def submit():
            output["text"] = text_widget.get("1.0", "end").strip()
            win.destroy()

        tk.Button(win, text="Submit", command=submit, font=("Helvetica", 14)).pack(pady=15)

        win.wait_window()
        return output["text"]

    def working_memory_task(self, seconds=WM_TASK_SECONDS):
        self.show_text("Working-memory task", 32)
        self.show_info("Answer as many multiplication problems as possible before recall.")
        self.wait(2)

        start = time.perf_counter()
        n_answered = 0
        n_correct = 0

        while time.perf_counter() - start < seconds:
            a = random.choice([3, 4, 6, 7, 8, 9])
            b = random.randint(2, 12)
            remaining = max(0, seconds - (time.perf_counter() - start))
            ans = simpledialog.askstring(
                "Working-memory task",
                f"{a} × {b} = ?\n\nAbout {remaining:0.0f} s remaining",
                parent=self.root
            )
            if ans is None:
                break

            n_answered += 1
            try:
                if int(ans.strip()) == a * b:
                    n_correct += 1
            except ValueError:
                pass

        return n_answered, n_correct

    def log_trial(
        self,
        experiment: str,
        condition: str,
        trial: int,
        mode: str,
        sequence: list[str],
        response: list[str],
        interval: float,
        delay_seconds: float = 0,
        wm_n=0,
        wm_correct=0,
        tap_count=0,
        chunk_label="",
        response_format="letters",
    ):
        n = len(sequence)
        serial_correct = sum(
            1 for i, x in enumerate(sequence)
            if i < len(response) and response[i] == x
        )
        free_correct, free_accuracy = multiset_free_recall_score(sequence, response)
        recall_weights = free_recall_weights(sequence, response)

        regions = split_regions(n)
        region_scores = {}
        for region in ["start", "middle", "end"]:
            idx = [i for i, r in enumerate(regions) if r == region]
            region_scores[region] = sum(recall_weights[i] for i in idx) / len(idx)

        trial_row = {
            "timestamp": now_iso(),
            "participant": self.participant,
            "experiment": experiment,
            "condition": condition,
            "trial": trial,
            "mode": mode,
            "response_format": response_format,
            "n_items": n,
            "interval_s": interval,
            "delay_s": delay_seconds,
            "sequence": " || ".join(sequence),
            "response": " || ".join(response),
            "free_correct_n": free_correct,
            "free_accuracy": round(free_accuracy, 6),
            "serial_correct_n": serial_correct,
            "serial_accuracy": round(serial_correct / n, 6),
            "start_accuracy": round(region_scores["start"], 6),
            "middle_accuracy": round(region_scores["middle"], 6),
            "end_accuracy": round(region_scores["end"], 6),
            "primacy_index": round(region_scores["start"] - region_scores["middle"], 6),
            "recency_index": round(region_scores["end"] - region_scores["middle"], 6),
            "wm_questions": wm_n,
            "wm_correct": wm_correct,
            "tap_count": tap_count,
            "chunk_label": chunk_label,
        }
        append_csv(self.trials_path, trial_row)

        for pos, item in enumerate(sequence):
            response_here = response[pos] if pos < len(response) else ""
            item_row = {
                "timestamp": trial_row["timestamp"],
                "participant": self.participant,
                "experiment": experiment,
                "condition": condition,
                "trial": trial,
                "mode": mode,
                "response_format": response_format,
                "n_items": n,
                "serial_position": pos + 1,
                "region": regions[pos],
                "presented_item": item,
                "response_at_position": response_here,
                "free_recalled": round(recall_weights[pos], 6),
                "serial_correct": int(response_here == item),
                "substitution": response_here if response_here and response_here != item else "",
                "interval_s": interval,
                "delay_s": delay_seconds,
                "chunk_label": chunk_label,
            }
            append_csv(self.items_path, item_row)

    # -----------------------------
    # FREE RECALL
    # -----------------------------

    def run_free_condition(self, condition: str, interval: float, reps=FREE_REPS,
                           delay=0, working_memory=False):
        label = {
            "baseline": "Baseline free recall",
            "unfilled_delay": "Free recall with unfilled delay",
            "fast_rate": "Free recall with fast presentation",
            "working_memory": "Free recall with working-memory interference",
        }[condition]

        instructions = [
            f"You will see {FREE_N} letters.",
            "Try to remember as many letters as possible.",
            "Order does NOT matter in this block.",
            f"Presentation speed: {interval:.0f} second(s) per item.",
        ]
        if delay > 0:
            instructions.append(f"There will be a {delay:.0f}-second pause before recall.")
        if working_memory:
            instructions.append("You will solve multiplication problems before recall.")

        self.show_block_intro(label, "This block measures free recall.", instructions)

        for t in range(1, reps + 1):
            seq = random_letters(FREE_N)
            self.show_text(f"{label}\nTrial {t}/{reps}", 30)
            self.show_info("Remember as many letters as possible. Order does not matter.")
            self.wait(2)

            self.present_items(seq, interval)

            wm_n = wm_correct = 0
            if working_memory:
                wm_n, wm_correct = self.working_memory_task(WM_TASK_SECONDS)
            elif delay > 0:
                self.wait(delay, "Pause before recall", "Wait quietly before answering.")

            raw = self.get_response(
                "Type every letter you remember.\nOrder does not matter.\n"
                f"You can enter at most {len(seq)} letter symbols.",
                max_letters=len(seq)
            )
            resp = normalize_letters(raw)

            self.log_trial(
                "free_recall", condition, t, FREE_MODE, seq, resp, interval,
                delay_seconds=delay, wm_n=wm_n, wm_correct=wm_correct,
                response_format="letters"
            )

        self.break_screen()

    # -----------------------------
    # SERIAL RECALL
    # -----------------------------

    def run_serial_condition(self, condition: str, reps=SERIAL_REPS,
                             finger_tapping=False, suppression=False):
        label = {
            "baseline": "Baseline serial recall",
            "finger_tapping": "Serial recall with finger tapping",
            "articulatory_suppression": "Serial recall with articulatory suppression",
        }[condition]

        instructions = [
            f"You will see {SERIAL_N} letters.",
            "Remember the exact sequence.",
            "Order matters in this block.",
            f"Presentation speed: {INTERVAL_SLOW:.0f} seconds per item.",
        ]
        if finger_tapping:
            instructions.append("Keep tapping SPACE during presentation.")
        if suppression:
            instructions.append("Continuously repeat: TAH-DAH-TAH-DAH... during presentation.")

        self.show_block_intro(label, "This block measures serial recall.", instructions)

        for t in range(1, reps + 1):
            seq = random_letters(SERIAL_N)
            self.show_text(f"{label}\nTrial {t}/{reps}", 30)
            self.show_info("Remember the letters in the same order.")
            self.wait(2)

            taps = self.present_items(
                seq, INTERVAL_SLOW,
                finger_tapping=finger_tapping,
                articulatory_suppression=suppression
            )

            raw = self.get_response(
                "Type the letters in the SAME ORDER.\n"
                f"You can enter at most {len(seq)} letter symbols.",
                max_letters=len(seq)
            )
            resp = normalize_letters(raw)

            self.log_trial(
                "serial_recall", condition, t, SERIAL_MODE, seq, resp,
                INTERVAL_SLOW, tap_count=taps, response_format="letters"
            )

        self.break_screen()

    def run_capacity(self):
        self.show_block_intro(
            "Memory span / capacity",
            "This block estimates the limited capacity of working memory.",
            [
                "List length changes from trial to trial.",
                "Remember the letters in the exact order.",
                "Type the whole sequence after each trial.",
            ],
        )

        trial = 0
        conditions = []
        for n in CAPACITY_LENGTHS:
            conditions += [n] * CAPACITY_REPS_PER_LENGTH

        for n in conditions:
            trial += 1
            seq = random_letters(n)
            self.show_text(f"Memory span block\n{n} letters\nTrial {trial}/{len(conditions)}", 30)
            self.show_info("Remember the letters in the same order.")
            self.wait(2)

            self.present_items(seq, INTERVAL_SLOW)
            raw = self.get_response(
                f"Type the letters in the SAME ORDER.\n"
                f"You can enter at most {len(seq)} letter symbols.",
                max_letters=len(seq)
            )
            resp = normalize_letters(raw)

            self.log_trial(
                "serial_recall", f"capacity_n{n}", trial, SERIAL_MODE,
                seq, resp, INTERVAL_SLOW, response_format="letters"
            )

        self.break_screen()

    def run_idiom_condition(self, condition: str, pool: list[str], reps=IDIOM_REPS):
        label = {
            "real_idioms": "Wordsporg / real idioms",
            "fake_idioms": "Wordsporg / fake idioms",
        }[condition]

        self.show_block_intro(
            label,
            "This block tests whether familiar idioms are easier to remember than unfamiliar fake idioms.",
            [
                f"You will see {IDIOMS_PER_TRIAL} idioms.",
                "Remember them in the exact order.",
                "Afterwards type one idiom per line, in the same order.",
                f"Presentation speed: {INTERVAL_SLOW:.0f} seconds per idiom.",
            ],
        )

        for t in range(1, reps + 1):
            seq_display = random_phrases(pool, IDIOMS_PER_TRIAL)
            seq_norm = [normalize_phrase(x) for x in seq_display]

            self.show_text(f"{label}\nTrial {t}/{reps}", 30)
            self.show_info("Remember the idioms in the same order.")
            self.wait(2)

            self.present_items(seq_display, INTERVAL_SLOW)

            raw = self.get_multiline_response(
                label,
                "Type the idioms in the SAME ORDER.\n"
                f"Write one idiom per line. Maximum {len(seq_display)} responses.\n"
                "Minor punctuation differences do not matter.",
                max_items=len(seq_display)
            )
            resp = parse_multiline_phrases(raw)

            self.log_trial(
                "idiom_serial_recall", condition, t, SERIAL_MODE,
                seq_norm, resp, INTERVAL_SLOW, response_format="phrases"
            )

        self.break_screen()

    def choose_experiments_gui(self) -> list[str]:
        """
        Graphical experiment selector.

        Returns experiment keys in the fixed experimental order, so the user
        can choose any subset without changing the planned order.
        """
        self.clear()

        win = tk.Toplevel(self.root)
        win.title("Choose experiments")
        win.geometry("720x720")
        win.configure(bg="white")
        win.transient(self.root)
        win.grab_set()

        tk.Label(
            win,
            text="Choose which experiments to run",
            bg="white",
            fg="black",
            font=("Helvetica", 24, "bold")
        ).pack(pady=(24, 8))

        tk.Label(
            win,
            text="Select any combination. They will run in the fixed experimental order.",
            bg="white",
            fg="black",
            font=("Helvetica", 14),
            wraplength=650
        ).pack(pady=(0, 18))

        experiments = [
            ("free_baseline", "Free recall — baseline"),
            ("free_delay", "Free recall — 60 s pause"),
            ("free_fast", "Free recall — fast presentation"),
            ("free_wm", "Free recall — working-memory task"),
            ("capacity", "Serial recall — capacity"),
            ("serial_baseline", "Serial recall — baseline"),
            ("finger_tapping", "Serial recall — finger tapping"),
            ("articulatory_suppression", "Serial recall — articulatory suppression"),
            ("real_idioms", "Wordsporg — real idioms"),
            ("fake_idioms", "Wordsporg — fake idioms"),
        ]

        vars_by_key = {}
        frame = tk.Frame(win, bg="white")
        frame.pack(fill="both", expand=True, padx=35, pady=5)

        for key, label in experiments:
            var = tk.BooleanVar(value=False)
            vars_by_key[key] = var
            tk.Checkbutton(
                frame,
                text=label,
                variable=var,
                bg="white",
                anchor="w",
                font=("Helvetica", 14),
                padx=8,
                pady=4
            ).pack(fill="x", anchor="w")

        result = {"keys": []}

        def select_all():
            for var in vars_by_key.values():
                var.set(True)

        def clear_all():
            for var in vars_by_key.values():
                var.set(False)

        def start():
            result["keys"] = [key for key, _ in experiments if vars_by_key[key].get()]
            if not result["keys"]:
                messagebox.showwarning(
                    "Choose an experiment",
                    "Select at least one experiment.",
                    parent=win
                )
                return
            win.destroy()

        button_row = tk.Frame(win, bg="white")
        button_row.pack(pady=15)

        tk.Button(button_row, text="Select all", command=select_all, width=12).pack(side="left", padx=6)
        tk.Button(button_row, text="Clear", command=clear_all, width=12).pack(side="left", padx=6)
        tk.Button(button_row, text="Start", command=start, width=12).pack(side="left", padx=6)

        win.wait_window()
        return result["keys"]

    def run_selected(self, selected: list[str]):
        """
        Run only the selected experiments, preserving the intended order.
        """
        action_map = {
            "free_baseline": lambda: self.run_free_condition("baseline", INTERVAL_SLOW, FREE_REPS, 0, False),
            "free_delay": lambda: self.run_free_condition("unfilled_delay", INTERVAL_SLOW, FREE_REPS, UNFILLED_DELAY_SECONDS, False),
            "free_fast": lambda: self.run_free_condition("fast_rate", INTERVAL_FAST, FREE_REPS, 0, False),
            "free_wm": lambda: self.run_free_condition("working_memory", INTERVAL_SLOW, FREE_REPS, 0, True),
            "capacity": self.run_capacity,
            "serial_baseline": lambda: self.run_serial_condition("baseline", SERIAL_REPS, False, False),
            "finger_tapping": lambda: self.run_serial_condition("finger_tapping", SERIAL_REPS, True, False),
            "articulatory_suppression": lambda: self.run_serial_condition("articulatory_suppression", SERIAL_REPS, False, True),
            "real_idioms": lambda: self.run_idiom_condition("real_idioms", REAL_IDIOMS, IDIOM_REPS),
            "fake_idioms": lambda: self.run_idiom_condition("fake_idioms", FAKE_IDIOMS, IDIOM_REPS),
        }

        fixed_order = [
            "free_baseline",
            "free_delay",
            "free_fast",
            "free_wm",
            "capacity",
            "serial_baseline",
            "finger_tapping",
            "articulatory_suppression",
            "real_idioms",
            "fake_idioms",
        ]

        for key in fixed_order:
            if key in selected:
                action_map[key]()

        messagebox.showinfo(
            "Done",
            f"Selected experiments complete.\\n\\nSaved:\\n{self.trials_path}\\n{self.items_path}",
            parent=self.root
        )

    # -----------------------------
    # MENUS
    # -----------------------------

    def run_free_all(self):
        conditions = [
            ("baseline", INTERVAL_SLOW, 0, False),
            ("unfilled_delay", INTERVAL_SLOW, UNFILLED_DELAY_SECONDS, False),
            ("fast_rate", INTERVAL_FAST, 0, False),
            ("working_memory", INTERVAL_SLOW, 0, True),
        ]
        for c, interval, delay, wm in conditions:
            self.run_free_condition(c, interval, FREE_REPS, delay, wm)

    def run_serial_all(self):
        self.run_capacity()
        self.run_serial_condition("baseline", SERIAL_REPS, False, False)
        self.run_serial_condition("finger_tapping", SERIAL_REPS, True, False)
        self.run_serial_condition("articulatory_suppression", SERIAL_REPS, False, True)
        self.run_idiom_condition("real_idioms", REAL_IDIOMS, IDIOM_REPS)
        self.run_idiom_condition("fake_idioms", FAKE_IDIOMS, IDIOM_REPS)

    def run_everything(self):
        self.run_free_all()
        self.run_serial_all()
        messagebox.showinfo(
            "Done",
            f"Experiment complete.\n\nSaved:\n{self.trials_path}\n{self.items_path}",
            parent=self.root
        )

def main():
    root = tk.Tk()
    root.withdraw()

    participant = simpledialog.askstring(
        "Participant ID",
        "Enter an anonymous participant ID (e.g. P01):",
        parent=root
    )
    if not participant:
        root.destroy()
        return

    participant = clean_participant_id(participant)
    root.deiconify()

    app = MemoryExperiment(root, participant)

    app.show_text("02464 Human Memory Experiment", 36)
    app.show_info(
        "All stimuli are presented visually.\n\n"
        "The program guides the participant through each block,\n"
        "collects responses automatically, and saves CSV files at the end."
    )
    app.wait(4)

    selected = app.choose_experiments_gui()
    if selected:
        app.run_selected(selected)

    root.destroy()

if __name__ == "__main__":
    main()
