"""
Human Memory Experiment — visual-only, CSV version
==================================================

A local browser-based experiment runner for free recall and serial recall.
All stimuli are presented visually. No audio, Tkinter, Excel, or third-party
Python packages are required.

Run:
    python human_memory_experiment_visual_csv.py

The script opens a browser at http://127.0.0.1:8765 and automatically appends
responses after every trial to:
    human_memory_results.csv
    human_memory_item_level.csv

Chunking stimuli are read from:
    human_memory_idioms.csv

Keep this script and human_memory_idioms.csv in the same folder.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import string
import threading
import uuid
import webbrowser
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
IDIOMS_CSV = BASE_DIR / "human_memory_idioms.csv"
RESULTS_CSV = BASE_DIR / "human_memory_results.csv"
ITEM_RESULTS_CSV = BASE_DIR / "human_memory_item_level.csv"

HOST = "127.0.0.1"
PORT = 8765

ALPHABET = list(string.ascii_uppercase)
FREE_RECALL_LENGTH = 16
SERIAL_RECALL_LENGTH = 12

TRIALS_PER_LETTER_CONDITION = 5
TRIALS_PER_IDIOM_CONDITION = 3

BASE_INTERVAL_S = 3.0
FAST_INTERVAL_S = 1.0
FREE_DELAY_S = 60
WORKING_MEMORY_DISTRACTOR_S = 30

SERIAL_SPAN_LENGTHS = [4, 5, 6, 7, 8, 9, 10]
SERIAL_SPAN_REPETITIONS = 2

# For free recall: positions 1–5 = primacy, 6–11 = middle, 12–16 = recency.
FREE_PRIMACY_END = 5
FREE_RECENCY_START = 12

DISTRACTOR_TABLES = [3, 4, 7, 8, 9]

TRIAL_HEADERS = [
    "timestamp_utc",
    "participant_id",
    "session_id",
    "block",
    "condition",
    "trial_number",
    "presentation_mode",
    "n_items",
    "item_interval_s",
    "pre_recall_delay_s",
    "secondary_task",
    "presented_sequence",
    "response_raw",
    "parsed_response",
    "total_correct",
    "proportion_correct",
    "primacy_proportion",
    "middle_proportion",
    "recency_proportion",
    "primacy_effect",
    "recency_effect",
    "exact_serial_correct",
    "omissions",
    "intrusions",
    "substitution_pairs",
    "phonologically_similar_substitutions",
]

ITEM_HEADERS = [
    "timestamp_utc",
    "participant_id",
    "session_id",
    "block",
    "condition",
    "trial_number",
    "serial_position",
    "region",
    "presented_item",
    "recalled_item_at_position",
    "correct",
    "error_type",
]


# -----------------------------------------------------------------------------
# DATA MODEL
# -----------------------------------------------------------------------------

@dataclass
class Trial:
    trial_id: str
    block: str
    condition: str
    trial_number: int
    items: list[str]
    interval_s: float
    pre_recall_delay_s: int = 0
    secondary_task: str = ""
    response_kind: str = "letters"  # letters | idioms
    distractor_table: Optional[int] = None


# -----------------------------------------------------------------------------
# STIMULUS LOADING / GENERATION
# -----------------------------------------------------------------------------

def load_idioms() -> tuple[list[str], list[str]]:
    if not IDIOMS_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {IDIOMS_CSV.name}. Keep it in the same folder as the script."
        )

    real: list[str] = []
    fake: list[str] = []
    with IDIOMS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kind = (row.get("type") or "").strip().lower()
            idiom = (row.get("idiom") or "").strip()
            if not idiom:
                continue
            if kind == "real":
                real.append(idiom)
            elif kind == "fake":
                fake.append(idiom)

    if len(real) < 12 or len(fake) < 12:
        raise ValueError(
            f"{IDIOMS_CSV.name} must contain at least 12 real and 12 fake idioms."
        )
    return real, fake


def make_unique_free_sequence(used: set[tuple[str, ...]]) -> list[str]:
    # Unique letters make free-recall scoring unambiguous.
    while True:
        seq = tuple(random.sample(ALPHABET, FREE_RECALL_LENGTH))
        if seq not in used:
            used.add(seq)
            return list(seq)


def make_serial_sequence(length: int, used: set[tuple[str, ...]]) -> list[str]:
    # Repeated letters are allowed in serial recall.
    while True:
        seq = tuple(random.choices(ALPHABET, k=length))
        if seq not in used:
            used.add(seq)
            return list(seq)


def make_idiom_trials(pool: list[str], condition: str) -> list[Trial]:
    trials: list[Trial] = []
    # Prefer no idiom repetition across the three trials when enough are available.
    shuffled = pool[:]
    random.shuffle(shuffled)
    cursor = 0
    for i in range(1, TRIALS_PER_IDIOM_CONDITION + 1):
        if cursor + 12 <= len(shuffled):
            items = shuffled[cursor:cursor + 12]
            cursor += 12
        else:
            items = random.sample(pool, 12)
        trials.append(
            Trial(
                trial_id=str(uuid.uuid4()),
                block=condition,
                condition=condition,
                trial_number=i,
                items=items,
                interval_s=BASE_INTERVAL_S,
                response_kind="idioms",
            )
        )
    return trials


def build_trials(block_key: str) -> list[Trial]:
    used: set[tuple[str, ...]] = set()
    trials: list[Trial] = []

    if block_key == "free_baseline":
        for i in range(1, 6):
            trials.append(Trial(str(uuid.uuid4()), "Free recall", "Baseline", i,
                                make_unique_free_sequence(used), BASE_INTERVAL_S))

    elif block_key == "free_delay":
        for i in range(1, 6):
            trials.append(Trial(str(uuid.uuid4()), "Free recall", "60-second pause", i,
                                make_unique_free_sequence(used), BASE_INTERVAL_S,
                                pre_recall_delay_s=FREE_DELAY_S))

    elif block_key == "free_fast":
        for i in range(1, 6):
            trials.append(Trial(str(uuid.uuid4()), "Free recall", "Fast presentation", i,
                                make_unique_free_sequence(used), FAST_INTERVAL_S))

    elif block_key == "free_distractor":
        tables = DISTRACTOR_TABLES[:]
        random.shuffle(tables)
        for i in range(1, 6):
            table = tables[i - 1]
            trials.append(Trial(
                str(uuid.uuid4()), "Free recall", "Working-memory distractor", i,
                make_unique_free_sequence(used), BASE_INTERVAL_S,
                pre_recall_delay_s=WORKING_MEMORY_DISTRACTOR_S,
                secondary_task=f"Say the {table}-times table aloud continuously",
                distractor_table=table,
            ))

    elif block_key == "serial_baseline":
        for i in range(1, 6):
            trials.append(Trial(str(uuid.uuid4()), "Serial recall", "Baseline", i,
                                make_serial_sequence(SERIAL_RECALL_LENGTH, used), BASE_INTERVAL_S))

    elif block_key == "serial_tapping":
        for i in range(1, 6):
            trials.append(Trial(
                str(uuid.uuid4()), "Serial recall", "Finger tapping", i,
                make_serial_sequence(SERIAL_RECALL_LENGTH, used), BASE_INTERVAL_S,
                secondary_task="Tap one finger at a steady pace throughout the sequence",
            ))

    elif block_key == "serial_suppression":
        for i in range(1, 6):
            trials.append(Trial(
                str(uuid.uuid4()), "Serial recall", "Articulatory suppression", i,
                make_serial_sequence(SERIAL_RECALL_LENGTH, used), BASE_INTERVAL_S,
                secondary_task="Sing or hum Axel F continuously throughout the sequence",
            ))

    elif block_key == "serial_real_idioms":
        real, _ = load_idioms()
        return make_idiom_trials(real, "Real idioms")

    elif block_key == "serial_fake_idioms":
        _, fake = load_idioms()
        return make_idiom_trials(fake, "Fake idioms")

    elif block_key == "serial_span":
        trial_no = 1
        lengths = [n for n in SERIAL_SPAN_LENGTHS for _ in range(SERIAL_SPAN_REPETITIONS)]
        random.shuffle(lengths)
        for n in lengths:
            trials.append(Trial(
                str(uuid.uuid4()), "Serial recall", "Capacity/span", trial_no,
                make_serial_sequence(n, used), BASE_INTERVAL_S,
            ))
            trial_no += 1

    else:
        raise ValueError(f"Unknown block: {block_key}")

    return trials


BLOCKS = {
    "free_baseline": "Free recall — baseline (16 letters, 3 s/item)",
    "free_delay": "Free recall — 60 s pause before recall",
    "free_fast": "Free recall — fast presentation (1 s/item)",
    "free_distractor": "Free recall — working-memory distractor",
    "serial_baseline": "Serial recall — baseline (12 letters, 3 s/item)",
    "serial_tapping": "Serial recall — finger tapping",
    "serial_suppression": "Serial recall — articulatory suppression",
    "serial_real_idioms": "Serial recall — real idioms",
    "serial_fake_idioms": "Serial recall — fake idioms",
    "serial_span": "Serial recall — capacity/span (4–10 letters)",
}


# -----------------------------------------------------------------------------
# RESPONSE PARSING / SCORING
# -----------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def parse_letters(text: str) -> list[str]:
    # Accept: "A J K", "AJK", "A, J, K", etc.
    return re.findall(r"[A-Za-z]", text.upper())


def parse_idioms(text: str) -> list[str]:
    # Best input: one idiom per line. Semicolons also work.
    parts = re.split(r"[\n;]+", text)
    return [p.strip() for p in parts if p.strip()]


def free_score(items: list[str], response: list[str]) -> dict:
    # Free recall ignores response order. Since presented letters are unique,
    # each serial position has a simple recalled/not-recalled score.
    response_unique = set(response)
    position_correctness = [1 if item in response_unique else 0 for item in items]
    total_correct = sum(position_correctness)

    primacy = position_correctness[:FREE_PRIMACY_END]
    middle = position_correctness[FREE_PRIMACY_END:FREE_RECENCY_START - 1]
    recency = position_correctness[FREE_RECENCY_START - 1:]

    p_primacy = sum(primacy) / len(primacy)
    p_middle = sum(middle) / len(middle)
    p_recency = sum(recency) / len(recency)

    intrusions = [x for x in response if x not in items]
    omissions = [x for x, ok in zip(items, position_correctness) if not ok]

    return {
        "parsed_response": response,
        "total_correct": total_correct,
        "proportion_correct": total_correct / len(items),
        "position_correctness": position_correctness,
        "recalled_at_position": [item if item in response_unique else "" for item in items],
        "primacy_proportion": p_primacy,
        "middle_proportion": p_middle,
        "recency_proportion": p_recency,
        "primacy_effect": p_primacy - p_middle,
        "recency_effect": p_recency - p_middle,
        "exact_serial_correct": "",
        "omissions": omissions,
        "intrusions": intrusions,
        "substitution_pairs": [],
        "phonologically_similar_substitutions": 0,
    }


# Rough English letter-name rhyme groups. This is not used to grade the trial;
# it is only stored to help inspect the *type* of serial-recall substitutions.
PHONO_GROUPS = [
    set("BCDEGPTVZ"),
    set("FLMNSX"),
    set("JK"),
    set("QU"),
]


def phonologically_similar(a: str, b: str) -> bool:
    return any(a in g and b in g for g in PHONO_GROUPS)


def serial_letter_score(items: list[str], response: list[str]) -> dict:
    recalled = response[:len(items)] + [""] * max(0, len(items) - len(response))
    correctness = [1 if p == r else 0 for p, r in zip(items, recalled)]
    total_correct = sum(correctness)
    substitutions: list[str] = []
    phon_similar = 0

    for p, r, ok in zip(items, recalled, correctness):
        if not ok and r:
            substitutions.append(f"{p}->{r}")
            if phonologically_similar(p, r):
                phon_similar += 1

    # Multiset accounting: extras are intrusions; missing occurrences are omissions.
    presented_counts = Counter(items)
    response_counts = Counter(response)
    omissions: list[str] = []
    intrusions: list[str] = []
    for item, count in (presented_counts - response_counts).items():
        omissions.extend([item] * count)
    for item, count in (response_counts - presented_counts).items():
        intrusions.extend([item] * count)

    return {
        "parsed_response": response,
        "total_correct": total_correct,
        "proportion_correct": total_correct / len(items),
        "position_correctness": correctness,
        "recalled_at_position": recalled,
        "primacy_proportion": "",
        "middle_proportion": "",
        "recency_proportion": "",
        "primacy_effect": "",
        "recency_effect": "",
        "exact_serial_correct": int(response[:len(items)] == items and len(response) == len(items)),
        "omissions": omissions,
        "intrusions": intrusions,
        "substitution_pairs": substitutions,
        "phonologically_similar_substitutions": phon_similar,
    }


def serial_idiom_score(items: list[str], response: list[str]) -> dict:
    p_norm = [normalise_text(x) for x in items]
    r_norm = [normalise_text(x) for x in response]
    recalled_norm = r_norm[:len(items)] + [""] * max(0, len(items) - len(r_norm))
    correctness = [1 if p == r else 0 for p, r in zip(p_norm, recalled_norm)]
    total_correct = sum(correctness)

    omissions = [items[i] for i, ok in enumerate(correctness) if not ok and not recalled_norm[i]]
    substitutions = [
        f"{items[i]} -> {response[i]}"
        for i, ok in enumerate(correctness)
        if not ok and i < len(response) and response[i].strip()
    ]

    return {
        "parsed_response": response,
        "total_correct": total_correct,
        "proportion_correct": total_correct / len(items),
        "position_correctness": correctness,
        "recalled_at_position": response[:len(items)] + [""] * max(0, len(items) - len(response)),
        "primacy_proportion": "",
        "middle_proportion": "",
        "recency_proportion": "",
        "primacy_effect": "",
        "recency_effect": "",
        "exact_serial_correct": int(len(response) == len(items) and all(correctness)),
        "omissions": omissions,
        "intrusions": response[len(items):],
        "substitution_pairs": substitutions,
        "phonologically_similar_substitutions": "",
    }


def score_trial(trial: Trial, raw_response: str) -> dict:
    if trial.response_kind == "idioms":
        parsed = parse_idioms(raw_response)
        return serial_idiom_score(trial.items, parsed)

    parsed = parse_letters(raw_response)
    if trial.block == "Free recall":
        return free_score(trial.items, parsed)
    return serial_letter_score(trial.items, parsed)


# -----------------------------------------------------------------------------
# CSV STORAGE
# -----------------------------------------------------------------------------

CSV_LOCK = threading.Lock()


def append_csv(path: Path, headers: list[str], row: dict) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def region_for(trial: Trial, pos: int) -> str:
    if trial.block != "Free recall" or len(trial.items) != FREE_RECALL_LENGTH:
        return ""
    if pos <= FREE_PRIMACY_END:
        return "primacy"
    if pos >= FREE_RECENCY_START:
        return "recency"
    return "middle"


def save_trial(participant_id: str, session_id: str, trial: Trial, raw_response: str) -> None:
    score = score_trial(trial, raw_response)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    trial_row = {
        "timestamp_utc": now,
        "participant_id": participant_id,
        "session_id": session_id,
        "block": trial.block,
        "condition": trial.condition,
        "trial_number": trial.trial_number,
        "presentation_mode": "visual",
        "n_items": len(trial.items),
        "item_interval_s": trial.interval_s,
        "pre_recall_delay_s": trial.pre_recall_delay_s,
        "secondary_task": trial.secondary_task,
        "presented_sequence": " | ".join(trial.items),
        "response_raw": raw_response.replace("\r", ""),
        "parsed_response": " | ".join(score["parsed_response"]),
        "total_correct": score["total_correct"],
        "proportion_correct": round(score["proportion_correct"], 6),
        "primacy_proportion": score["primacy_proportion"],
        "middle_proportion": score["middle_proportion"],
        "recency_proportion": score["recency_proportion"],
        "primacy_effect": score["primacy_effect"],
        "recency_effect": score["recency_effect"],
        "exact_serial_correct": score["exact_serial_correct"],
        "omissions": " | ".join(score["omissions"]),
        "intrusions": " | ".join(score["intrusions"]),
        "substitution_pairs": " | ".join(score["substitution_pairs"]),
        "phonologically_similar_substitutions": score["phonologically_similar_substitutions"],
    }

    with CSV_LOCK:
        append_csv(RESULTS_CSV, TRIAL_HEADERS, trial_row)

        for idx, presented in enumerate(trial.items):
            recalled = score["recalled_at_position"][idx] if idx < len(score["recalled_at_position"]) else ""
            ok = score["position_correctness"][idx]
            error_type = ""
            if not ok:
                if not recalled:
                    error_type = "omission"
                elif trial.response_kind == "letters" and phonologically_similar(presented, recalled):
                    error_type = "phonologically_similar_substitution"
                else:
                    error_type = "substitution"

            item_row = {
                "timestamp_utc": now,
                "participant_id": participant_id,
                "session_id": session_id,
                "block": trial.block,
                "condition": trial.condition,
                "trial_number": trial.trial_number,
                "serial_position": idx + 1,
                "region": region_for(trial, idx + 1),
                "presented_item": presented,
                "recalled_item_at_position": recalled,
                "correct": ok,
                "error_type": error_type,
            }
            append_csv(ITEM_RESULTS_CSV, ITEM_HEADERS, item_row)


# -----------------------------------------------------------------------------
# SERVER STATE
# -----------------------------------------------------------------------------

SESSIONS: dict[str, dict] = {}
SESSION_LOCK = threading.Lock()


def public_trial(trial: Trial) -> dict:
    data = asdict(trial)
    data["presentation_mode"] = "visual"
    return data


# -----------------------------------------------------------------------------
# BROWSER UI
# -----------------------------------------------------------------------------

HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Human Memory Experiment</title>
<style>
  :root { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  body { margin:0; background:#f5f5f7; color:#1d1d1f; }
  .page { max-width:980px; margin:0 auto; padding:32px 18px 60px; }
  .card { background:white; border:1px solid #dedee3; border-radius:18px; padding:26px; box-shadow:0 6px 24px rgba(0,0,0,.05); }
  h1 { margin:0 0 6px; font-size:30px; }
  h2 { margin:0 0 14px; }
  p { line-height:1.45; }
  .muted { color:#67676b; }
  label { font-weight:650; display:block; margin:16px 0 7px; }
  input, select, textarea { width:100%; box-sizing:border-box; font:inherit; padding:12px 13px; border:1px solid #b9bac0; border-radius:10px; background:#fff; }
  textarea { min-height:155px; resize:vertical; }
  button { font:inherit; font-weight:700; border:0; border-radius:11px; padding:12px 18px; cursor:pointer; background:#1d1d1f; color:white; }
  button.secondary { background:#e8e8ed; color:#1d1d1f; }
  button:disabled { opacity:.4; cursor:not-allowed; }
  .row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
  .row > * { flex:1; }
  #experiment { display:none; }
  .status { display:flex; justify-content:space-between; gap:12px; margin-bottom:14px; color:#67676b; font-weight:600; }
  #taskBox { min-height:52px; padding:12px 14px; margin:12px 0; border-radius:10px; background:#f2f2f7; }
  #stimulusArea { min-height:310px; display:flex; align-items:center; justify-content:center; text-align:center; border-radius:18px; background:#101012; color:#fff; margin:18px 0; padding:24px; }
  #stimulus { font-size:86px; font-weight:760; letter-spacing:.03em; max-width:900px; }
  #stimulus.idiom { font-size:38px; line-height:1.25; letter-spacing:0; }
  #countdown { font-size:52px; font-weight:750; }
  #responsePanel { display:none; }
  #saved { color:#166534; font-weight:700; min-height:24px; }
  .warning { background:#fff7ed; border:1px solid #fed7aa; border-radius:10px; padding:10px 12px; }
  code { background:#efeff4; padding:2px 5px; border-radius:5px; }
</style>
</head>
<body>
<div class="page">
  <div id="setup" class="card">
    <h1>Human Memory Experiment</h1>
    <p class="muted">Visual presentation only. Responses are saved automatically after every trial.</p>

    <label for="participant">Participant ID</label>
    <input id="participant" placeholder="e.g. P01" autocomplete="off">

    <label for="block">Experiment block</label>
    <select id="block"></select>

    <p class="warning">The participant should not write anything down during presentation. For serial recall, enter items in the exact order remembered.</p>
    <button id="beginBtn">Begin block</button>
  </div>

  <div id="experiment" class="card">
    <div class="status">
      <span id="blockLabel"></span>
      <span id="progress"></span>
    </div>
    <h2 id="phaseTitle">Ready</h2>
    <div id="taskBox"></div>
    <div id="stimulusArea">
      <div>
        <div id="stimulus">+</div>
        <div id="countdown"></div>
      </div>
    </div>

    <div id="preTrial">
      <p id="trialInstructions"></p>
      <button id="startTrialBtn">Start trial</button>
    </div>

    <div id="responsePanel">
      <label for="response">Your response</label>
      <textarea id="response" autocomplete="off" spellcheck="false"></textarea>
      <p id="responseHint" class="muted"></p>
      <div class="row">
        <button id="submitBtn">Save answer and continue</button>
      </div>
    </div>
    <div id="saved"></div>
  </div>
</div>

<script>
const blocks = __BLOCKS__;
const blockSelect = document.getElementById('block');
Object.entries(blocks).forEach(([key,label]) => {
  const o = document.createElement('option'); o.value=key; o.textContent=label; blockSelect.appendChild(o);
});

let sessionId = null;
let participantId = null;
let trials = [];
let index = 0;
let currentTrial = null;
let busy = false;

const sleep = ms => new Promise(r => setTimeout(r, ms));

function setStimulus(text, idiom=false) {
  const el = document.getElementById('stimulus');
  el.textContent = text;
  el.className = idiom ? 'idiom' : '';
}

function taskInstruction(t) {
  if (!t.secondary_task) return 'Keep your attention on the centre of the screen.';
  return '<strong>Secondary task:</strong> ' + t.secondary_task + '.';
}

function trialInstruction(t) {
  if (t.response_kind === 'idioms') {
    return 'You will see 12 idioms, one at a time. Remember both the idioms and their order.';
  }
  if (t.block === 'Free recall') {
    return 'You will see ' + t.items.length + ' letters, one at a time. Recall as many as possible; order does not matter.';
  }
  return 'You will see ' + t.items.length + ' letters, one at a time. Recall them in the same order.';
}

async function beginBlock() {
  participantId = document.getElementById('participant').value.trim();
  if (!participantId) { alert('Enter a Participant ID first.'); return; }
  const block = blockSelect.value;
  const res = await fetch('/api/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({participant_id:participantId, block})});
  const data = await res.json();
  if (!res.ok) { alert(data.error || 'Could not start experiment.'); return; }
  sessionId = data.session_id; trials = data.trials; index = 0;
  document.getElementById('setup').style.display='none';
  document.getElementById('experiment').style.display='block';
  document.getElementById('blockLabel').textContent = blocks[block];
  prepareTrial();
}

function prepareTrial() {
  busy = false;
  document.getElementById('saved').textContent='';
  document.getElementById('responsePanel').style.display='none';
  document.getElementById('preTrial').style.display='block';
  document.getElementById('startTrialBtn').disabled=false;
  document.getElementById('countdown').textContent='';
  setStimulus('+');

  if (index >= trials.length) {
    finishBlock(); return;
  }
  currentTrial = trials[index];
  document.getElementById('progress').textContent = `Trial ${index+1} of ${trials.length}`;
  document.getElementById('phaseTitle').textContent = 'Ready';
  document.getElementById('taskBox').innerHTML = taskInstruction(currentTrial);
  document.getElementById('trialInstructions').textContent = trialInstruction(currentTrial);
}

async function runTrial() {
  if (busy) return;
  busy = true;
  document.getElementById('startTrialBtn').disabled=true;
  document.getElementById('phaseTitle').textContent='Presentation';
  document.getElementById('preTrial').style.display='none';
  document.getElementById('responsePanel').style.display='none';
  document.getElementById('saved').textContent='';

  await sleep(800);
  const idiom = currentTrial.response_kind === 'idioms';
  for (const item of currentTrial.items) {
    setStimulus(item, idiom);
    const onMs = idiom ? Math.min(2200, currentTrial.interval_s*1000*0.8) : Math.min(750, currentTrial.interval_s*1000*0.75);
    await sleep(onMs);
    setStimulus('');
    await sleep(Math.max(0, currentTrial.interval_s*1000 - onMs));
  }

  if (currentTrial.pre_recall_delay_s > 0) {
    await runDelay(currentTrial.pre_recall_delay_s, currentTrial);
  }
  enableResponse();
}

async function runDelay(seconds, t) {
  document.getElementById('phaseTitle').textContent = t.secondary_task ? 'Distractor task' : 'Pause before recall';
  if (t.secondary_task) {
    document.getElementById('taskBox').innerHTML = '<strong>Do this now:</strong> ' + t.secondary_task + '.';
  } else {
    document.getElementById('taskBox').textContent = 'Wait until the countdown finishes. Do not rehearse the sequence deliberately.';
  }
  setStimulus('');
  for (let s=seconds; s>0; s--) {
    document.getElementById('countdown').textContent = s;
    await sleep(1000);
  }
  document.getElementById('countdown').textContent='';
}

function enableResponse() {
  document.getElementById('phaseTitle').textContent='Recall';
  document.getElementById('taskBox').textContent='Enter your answer now.';
  document.getElementById('responsePanel').style.display='block';
  const box = document.getElementById('response');
  box.value='';
  if (currentTrial.response_kind === 'idioms') {
    document.getElementById('responseHint').textContent='Enter one idiom per line, in the order remembered.';
  } else if (currentTrial.block === 'Free recall') {
    document.getElementById('responseHint').textContent='Example: A J K R S. Order does not matter.';
  } else {
    document.getElementById('responseHint').textContent='Example: A J K R S. Enter the letters in recalled order.';
  }
  box.focus();
  busy=false;
}

async function submitResponse() {
  if (busy) return;
  const raw = document.getElementById('response').value.trim();
  busy=true;
  document.getElementById('submitBtn').disabled=true;
  const res = await fetch('/api/save', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({session_id:sessionId, trial_id:currentTrial.trial_id, response:raw})});
  const data = await res.json();
  document.getElementById('submitBtn').disabled=false;
  if (!res.ok) { busy=false; alert(data.error || 'Could not save response.'); return; }
  document.getElementById('saved').textContent='Saved.';
  index += 1;
  await sleep(450);
  prepareTrial();
}

function finishBlock() {
  document.getElementById('phaseTitle').textContent='Block complete';
  document.getElementById('progress').textContent='';
  document.getElementById('taskBox').innerHTML='All responses were saved automatically to <code>human_memory_results.csv</code> and <code>human_memory_item_level.csv</code>.';
  document.getElementById('stimulusArea').style.display='none';
  document.getElementById('preTrial').innerHTML='<button class="secondary" onclick="location.reload()">Run another block</button>';
  document.getElementById('preTrial').style.display='block';
  document.getElementById('responsePanel').style.display='none';
}

document.getElementById('beginBtn').addEventListener('click', beginBlock);
document.getElementById('startTrialBtn').addEventListener('click', runTrial);
document.getElementById('submitBtn').addEventListener('click', submitResponse);
</script>
</body>
</html>'''


# -----------------------------------------------------------------------------
# HTTP API
# -----------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        # Keep the terminal clean; only show startup/errors we explicitly print.
        return

    def send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            html = HTML.replace("__BLOCKS__", json.dumps(BLOCKS, ensure_ascii=False))
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/health":
            self.send_json({"ok": True})
        else:
            self.send_error(404)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            data = self.read_json()
            if path == "/api/start":
                self.api_start(data)
            elif path == "/api/save":
                self.api_save(data)
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def api_start(self, data: dict) -> None:
        participant = str(data.get("participant_id", "")).strip()
        block_key = str(data.get("block", "")).strip()
        if not participant:
            raise ValueError("Participant ID is required.")
        if block_key not in BLOCKS:
            raise ValueError("Choose a valid experiment block.")

        # Deterministic block-order seed for reproducible participant sessions,
        # combined with fresh randomness within the generated trial sequences.
        seed_text = f"{participant}|{block_key}|{uuid.uuid4()}"
        random.seed(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16))
        trials = build_trials(block_key)
        session_id = str(uuid.uuid4())

        with SESSION_LOCK:
            SESSIONS[session_id] = {
                "participant_id": participant,
                "block_key": block_key,
                "trials": {t.trial_id: t for t in trials},
                "saved": set(),
            }

        self.send_json({
            "session_id": session_id,
            "trials": [public_trial(t) for t in trials],
        })

    def api_save(self, data: dict) -> None:
        session_id = str(data.get("session_id", ""))
        trial_id = str(data.get("trial_id", ""))
        response = str(data.get("response", ""))

        with SESSION_LOCK:
            session = SESSIONS.get(session_id)
            if not session:
                raise ValueError("Session not found. Restart the block.")
            if trial_id in session["saved"]:
                # Idempotent: browser double-clicks do not duplicate rows.
                self.send_json({"ok": True, "already_saved": True})
                return
            trial = session["trials"].get(trial_id)
            if not trial:
                raise ValueError("Trial not found.")
            participant = session["participant_id"]

        save_trial(participant, session_id, trial, response)

        with SESSION_LOCK:
            session["saved"].add(trial_id)

        self.send_json({"ok": True})


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main() -> None:
    # Validate idiom file early so the user gets a useful message before testing.
    if not IDIOMS_CSV.exists():
        print(f"WARNING: {IDIOMS_CSV.name} was not found.")
        print("Letter experiments will work, but idiom blocks will not until the file is added.\n")

    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as exc:
        raise SystemExit(
            f"Could not start the experiment server on port {PORT}.\n"
            f"If another copy is running, stop it with Ctrl+C and try again.\n\n{exc}"
        )

    url = f"http://{HOST}:{PORT}"
    print("Human Memory Experiment — VISUAL ONLY")
    print(f"Open: {url}")
    print(f"Main results: {RESULTS_CSV.name}")
    print(f"Item-level results: {ITEM_RESULTS_CSV.name}")
    print("Press Ctrl+C in this terminal when you are completely finished.\n")

    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nExperiment server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
