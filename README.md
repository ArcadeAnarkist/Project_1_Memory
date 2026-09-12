
# 02464 Human Memory Mini Project – visual experiment kit

Upload these three files directly to the root of a GitHub repository — no ZIP file and no folders are required:

- `memory_experiment.py` — runs the experiment and automatically stores CSV data.
- `memory_analysis.ipynb` — pools participants, plots the effects, and computes 95% bootstrap confidence intervals.
- The experiment creates `<ID>_trials.csv` and `<ID>_items.csv` beside the Python file.

## Run in VS Code

Download or clone the repository, open it in VS Code, then run:

```bash
python memory_experiment.py
```

No extra package install is needed for the runner itself. It uses only the Python standard library (`tkinter`, `csv`, `random`, etc.).

For the analysis notebook, use an environment with:

```bash
python -m pip install numpy pandas matplotlib ipykernel
```

---

## What the program now automates

- fixed block order
- automatic trial generation
- automatic on-screen instructions for each block
- automatic working-memory distractor task
- automatic response windows
- automatic short breaks between conditions
- automatic CSV logging at both trial level and item level

So the participant flow is basically:
1. enter participant ID
2. choose block
3. watch stimuli
4. type response
5. continue to the next trial / block

---

## Current experiment design

### Free recall
All items are presented **visually**.

1. Baseline  
   - 5 trials
   - 16 letters
   - 3 s/item
   - immediate recall
2. Unfilled delay  
   - 5 trials
   - 16 letters
   - 3 s/item
   - 60 s delay before recall
3. Fast presentation  
   - 5 trials
   - 16 letters
   - 1 s/item
   - immediate recall
4. Working-memory interference  
   - 5 trials
   - 16 letters
   - 3 s/item
   - multiplication task before recall

Primary measures:
- **Primacy index** = start accuracy − middle accuracy
- **Recency index** = end accuracy − middle accuracy

### Serial recall
All items are presented **visually**.

1. Capacity block  
   - lengths 4–9
   - 2 trials per length
2. Baseline  
   - 5 trials
   - 12 letters
3. Finger tapping  
   - 5 trials
   - same as baseline + repeated SPACE tapping
4. Articulatory suppression  
   - 5 trials
   - same as baseline + repeated “TAH-DAH-TAH-DAH...”
5. Wordsporg / idioms  
   - 3 trials of **real idioms**
   - 3 trials of **fake idioms**
   - 12 idioms per trial
   - response typed one idiom per line

---

## Display behaviour

- no `+` fixation sign
- no blank screen between ordinary stimuli
- each stimulus stays on screen until the next one replaces it
- if the exact same item appears twice in a row, the program inserts a very brief blink so the participant can perceive the repetition

---

## Files saved

Each participant gets two files in the repository root, beside `memory_experiment.py`:

### 1. `<ID>_trials.csv`
One row per trial. Useful for condition-level summaries.

Contains variables such as:
- participant
- experiment
- condition
- trial
- n_items
- interval_s
- delay_s
- sequence
- response
- free_accuracy
- serial_accuracy
- primacy_index
- recency_index
- wm_questions
- wm_correct
- tap_count

### 2. `<ID>_items.csv`
One row per presented item. Useful for serial-position curves and error analysis.

Contains variables such as:
- serial_position
- region
- presented_item
- response_at_position
- free_recalled
- serial_correct
- substitution

---

## Why the idiom block is separate

The idiom block is included because you asked to add the **wordsporg** part.  
It is logged as a separate experiment name: `idiom_serial_recall`.

That makes analysis easier:
- normal letter-based serial recall stays clean
- idiom memory can be analysed separately
- real idioms can be compared directly with fake idioms

---

## Suggested workflow

1. Run a **pilot** on 1–2 people.
2. Check whether:
   - the total duration is still acceptable
   - baseline accuracy is neither near 0 nor near 100%
   - participants understand the idiom response format
   - articulatory suppression is feasible
3. Only then lock the final number of repetitions.

If the full experiment becomes too long, the first thing to cut is usually the extra exploratory block(s), not the baseline conditions.


## New: response limits

For letter-based tasks, the response box now prevents the participant from entering more letter symbols than were shown.

Examples:
- 16-item free recall → maximum 16 letter symbols
- 12-item serial recall → maximum 12 letter symbols
- capacity trial with 7 items → maximum 7 letter symbols

Spaces and commas are allowed and do not count toward the limit.

For the wordsporg / idiom task, the multiline response box allows at most the same number of responses as idioms shown in that trial.

## New: graphical experiment selection

At startup, the participant/researcher now gets a checkbox menu where they can choose exactly which experiments to run.

The selected experiments still run in the intended fixed order:
1. free recall baseline
2. free recall pause
3. free recall fast rate
4. free recall working-memory interference
5. serial capacity
6. serial baseline
7. finger tapping
8. articulatory suppression
9. real idioms
10. fake idioms

## Repeated letters

Letters are sampled **with replacement**, so the same letter may appear more than once in a sequence.

Free recall uses multiset scoring: if `A` was shown three times and the participant enters two A's, exactly two recalled items are counted. For serial-position analysis, that score is distributed equally across the three A positions, so a repeated letter does not unfairly boost the beginning or end of the sequence.
