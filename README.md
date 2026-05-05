# FOR_Jacob_Data

This folder contains scripts and data files prepared for the Human Data project in Simon Jacob's lab. The core session files used across the repository are described below.

---

## Session files

### `STMtx.mat`

Matrix of spike times for all neurons recorded in a session.

- **Size:** `max_spikes × nNeurons` (double)
- **Columns:** one per neuron (STMtx column index = neuron ID used throughout the pipeline)
- **Rows:** spike times in seconds. Each column is padded with `NaN` at the bottom to fill the matrix to uniform size.

---

### `SR.mat`

Sampling rate of the session.

- **Variable:** `SR` — scalar double (e.g. 30000 Hz)
- Used to convert timing columns in `Trials_Sync` (columns 15–19, stored in sampling points) to seconds.

---

### `Trials_Sync.mat`

Behavioural trial table. Each row is one trial, each column is a behavioural variable.

- **Size:** `nTrials × 19` (double)

| Column | Description |
|--------|-------------|
| 1  | Trial Start (seconds from behavioural start) |
| 2  | Trial End (seconds from behavioural start) |
| 3  | Trial duration (seconds, from synchronisation) |
| 4  | Block number |
| 5  | Gamble Arm side (Right = 1, Left = 0) |
| 6  | Probability Big Reward (Gamble arm) |
| 7  | Probability Small Reward (Safe arm) |
| 8  | Amount Big Reward (Gamble arm) |
| 9  | Amount Small Reward (Safe arm) |
| 10 | Number of previous wheel-not-stopping events |
| 11 | Not-responding trial (1 = no response) |
| 12 | Chosen side (Right = 1, Left = 0) — **see warning below** |
| 13 | Chosen arm (Gamble = 1, Safe = 0) |
| 14 | Rewarded trial (1 = rewarded, 0 = not) |
| 15 | Start of the trial (sampling points) |
| 16 | Cue presentation (sampling points) |
| 17 | Start of response window (sampling points) |
| 18 | Reward Period onset (sampling points) |
| 19 | End of the trial (sampling points) |

> **Warning — column 12 (Chosen side):** This column is not reliably extracted and should not be trusted. Use column 13 (Chosen arm: Gamble/Safe) for trial-by-trial choice information.

---

## Derived files

### `Human_Data_Table.mat`

MATLAB table derived from `Trials_Sync` and `SR` by `Scripts_Matlab/TrialsSync_to_HumanTable.m`. Contains one row per trial and only includes columns that can be fully computed from `Trials_Sync` — no pipeline-specific variables (strategy detection, RL outputs, neural data) are included.

All timing values are in **seconds** (sampling-point columns divided by SR).

| Column | Description |
|--------|-------------|
| `F` | Forward arm choice (always 0 — not present in this task) |
| `L` | Left arm chosen (1/0) |
| `R` | Right arm chosen (1/0) |
| `REWARD` | Rewarded trial (1/0) |
| `gamble` | Gamble arm chosen (1/0) |
| `safe` | Safe arm chosen (1/0) |
| `R_sum` | Cumulative right choices |
| `L_sum` | Cumulative left choices |
| `F_sum` | Cumulative forward choices (always 0) |
| `G_sum` | Cumulative gamble choices |
| `S_sum` | Cumulative safe choices |
| `duration` | Trial duration: Reward Period onset − Cue presentation (seconds) |
| `total_sum` | Cumulative non-missed trials |
| `probabilityL` | Reward probability of the Left arm |
| `probabilityR` | Reward probability of the Right arm |
| `probabilityG` | Reward probability of the Gamble arm |
| `probabilityS` | Reward probability of the Safe arm |
| `dropsOfReward` | Reward amount received on this trial (drops) |
| `max_reward` | Maximum possible reward on this trial (Gamble + Safe amounts) |
| `objValueGamble` | Objective value of Gamble arm (= probabilityG) |
| `objValueSafe` | Objective value of Safe arm (= probabilityS) |
| `objectiveValueR` | Objective value of Right arm |
| `objectiveValueL` | Objective value of Left arm |
| `valueDiffObj` | probabilityG − probabilityS |
| `correctDecisions` | 1 = chose the higher-probability arm, 0 = did not, NaN = missed |
| `block_change` | 1 on the first trial of a new block, 0 otherwise |
| `reward` | Reward Period onset (seconds) |
| `Cue_present` | Cue presentation time (seconds) |
| `trial_start` | Trial start time (seconds) |
| `trial_end` | Trial end time (seconds) |
| `wheel_stop` | Start of response window (seconds) |
| `inter_trial_int_start` | Inter-trial interval start (= reward onset) |
| `inter_trial_int_end` | Inter-trial interval end (= trial end) |
| `reward_iti_start` | Reward ITI start (= reward onset) |
| `reward_iti_end` | Reward ITI end (= trial end) |
| `iti_duration` | ITI duration: trial end − reward onset (seconds) |
| `diff_than_previous` | 1 if arm choice differs from previous trial, 0 if same, NaN if either missed |
| `previous_rewarded` | Reward outcome of the previous trial (1/0/NaN) |
| `diff_than_next` | 1 if arm choice differs from next trial, 0 if same, NaN if either missed |
| `consecutive_rewards` | Number of consecutive rewarded trials up to and including this one |
| `consecutive_not_rewarded` | Number of consecutive non-rewarded trials up to and including this one |
| `no_rewards_until_reward` | For reward trials: how many consecutive non-rewards preceded this reward. NaN on non-reward trials |
| `rewards_until_no_reward` | For non-reward trials: how many consecutive rewards preceded this non-reward. NaN on reward trials |
| `choice_change_after_no_reward` | 1 if arm changed on the next trial after a non-reward, 0 if not, NaN if missed |
| `choice_change_after_reward` | 1 if arm changed on the next trial after a reward, 0 if not, NaN if missed |
| `not_REWARD` | 1 − REWARD (non-rewarded trial flag) |
| `not_REWARD_gamble` | Non-rewarded trial on the Gamble arm (1/0) |
| `not_REWARD_safe` | Non-rewarded trial on the Safe arm (1/0) |

Generated by: `Scripts_Matlab/Run_TrialsSync_to_HumanTable.m` → calls `Scripts_Matlab/TrialsSync_to_HumanTable.m` for each session in `Human_Folders.mat`.
