"""Per-session sequence extraction for the HGF.

Turns one session's ``Trials_Sync`` table into the arrays the HGF needs:

  * ``u``         — the binary perceptual input: did the GAMBLE arm pay its big
                    reward? Defined on gamble-choice trials (= ``Rewarded``);
                    on safe-choice trials the gamble outcome is UNOBSERVED.
  * ``observed``  — mask, 1 where the gamble outcome was observed (gamble chosen)
                    and 0 where it was not (safe chosen). Partial feedback.
  * ``y``         — the participant's choice (1 = gamble, 0 = safe), the target
                    of the response model.
  * ``p_schedule``— the true scheduled P(big reward) of the gamble arm per trial
                    (for plotting / ground-truth overlay).

Only RESPONDING trials are kept (``NotResponding == 0``); non-responding trials
carry neither a choice nor an outcome and are dropped so the belief updates stay
clean. Beliefs are reset to the prior at the start of every session (sessions are
weeks apart with re-randomised schedules), so each session is an independent
sequence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# Repo root importable when run via ``python -m analysis.hgf.<mod>``.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from session import Session
from .config import REPO_ROOT


@dataclass
class SessionData:
    """Arrays for one session, responding trials only.

    Attributes
    ----------
    session_id : str
    u : np.ndarray
        Binary perceptual input (gamble big-reward outcome), float ``0/1``;
        the value on unobserved (safe-choice) trials is a placeholder (0.0) and
        is ignored by the HGF because ``observed`` is 0 there.
    observed : np.ndarray
        Mask (int ``0/1``); 1 where the gamble outcome was observed.
    y : np.ndarray
        Participant choice (float ``1`` = gamble, ``0`` = safe).
    p_schedule : np.ndarray
        True scheduled P(big reward) of the gamble arm per trial.
    trial_index : np.ndarray
        Original (pre-filter) trial indices of the kept trials.
    n_trials : int
    """

    session_id: str
    u: np.ndarray
    observed: np.ndarray
    y: np.ndarray
    p_schedule: np.ndarray
    trial_index: np.ndarray
    n_trials: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_trials = int(len(self.y))

    # Convenience views ----------------------------------------------------
    @property
    def n_gamble(self) -> int:
        return int(np.sum(self.y == 1))

    @property
    def n_safe(self) -> int:
        return int(np.sum(self.y == 0))

    def __repr__(self) -> str:
        return (
            f"SessionData(session_id={self.session_id!r}, n_trials={self.n_trials}, "
            f"n_gamble={self.n_gamble}, n_safe={self.n_safe})"
        )


def list_sessions(data_root: str = REPO_ROOT) -> list[str]:
    """Sorted list of ``YYYYMMDD`` session directories under ``data_root``."""
    return sorted(
        d for d in os.listdir(data_root)
        if len(d) == 8 and d.isdigit() and os.path.isdir(os.path.join(data_root, d))
    )


def load_session(session_id: str, data_root: str = REPO_ROOT) -> SessionData:
    """Extract the HGF sequence for one session (responding trials only)."""
    sess = Session(session_id, data_root=data_root)
    trials = sess.trials
    responding = sess.responding_mask

    arm = trials["ChosenArm_G1S0"].to_numpy()          # 1 = gamble, 0 = safe
    rewarded = trials["Rewarded"].to_numpy()           # 1 = paid out
    p_sched = trials["P_BigReward_Gamble"].to_numpy()  # scheduled gamble prob

    idx = np.where(responding)[0]
    arm_r = arm[idx].astype(float)
    rew_r = rewarded[idx].astype(float)

    observed = (arm_r == 1).astype(int)                # gamble outcome observed
    u = np.where(observed == 1, rew_r, 0.0).astype(float)
    y = arm_r.astype(float)

    return SessionData(
        session_id=session_id,
        u=u,
        observed=observed,
        y=y,
        p_schedule=p_sched[idx].astype(float),
        trial_index=idx,
    )


def load_all_sessions(
    session_ids: list[str] | None = None, data_root: str = REPO_ROOT
) -> list[SessionData]:
    """Load :class:`SessionData` for every session (or a given subset)."""
    if session_ids is None:
        session_ids = list_sessions(data_root)
    return [load_session(s, data_root=data_root) for s in session_ids]
