"""Session: centralised per-session data loading and alignment helpers.

Usage
-----
    from session import Session

    sess = Session("20250602")            # uses REPO_ROOT as data_root
    sess = Session("20250602", data_root) # explicit root

    sess.trials          # pd.DataFrame  (Trials_Sync.mat)
    sess.sampling_rate   # int  (Hz)
    sess.spike_trains    # (list[ndarray], list[str])

    sess.event_times("cue")                              # seconds, NaN for non-responding
    sess.responding_mask                                 # boolean ndarray
    sess.marker_times_ms("cue", pre_ms=500, post_ms=1000)
    sess.condition_masks()
    sess.condition_event_times(event="reward")
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np

from utils import (
    REPO_ROOT,
    EVENTS,
    load_sr,
    load_trials_sync,
    get_spike_trains,
    event_times as _event_times,
    condition_masks as _condition_masks,
    condition_event_times as _condition_event_times,
)


class Session:
    def __init__(self, session_id: str, data_root=REPO_ROOT):
        self.id   = session_id
        self.root = Path(data_root)

    @property
    def _data_dir(self) -> str:
        return str(self.root / self.id)

    @cached_property
    def trials(self):
        return load_trials_sync(data_dir=self._data_dir)

    @cached_property
    def sampling_rate(self) -> int:
        return load_sr(data_dir=self._data_dir)["SamplingRate_Hz"].iloc[0]

    @cached_property
    def spike_trains(self):
        """(trains, labels): list of 1-D spike-time arrays (seconds) + column labels."""
        return get_spike_trains(data_dir=self._data_dir)

    @cached_property
    def responding_mask(self) -> np.ndarray:
        """Boolean array: True for trials where NotResponding == 0."""
        return (self.trials["NotResponding"] == 0).to_numpy()

    def event_times(self, event: str) -> np.ndarray:
        """Event times in seconds (one per trial); non-responding trials are NaN."""
        times = _event_times(self.trials, self.sampling_rate, event)
        return np.where(self.responding_mask, times, np.nan)

    def marker_times_ms(self, align_event: str, pre_ms: float, post_ms: float) -> dict:
        """Mean relative timing (ms) of all other events within the plot window."""
        align = self.event_times(align_event)
        markers: dict[str, float] = {}
        for name in EVENTS:
            if name == align_event:
                continue
            rel = _event_times(self.trials, self.sampling_rate, name) - align
            if not np.any(np.isfinite(rel)):
                continue
            mean_rel_ms = float(np.nanmean(rel)) * 1000
            if -pre_ms <= mean_rel_ms <= post_ms:
                markers[name] = mean_rel_ms
        return markers

    def condition_masks(self) -> dict:
        return _condition_masks(self.trials)

    def condition_event_times(self, event: str = "reward") -> dict:
        return _condition_event_times(self.trials, self.sampling_rate, event)

    def aligned_event_times(self, align: str, condition: str | None = None) -> np.ndarray:
        """Event times (s) to align on, optionally filtered to one condition.

        condition=None  -> all responding trials (non-responding are NaN and are
                           skipped by the downstream compute_* functions).
        condition=<key> -> only trials in that CONDITIONS key (e.g. "G+R").
        """
        times = self.event_times(align)
        if condition is None:
            return times
        masks = self.condition_masks()
        if condition not in masks:
            raise ValueError(f"condition must be None or one of {list(masks)}")
        return times[masks[condition]]

    def neuron(self, idx: int):
        """Return a NeuronView — the fluent single-neuron plotting entry point.

            sess.neuron(7).psth(condition="G+R", align="reward").save()
        """
        from explore import NeuronView   # lazy: explore imports Session
        return NeuronView(self, idx)

    def __repr__(self) -> str:
        return f"Session(id={self.id!r})"
