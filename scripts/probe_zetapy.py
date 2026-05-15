"""Probe the installed zetapy API to confirm return signatures."""
import numpy as np
from zetapy import zetatest, ifr as zeta_ifr
import inspect

if __name__ == '__main__':
    print("=== zetatest signature ===")
    print(inspect.signature(zetatest))
    print("=== ifr signature ===")
    print(inspect.signature(zeta_ifr))

    rng = np.random.default_rng(0)
    spike_times = np.sort(rng.uniform(0, 20, 100))
    event_times = np.arange(0, 20, 1.0)

    print("\n--- default call (boolParallel=False) ---")
    result = zetatest(spike_times, event_times, dblUseMaxDur=0.9, intResampNum=20, boolParallel=False)
    print(f"type: {type(result)}, len: {len(result)}")
    for i, v in enumerate(result):
        if isinstance(v, (int, float, np.floating, np.integer)):
            print(f"  [{i}] {type(v).__name__} = {v}")
        elif isinstance(v, np.ndarray):
            print(f"  [{i}] ndarray shape={v.shape} dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  [{i}] dict keys={list(v.keys())}")
        else:
            print(f"  [{i}] {type(v).__name__} = {v!r}")

    print("\n--- boolReturnRate=True ---")
    result2 = zetatest(spike_times, event_times, dblUseMaxDur=0.9, intResampNum=20,
                       boolReturnRate=True, boolParallel=False)
    print(f"type: {type(result2)}, len: {len(result2)}")
    for i, v in enumerate(result2):
        if isinstance(v, (int, float, np.floating, np.integer)):
            print(f"  [{i}] {type(v).__name__} = {v}")
        elif isinstance(v, np.ndarray):
            print(f"  [{i}] ndarray shape={v.shape} dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  [{i}] dict keys={list(v.keys())}")
        else:
            print(f"  [{i}] {type(v).__name__} = {v!r}")
