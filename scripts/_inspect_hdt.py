"""Throwaway: recurse into the FileWrapper struct to find table columns."""
import numpy as np
import scipy.io
from io import BytesIO
from scipy.io.matlab._mio5 import MatFile5Reader

path = "20250714/Human_Data_Table.mat"
d = scipy.io.loadmat(path)
fw = d["__function_workspace__"].ravel().tobytes()
header = b" " * 116 + b"\x00" * 8 + b"\x00\x01" + b"IM"
reader = MatFile5Reader(BytesIO(header + fw[8:]), struct_as_record=False, squeeze_me=True)
reader.mat_stream.seek(0)
reader.initialize_read()
reader.read_file_header()

hdr, nxt = reader.read_var_header()
struct = reader.read_var_array(hdr)


def describe(x, prefix="", depth=0):
    if depth > 5:
        return
    pad = "  " * depth
    if hasattr(x, "_fieldnames"):
        print(f"{pad}{prefix} mat_struct fields={x._fieldnames}")
        for f in x._fieldnames:
            describe(getattr(x, f), prefix=f, depth=depth + 1)
    elif isinstance(x, np.ndarray) and x.dtype == object:
        print(f"{pad}{prefix} cell shape={x.shape}")
        for j, e in enumerate(x.ravel()[:60]):
            describe(e, prefix=f"[{j}]", depth=depth + 1)
    elif isinstance(x, np.ndarray):
        prev = x.ravel()[:4] if x.size else x
        print(f"{pad}{prefix} ndarray shape={x.shape} dtype={x.dtype} first={prev}")
    else:
        print(f"{pad}{prefix} {type(x).__name__}={x!r}"[:120])


describe(struct, prefix="ROOT")
