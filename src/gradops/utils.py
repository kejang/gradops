from importlib.abc import Traversable
from pathlib import Path
import numpy as np
import cupy as cp
from cupy.cuda import get_cuda_path


def read_cu_file(cuda_dir: Traversable, filename: str) -> str:
    try:
        return (cuda_dir / filename).read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Missing CUDA source '{filename}'. Expected package resource "
            f"'{__package__}/cuda/{filename}'. Ensure it's included as package-data."
        ) from e


def compile_rawkernel(src: str, kernel_name: str, gpu_id: int):

    include_dir = str(Path(get_cuda_path()) / "include")
    options = ("--std=c++11", f"--include-path={include_dir}")
    with cp.cuda.Device(gpu_id):
        kernel = cp.RawKernel(src, kernel_name, options=options)
    return kernel


def to_device_array(x, gpu_id, stream, dtype_real="float32", dtype_complex="complex64"):

    xp = cp.get_array_module(x)

    if xp.iscomplexobj(x):
        dtype = cp.dtype(dtype_complex)
    else:
        dtype = cp.dtype(dtype_real)

    if xp == np:  # numpy -> cupy
        with cp.cuda.Device(gpu_id), stream as stream:
            block_events = cp.cuda.Event(block=True)
            d_x = cp.empty(x.shape, dtype=dtype)
            d_x.set(x.astype(dtype), stream=stream)

            block_events.record(stream)
            block_events.synchronize()
            block_events = None

    else:  # already on device
        with cp.cuda.Device(gpu_id), stream as stream:
            d_x = x.astype(dtype)

    return d_x


def from_device_array(d_x, gpu_id, stream):

    xp = cp.get_array_module(d_x)

    if xp == np:
        x = d_x.copy()  # already on host

    else:
        with cp.cuda.Device(gpu_id), stream as stream:
            block_events = cp.cuda.Event(block=True)

            x = np.empty(d_x.shape, dtype=d_x.dtype)
            d_x.get(out=x, stream=stream)

            block_events.record(stream)
            block_events.synchronize()
            block_events = None

    return x
