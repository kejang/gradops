from importlib import resources as ir

import numpy as np
import cupy as cp

from .utils import to_device_array, from_device_array


module_path = ir.files(__package__)
cuda_dir = module_path / "cuda"
n_devices = cp.cuda.runtime.getDeviceCount()

from .utils import (
    read_cu_file,
    compile_rawkernel_per_device,
    to_device_array,
    from_device_array,
)


_KERNEL_SPECS = {
    "d_mag_grad_1d_real": ("d_mag_grad_1d_real.cu", "d_mag_grad_1d_real"),
    "d_mag_grad_2d_real": ("d_mag_grad_2d_real.cu", "d_mag_grad_2d_real"),
    "d_mag_grad_3d_real": ("d_mag_grad_3d_real.cu", "d_mag_grad_3d_real"),
    "d_mag_grad_1d_complex": ("d_mag_grad_1d_complex.cu", "d_mag_grad_1d_complex"),
    "d_mag_grad_2d_complex": ("d_mag_grad_2d_complex.cu", "d_mag_grad_2d_complex"),
    "d_mag_grad_3d_complex": ("d_mag_grad_3d_complex.cu", "d_mag_grad_3d_complex"),
}

_globals = globals()
for varname, (cu_file, func_name) in _KERNEL_SPECS.items():
    _globals[varname] = compile_rawkernel_per_device(
        read_cu_file(cuda_dir, cu_file), func_name, n_devices
    )


def _select_absgrad_kernels(ndim: int, is_complex: bool) -> list:
    if ndim not in (1, 2, 3):
        raise ValueError(f"FiniteDifferenceMagOp supports ndim <= 3, got {ndim}")
    suffix = "complex" if is_complex else "real"
    return _globals[f"d_mag_grad_{ndim}d_{suffix}"]


class FiniteDifferenceMagOp:
    def __init__(self, sz_im, is_complex, gpu_id, stream):
        self.sz_im = tuple(int(s) for s in sz_im)
        self.ndim = len(self.sz_im)
        self.n = np.int64(np.prod(self.sz_im))
        self.is_complex = bool(is_complex)
        self.dtype = cp.float32
        self.gpu_id = int(gpu_id)
        self.stream = stream

        # dims (nx, ny, nz)
        if self.ndim == 1:
            self.nz, self.ny, self.nx = 1, 1, self.sz_im[0]
        elif self.ndim == 2:
            self.nz, self.ny, self.nx = 1, self.sz_im[-2], self.sz_im[-1]
        elif self.ndim == 3:
            self.nz, self.ny, self.nx = self.sz_im[-3], self.sz_im[-2], self.sz_im[-1]
        else:
            raise ValueError("FiniteDifferenceMagOp Error: only support dim <= 3")

        self.kernels = _select_absgrad_kernels(self.ndim, self.is_complex)

        with cp.cuda.Device(self.gpu_id), self.stream as _:
            self.block_events = cp.cuda.Event(block=True)

        if (self.ny == 1) and (self.nz == 1):
            self.blk = (256, 1, 1)
        elif self.nz == 1:
            self.blk = (16, 16, 1)
        else:
            self.blk = (8, 8, 8)

        self.grd = (
            int((self.nx + self.blk[0] - 1) / self.blk[0]),
            int((self.ny + self.blk[1] - 1) / self.blk[1]),
            int((self.nz + self.blk[2] - 1) / self.blk[2]),
        )

    def run(self, x):
        with cp.cuda.Device(self.gpu_id), self.stream as _:
            d_y = cp.empty((self.n,), dtype=self.dtype)

        d_x = to_device_array(
            x,
            gpu_id=self.gpu_id,
            stream=self.stream,
            dtype_real="float32",
            dtype_complex="complex64",
        )

        if self.ndim == 1:
            args = (d_y, d_x, np.uint32(self.nx))
        elif self.ndim == 2:
            args = (d_y, d_x, np.uint32(self.nx), np.uint32(self.ny))
        else:
            args = (
                d_y,
                d_x,
                np.uint32(self.nx),
                np.uint32(self.ny),
                np.uint32(self.nz),
            )

        with cp.cuda.Device(self.gpu_id):
            self.kernels[self.gpu_id](self.grd, self.blk, args, stream=self.stream)

        xp = cp.get_array_module(x)
        if xp == np:
            y = from_device_array(d_y, gpu_id=self.gpu_id, stream=self.stream)
        else:
            y = d_y

        return y.reshape(self.sz_im)
