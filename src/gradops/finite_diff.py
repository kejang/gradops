from importlib import resources as ir

import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator

from .utils import (
    read_cu_file,
    compile_rawkernel_per_device,
    to_device_array,
    from_device_array,
)


module_path = ir.files(__package__)
cuda_dir = module_path / "cuda"
n_devices = cp.cuda.runtime.getDeviceCount()


_KERNEL_SPECS = {
    "d_x_real": ("d_x_real.cu", "d_x_real"),
    "d_y_real": ("d_y_real.cu", "d_y_real"),
    "d_z_real": ("d_z_real.cu", "d_z_real"),
    "dt_x_real": ("dt_x_real.cu", "dt_x_real"),
    "dt_y_real": ("dt_y_real.cu", "dt_y_real"),
    "dt_z_real": ("dt_z_real.cu", "dt_z_real"),
    "dtd_x_real": ("dtd_x_real.cu", "dtd_x_real"),
    "dtd_y_real": ("dtd_y_real.cu", "dtd_y_real"),
    "dtd_z_real": ("dtd_z_real.cu", "dtd_z_real"),
    "d_x_complex": ("d_x_complex.cu", "d_x_complex"),
    "d_y_complex": ("d_y_complex.cu", "d_y_complex"),
    "d_z_complex": ("d_z_complex.cu", "d_z_complex"),
    "dt_x_complex": ("dt_x_complex.cu", "dt_x_complex"),
    "dt_y_complex": ("dt_y_complex.cu", "dt_y_complex"),
    "dt_z_complex": ("dt_z_complex.cu", "dt_z_complex"),
    "dtd_x_complex": ("dtd_x_complex.cu", "dtd_x_complex"),
    "dtd_y_complex": ("dtd_y_complex.cu", "dtd_y_complex"),
    "dtd_z_complex": ("dtd_z_complex.cu", "dtd_z_complex"),
}

_globals = globals()
for varname, (cu_file, func_name) in _KERNEL_SPECS.items():
    _globals[varname] = compile_rawkernel_per_device(
        read_cu_file(cuda_dir, cu_file), func_name, n_devices
    )


def get_kernels(is_complex: bool, is_normal: bool, direction: str):
    if direction not in ("x", "y", "z"):
        raise ValueError(f"direction must be one of ('x','y','z'), got {direction!r}")

    suffix = "complex" if is_complex else "real"

    if is_normal:
        k = _globals[f"dtd_{direction}_{suffix}"]
        return k, k

    return _globals[f"d_{direction}_{suffix}"], _globals[f"dt_{direction}_{suffix}"]


class _FiniteDifferenceOp:
    def __init__(self, direction, sz_im, dtype, gpu_id, stream, edge):
        self.direction = direction
        self.dtype = cp.dtype(dtype)
        self.n = np.int64(np.prod(sz_im))
        self.edge = edge
        self.gpu_id = int(gpu_id)
        self.stream = stream

        if len(sz_im) == 1:
            self.nz, self.ny, self.nx = 1, 1, int(sz_im[0])
        elif len(sz_im) == 2:
            self.nz, self.ny, self.nx = 1, int(sz_im[-2]), int(sz_im[-1])
        elif len(sz_im) == 3:
            self.nz, self.ny, self.nx = int(sz_im[-3]), int(sz_im[-2]), int(sz_im[-1])
        else:
            raise ValueError("sz_im error: only support dim <= 3")

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

    def run(self, x, kernel):
        with cp.cuda.Device(self.gpu_id), self.stream as _:
            d_y = cp.empty((self.n,), dtype=self.dtype)

        d_x = to_device_array(
            x,
            gpu_id=self.gpu_id,
            stream=self.stream,
            dtype_real="float32",
            dtype_complex="complex64",
        )

        args = (
            d_y,
            d_x,
            np.uint32(self.nx),
            np.uint32(self.ny),
            np.uint32(self.nz),
            np.int32(self.edge),
        )

        with cp.cuda.Device(self.gpu_id):
            kernel[self.gpu_id](self.grd, self.blk, args, stream=self.stream)

        xp = cp.get_array_module(x)
        if xp == np:
            return from_device_array(
                d_y, gpu_id=self.gpu_id, stream=self.stream
            ).astype(x.dtype)
        return d_y


class FiniteDifferenceOp(LinearOperator):

    def __init__(
        self, direction, sz_im, dtype, gpu_id, stream, edge=False, is_normal=False
    ):
        self.direction = direction
        self.edge = edge
        self.dtype = cp.dtype(dtype)

        if self.dtype not in (cp.float32, cp.complex64):
            raise TypeError(
                f"Expected float32/complex64 for these CUDA kernels, got {self.dtype}."
            )

        is_complex = np.iscomplexobj(np.ones((1,), dtype=dtype))
        self.kernel_forward, self.kernel_adjoint = get_kernels(
            is_complex, is_normal, direction
        )

        self.d_op = _FiniteDifferenceOp(
            direction=direction,
            sz_im=sz_im,
            dtype=dtype,
            gpu_id=gpu_id,
            stream=stream,
            edge=edge,
        )

        self.shape = (self.d_op.n, self.d_op.n)

    def _matvec(self, x):
        return self.d_op.run(x, self.kernel_forward)

    def _rmatvec(self, x):
        return self.d_op.run(x, self.kernel_adjoint)
