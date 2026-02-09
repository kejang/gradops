from importlib import resources as ir

import numpy as np
import cupy as cp

from .utils import (
    read_cu_file,
    compile_rawkernel,
    to_device_array,
    from_device_array,
)

module_path = ir.files(__package__)
cuda_dir = module_path / "cuda"


_KERNEL_SPECS = {
    "d_mag_grad_1d_real": ("d_mag_grad_1d_real.cu", "d_mag_grad_1d_real"),
    "d_mag_grad_2d_real": ("d_mag_grad_2d_real.cu", "d_mag_grad_2d_real"),
    "d_mag_grad_3d_real": ("d_mag_grad_3d_real.cu", "d_mag_grad_3d_real"),
    "d_mag_grad_1d_complex": ("d_mag_grad_1d_complex.cu", "d_mag_grad_1d_complex"),
    "d_mag_grad_2d_complex": ("d_mag_grad_2d_complex.cu", "d_mag_grad_2d_complex"),
    "d_mag_grad_3d_complex": ("d_mag_grad_3d_complex.cu", "d_mag_grad_3d_complex"),
}


def _get_spec_key(ndim: int, is_complex: bool) -> str:

    suffix = "complex" if is_complex else "real"
    spec_key = f"d_mag_grad_{ndim}d_{suffix}"
    return spec_key


class FiniteDifferenceMagOp:

    def __init__(self, sz_im, ndim, is_complex, gpu_id, stream):

        if ndim not in (1, 2, 3):
            raise ValueError("FiniteDifferenceMagOp Error: only support dim <= 3")

        self.ndim = int(ndim)
        self.is_complex = bool(is_complex)
        self.gpu_id = gpu_id
        self.stream = stream
        self.dtype = cp.float32  # magnitude is always real-valued

        self.set_size(sz_im)  # sz_im can be changed later

        # --- prepare kernel: assume ndim and is_complex do not change --- #

        spec_key = _get_spec_key(self.ndim, self.is_complex)
        cu_filename, func_name = _KERNEL_SPECS[spec_key]
        self.kernel = compile_rawkernel(
            read_cu_file(cuda_dir, cu_filename), func_name, self.gpu_id
        )

    def set_size(self, sz_im):

        if self.ndim != len(sz_im):
            raise ValueError(
                f"FiniteDifferenceMagOp Error: ndim {self.ndim} does not match length of sz_im {len(sz_im)}."
            )

        self.sz_im = tuple(int(s) for s in sz_im)
        if self.ndim == 1:
            self.nx = sz_im[0]
            self.ny = 1
            self.nz = 1
        elif self.ndim == 2:
            self.ny, self.nx = sz_im
            self.nz = 1
        else:
            self.nz, self.ny, self.nx = sz_im

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

    def run(self, x, out=None):

        xp = cp.get_array_module(x)
        if xp == np:
            d_x = to_device_array(
                x,
                gpu_id=self.gpu_id,
                stream=self.stream,
                dtype_real="float32",
                dtype_complex="complex64",
            )
        else:
            if cp.iscomplexobj(x):
                d_x = x.astype("complex64", copy=False)
            else:
                d_x = x.astype("float32", copy=False)

        with cp.cuda.Device(self.gpu_id), self.stream:
            if out is None:
                d_y = cp.empty(self.sz_im, dtype=self.dtype)
            else:
                d_y = out

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
            self.kernel(self.grd, self.blk, args, stream=self.stream)

        if out is not None:
            return out

        if xp == np:
            h_y = from_device_array(d_y, gpu_id=self.gpu_id, stream=self.stream)
            return h_y

        return d_y
