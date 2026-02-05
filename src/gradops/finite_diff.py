from importlib import resources as ir

import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator

from .utils import (
    read_cu_file,
    compile_rawkernel,
    to_device_array,
    from_device_array,
)


module_path = ir.files(__package__)
cuda_dir = module_path / "cuda"


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


def _get_spec_key(is_complex: bool, is_normal: bool, direction: str):

    suffix = "complex" if is_complex else "real"

    if is_normal:
        spec_key_fwd = f"dtd_{direction}_{suffix}"
        spec_key_adj = f"dtd_{direction}_{suffix}"
    else:
        spec_key_fwd = f"d_{direction}_{suffix}"
        spec_key_adj = f"dt_{direction}_{suffix}"

    return spec_key_fwd, spec_key_adj


class _FiniteDifferenceOp:

    def __init__(self, direction, sz_im, dtype, gpu_id, stream, edge):

        self.direction = direction
        self.dtype = cp.dtype(dtype)
        self.edge = edge
        self.gpu_id = gpu_id
        self.stream = stream
        self.set_size(sz_im)

    def set_size(self, sz_im):

        self.sz_im = tuple(int(s) for s in sz_im)
        self.n = np.int64(np.prod(sz_im))
        self.ndim = len(sz_im)

        if self.ndim == 1:
            self.nx = sz_im[0]
            self.ny = 1
            self.nz = 1
        elif self.ndim == 2:
            self.ny, self.nx = sz_im
            self.nz = 1
        elif self.ndim == 3:
            self.nz, self.ny, self.nx = sz_im
        else:
            raise ValueError("sz_im error: only support dim <= 3")

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
            d_y = cp.empty(self.sz_im, dtype=self.dtype)

        args = (
            d_y,
            d_x,
            np.uint32(self.nx),
            np.uint32(self.ny),
            np.uint32(self.nz),
            np.int32(self.edge),
        )

        with cp.cuda.Device(self.gpu_id):
            kernel(self.grd, self.blk, args, stream=self.stream)

        if xp == np:
            y = from_device_array(d_y, gpu_id=self.gpu_id, stream=self.stream)
        else:
            y = d_y

        return y


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

        spec_key_fwd, spec_key_adj = _get_spec_key(is_complex, is_normal, direction)

        if is_normal:
            cu_filename, func_name = _KERNEL_SPECS[spec_key_fwd]
            self.kernel_forward = compile_rawkernel(
                read_cu_file(cuda_dir, cu_filename), func_name, gpu_id
            )
            self.kernel_adjoint = self.kernel_forward
        else:
            cu_filename_fwd, func_name_fwd = _KERNEL_SPECS[spec_key_fwd]
            cu_filename_adj, func_name_adj = _KERNEL_SPECS[spec_key_adj]
            self.kernel_forward = compile_rawkernel(
                read_cu_file(cuda_dir, cu_filename_fwd), func_name_fwd, gpu_id
            )
            self.kernel_adjoint = compile_rawkernel(
                read_cu_file(cuda_dir, cu_filename_adj), func_name_adj, gpu_id
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

    def set_size(self, sz_im):

        self.d_op.set_size(sz_im)
        self.shape = (self.d_op.n, self.d_op.n)

    def _matvec(self, x):
        return self.d_op.run(x, self.kernel_forward).ravel()

    def _rmatvec(self, x):
        return self.d_op.run(x, self.kernel_adjoint).ravel()
