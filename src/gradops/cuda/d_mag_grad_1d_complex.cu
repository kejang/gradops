#include <cuComplex.h>
extern "C" __global__ void d_mag_grad_1d_complex(float *out, cuComplex *in,
                                                 unsigned int nx) {
    const unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix >= nx)
        return;

    const unsigned int ind = ix;

    cuComplex dx = make_cuComplex(0.0f, 0.0f);

    if (ix > 0) {
        dx.x = in[ind].x - in[ind - 1].x;
        dx.y = in[ind].y - in[ind - 1].y;
    }

    out[ind] = sqrtf(dx.x * dx.x + dx.y * dx.y);
}