#include <cuComplex.h>
extern "C" __global__ void d_mag_grad_2d_complex(float *out, cuComplex *in,
                                                 unsigned int nx,
                                                 unsigned int ny) {
    const unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= nx || iy >= ny)
        return;

    const size_t ind = (size_t)ix + (size_t)iy * nx;

    cuComplex dx = make_cuComplex(0.0f, 0.0f);
    cuComplex dy = make_cuComplex(0.0f, 0.0f);

    if (ix > 0) {
        dx.x = in[ind].x - in[ind - 1].x;
        dx.y = in[ind].y - in[ind - 1].y;
    }
    if (iy > 0) {
        dy.x = in[ind].x - in[ind - nx].x;
        dy.y = in[ind].y - in[ind - nx].y;
    }

    out[ind] = sqrtf(dx.x * dx.x + dx.y * dx.y + dy.x * dy.x + dy.y * dy.y);
}