#include <cuComplex.h>
extern "C" __global__ void d_mag_grad_3d_complex(float *out, cuComplex *in,
                                                 unsigned int nx,
                                                 unsigned int ny,
                                                 unsigned int nz) {
    const unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= nx || iy >= ny || iz >= nz)
        return;

    const size_t ind = (size_t)ix + (size_t)iy * nx + (size_t)iz * nx * ny;

    cuComplex dx = make_cuComplex(0.0f, 0.0f);
    cuComplex dy = make_cuComplex(0.0f, 0.0f);
    cuComplex dz = make_cuComplex(0.0f, 0.0f);

    if (ix > 0) {
        dx.x = in[ind].x - in[ind - 1].x;
        dx.y = in[ind].y - in[ind - 1].y;
    }
    if (iy > 0) {
        dy.x = in[ind].x - in[ind - nx].x;
        dy.y = in[ind].y - in[ind - nx].y;
    }
    if (iz > 0) {
        dz.x = in[ind].x - in[ind - (size_t)nx * ny].x;
        dz.y = in[ind].y - in[ind - (size_t)nx * ny].y;
    }

    out[ind] = sqrtf(dx.x * dx.x + dx.y * dx.y + dy.x * dy.x + dy.y * dy.y +
                     dz.x * dz.x + dz.y * dz.y);
}