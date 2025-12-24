#include <cuComplex.h>
extern "C"
__global__
void dtd_x_complex(
    cuComplex *out, cuComplex *in,
    const unsigned int nx, const unsigned int ny, const unsigned int nz,
    const int edge
)
{
    const unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if ((ix >= nx) || (iy >= ny) || (iz >= nz)) {
    	return;
    }

    const unsigned long long ind = ix + iy*nx + iz*nx*ny;

    if (ix == 0) {
        if (edge) {
            out[ind].x = 2.0*in[ind].x - in[ind + 1].x;
            out[ind].y = 2.0*in[ind].y - in[ind + 1].y;
        }
        else {
            out[ind].x = in[ind].x - in[ind + 1].x;
            out[ind].y = in[ind].y - in[ind + 1].y;
        }
    }
    else if (ix == (nx - 1)) {
        out[ind].x = in[ind].x - in[ind - 1].x;
        out[ind].y = in[ind].y - in[ind - 1].y;
    }
    else {
        out[ind].x = 2.0*in[ind].x - in[ind + 1].x - in[ind - 1].x;
        out[ind].y = 2.0*in[ind].y - in[ind + 1].y - in[ind - 1].y;

    }
}