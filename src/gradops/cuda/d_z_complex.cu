#include <cuComplex.h>
extern "C"
__global__
void d_z_complex(
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

    if (iz > 0) {
        out[ind].x = in[ind].x - in[ind - nx*ny].x;
        out[ind].y = in[ind].y - in[ind - nx*ny].y;
    }
    else {
        if (edge) {
            out[ind].x = in[ind].x;
            out[ind].y = in[ind].y;

        }
        else {
            out[ind].x = 0;
            out[ind].y = 0;
        }
    }
}