
extern "C"
__global__
void dt_z_real(
    float *out, float *in,
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

    if (iz == 0) {
        if (edge) {
            out[ind] = in[ind] - in[ind + nx*ny];
        }
        else {
            out[ind] = -in[ind + nx*ny];
        }
    }
    else if (iz == (nz - 1)) {
        out[ind] = in[ind];
    }
    else {
        out[ind] = in[ind] - in[ind + nx*ny];
    }
}
