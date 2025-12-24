
extern "C" __global__ void d_mag_grad_1d_real(float *out, float *in,
                                              unsigned int nx) {
    const unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix >= nx)
        return;

    const unsigned int ind = ix;

    float dx = 0.0f;

    if (ix > 0) {
        dx = in[ind] - in[ind - 1];
    }

    out[ind] = dx if (dx > 0) else - dx;
}