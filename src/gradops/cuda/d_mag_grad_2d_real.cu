
extern "C" __global__ void
d_mag_grad_2d_real(float *out, float *in, unsigned int nx, unsigned int ny) {
    const unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= nx || iy >= ny)
        return;

    const size_t ind = (size_t)ix + (size_t)iy * nx;

    float dx = 0.0f, dy = 0.0f;

    if (ix > 0) {
        dx = in[ind] - in[ind - 1];
    }
    if (iy > 0) {
        dy = in[ind] - in[ind - nx];
    }

    out[ind] = sqrtf(dx * dx + dy * dy);
}