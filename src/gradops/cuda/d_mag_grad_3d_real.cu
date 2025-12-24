
extern "C" __global__ void d_mag_grad_3d_real(float *out, float *in,
                                              unsigned int nx,
                                              unsigned int ny,
                                              unsigned int nz) {
  const unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  const unsigned int iz = threadIdx.z + blockIdx.z * blockDim.z;

  if (ix >= nx || iy >= ny || iz >= nz)
    return;

  const size_t ind = (size_t)ix + (size_t)iy * nx + (size_t)iz * nx * ny;

  float dx = 0.0f, dy = 0.0f, dz = 0.0f;

  if (ix > 0) {
    dx = in[ind] - in[ind - 1];
  }
  if (iy > 0) {
    dy = in[ind] - in[ind - nx];
  }
  if (iz > 0) {
    dz = in[ind] - in[ind - (size_t)nx * ny];
  }

  out[ind] = sqrtf(dx * dx + dy * dy + dz * dz);
}