#include <metal_stdlib>

using u32 = unsigned;
using f32 = float;

void axpy_kernel(u32 i, float alpha, const device f32 X[], device f32 Y[], u32 N) {
  if (i >= N) {
    return;
  }
  Y[i] = alpha * X[i] + Y[i];
}

[[kernel]]
void _mk_axpy(constant float& alpha [[buffer(0)]],
              device float* X [[buffer(1)]],
              device float* Y [[buffer(2)]],
              constant u32& n [[buffer(3)]],
              uint2 id [[thread_position_in_grid]]) {
  axpy_kernel(id.x, alpha, X, Y, n);
}
