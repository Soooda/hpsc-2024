#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 xj = _mm512_load_ps(x);
    __m512 yi = _mm512_set1_ps(y[i]);
    __m512 yj = _mm512_load_ps(y);
    __m512 mvec = _mm512_load_ps(m);

    __m512 rx = _mm512_sub_ps(xi, xj);
    __m512 ry = _mm512_sub_ps(yi, yj);

    __m512 r = _mm512_sqrt_ps(_mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry)));
    __m512 r2 = _mm512_mul_ps(r, r);
    __m512 r3 = _mm512_mul_ps(r2, r);

    __m512 dfx = _mm512_div_ps(_mm512_mul_ps(rx, mvec), r3);
    __m512 dfy = _mm512_div_ps(_mm512_mul_ps(ry, mvec), r3);

    __m512 zeros = _mm512_setzero_ps();
    __mmask16 mask = _mm512_cmp_ps_mask(r, zeros, _MM_CMPINT_GT);
    dfx = _mm512_mask_blend_ps(mask, zeros, dfx);
    dfy = _mm512_mask_blend_ps(mask, zeros, dfy);

    __m512 inv = _mm512_set1_ps(-1);
    dfx = _mm512_mul_ps(dfx, inv);
    dfy = _mm512_mul_ps(dfy, inv);
    fx[i] = _mm512_reduce_add_ps(dfx);
    fy[i] = _mm512_reduce_add_ps(dfy);

    // for(int j=0; j<N; j++) {
    //   if(i != j) {
    //     float rx = x[i] - x[j];
    //     float ry = y[i] - y[j];
    //     float r = std::sqrt(rx * rx + ry * ry);
    //     fx[i] -= rx * m[j] / (r * r * r);
    //     fy[i] -= ry * m[j] / (r * r * r);
    //   }
    // }
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
