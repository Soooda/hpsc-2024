#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void addBucket(int* key, int *bucket) {
  int i = threadIdx.x;
  atomicAdd(&bucket[key[i]], 1);
}

__global__ void addKey(int *key, int *bucket) {
  int i = threadIdx.x;
  int j = bucket[i];
  for (int offset = 1; offset < 8; offset <<= 1) {
    int temp = __shfl_up_sync(0xffffffff, j, offset);
    if (i >= offset) {
      j += temp;
    }
  }
  j -= bucket[i];

  for (; bucket[i] > 0; bucket[i]--) {
    key[j++] = i;
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket;
  cudaMallocManaged(&key, n * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));
  cudaMemset(bucket, 0, range * sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  addBucket<<<1,n>>>(key, bucket);
  addKey<<<1,range>>>(key, bucket);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
}
