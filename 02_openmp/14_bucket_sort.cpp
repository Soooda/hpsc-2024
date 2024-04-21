#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
#pragma omp parallel for
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
  for (int i=0; i<n; i++)
    bucket[key[i]]++;

  std::vector<int> offset(range,0);
  int tmp[range];
#pragma omp parallel for
  for (int i = 0; i < range; i++) {
      int temp = offset[0];
      for (int j = 0; j < i; j++) {
        temp += bucket[j];
      }
      tmp[i] = temp;
  }
#pragma omp parallel for
  for (int i=0; i<range; i++) 
    offset[i] = tmp[i];

#pragma omp parallel for shared(key)
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
