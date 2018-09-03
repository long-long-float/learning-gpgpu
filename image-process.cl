kernel void KERNEL(
                   const global uchar* restrict image,
                   global uchar* result,
                   int N) {
  for (int i = 0; i < N; i++) {
     result[i] = image[i] * 2;
  }
}

