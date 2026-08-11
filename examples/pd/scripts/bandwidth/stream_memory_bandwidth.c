// Compact STREAM-style benchmark for the Host HiCache and Mooncake DRAM path.
#define _GNU_SOURCE
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static double best_copy(double *restrict a, const double *restrict b, size_t n,
                        int repeats) {
  double best = 1e99;
  for (int r = 0; r < repeats; ++r) {
    double t = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) a[i] = b[i];
    t = omp_get_wtime() - t;
    if (t < best) best = t;
  }
  return best;
}

static double best_triad(double *restrict a, const double *restrict b,
                         const double *restrict c, size_t n, int repeats) {
  double best = 1e99;
  for (int r = 0; r < repeats; ++r) {
    double t = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) a[i] = b[i] + 3.0 * c[i];
    t = omp_get_wtime() - t;
    if (t < best) best = t;
  }
  return best;
}

int main(int argc, char **argv) {
  size_t mib_per_array = argc > 1 ? strtoull(argv[1], 0, 10) : 512;
  int repeats = argc > 2 ? atoi(argv[2]) : 7;
  size_t bytes = mib_per_array << 20;
  size_t n = bytes / sizeof(double);
  double *a = aligned_alloc(4096, bytes);
  double *b = aligned_alloc(4096, bytes);
  double *c = aligned_alloc(4096, bytes);
  if (!a || !b || !c) return 2;
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) { a[i] = 0; b[i] = 1; c[i] = 2; }
  double copy_s = best_copy(a, b, n, repeats);
  double triad_s = best_triad(a, b, c, n, repeats);
  printf("{\"threads\":%d,\"array_mib\":%zu,", omp_get_max_threads(), mib_per_array);
  printf("\"copy_GB_s\":%.6f,\"copy_GiB_s\":%.6f,", 2.0*bytes/copy_s/1e9, 2.0*bytes/copy_s/(1ull<<30));
  printf("\"triad_GB_s\":%.6f,\"triad_GiB_s\":%.6f}\n", 3.0*bytes/triad_s/1e9, 3.0*bytes/triad_s/(1ull<<30));
  free(a); free(b); free(c);
  return 0;
}
