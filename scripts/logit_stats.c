// logit_stats.c: per-row statistics of one rank's logits shard of a run B against the same shard of a reference run A, for scripts/logit_fidelity.py.
//   logit_stats A.logits.f32 B.logits.f32 SHARD_N OUT
// Both files are [rows][SHARD_N] little-endian f32 (the engine's REXL3_INKLING2_DUMP_LOGITS_ONLY dump). OUT gets, per row, 10 little-endian f64:
//   mA, ZA, SA, iA, a2, mB, ZB, iB, bAt, aAt
// mA/mB the shard maximum of A/B, ZA = sum exp(a - mA), SA = sum exp(a - mA) * (a - b), iA/iB its index in the shard, a2 the shard's second largest A,
// bAt the B logit at A's argmax, aAt the A logit at B's argmax. Rows combine across the four shards in logit_fidelity.py (log-sum-exp, KL(A||B), top-1).
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s A B SHARD_N OUT\n", argv[0]);
        return 2;
    }
    size_t n = strtoull(argv[3], 0, 10);
    FILE *fa = fopen(argv[1], "rb"), *fb = fopen(argv[2], "rb"), *fo = fopen(argv[4], "wb");
    if (!fa || !fb || !fo || n == 0) {
        perror("open");
        return 1;
    }
    float *a = malloc(n * 4), *b = malloc(n * 4);
    if (!a || !b) return 1;
    size_t rows = 0;
    while (fread(a, 4, n, fa) == n) {
        if (fread(b, 4, n, fb) != n) {
            fprintf(stderr, "B is shorter than A at row %zu\n", rows);
            return 1;
        }
        double ma = -INFINITY, a2 = -INFINITY, mb = -INFINITY;
        size_t ia = 0, ib = 0;
        for (size_t i = 0; i < n; i++) {
            if (a[i] > ma) {
                a2 = ma;
                ma = a[i];
                ia = i;
            } else if (a[i] > a2) {
                a2 = a[i];
            }
            if (b[i] > mb) {
                mb = b[i];
                ib = i;
            }
        }
        double za = 0, sa = 0, zb = 0;
        for (size_t i = 0; i < n; i++) {
            if (!isfinite(a[i])) {  // the vocabulary tail the engine masks to -inf: it must be masked in B too, and it carries no probability
                if (isfinite(b[i])) {
                    fprintf(stderr, "row %zu column %zu: masked in A, not in B\n", rows, i);
                    return 1;
                }
                continue;
            }
            double e = exp((double)a[i] - ma);
            za += e;
            sa += e * ((double)a[i] - (double)b[i]);
            zb += exp((double)b[i] - mb);
        }
        double rec[10] = {ma, za, sa, (double)ia, a2, mb, zb, (double)ib, (double)b[ia], (double)a[ib]};
        fwrite(rec, 8, 10, fo);
        rows++;
    }
    if (fread(b, 4, 1, fb) == 1) {
        fprintf(stderr, "B is longer than A\n");
        return 1;
    }
    fclose(fo);
    printf("%zu rows\n", rows);
    return 0;
}
