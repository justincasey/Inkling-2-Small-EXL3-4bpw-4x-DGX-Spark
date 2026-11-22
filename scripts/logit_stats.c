// logit_stats.c: per-row statistics of one rank's logits shard of a run B against the same shard of a reference run A, for scripts/logit_fidelity.py.
//   logit_stats A.logits.f32 B.logits.f32 SHARD_N OUT LABELS.i32 VOCAB_BASE
// Both logit files are [rows][SHARD_N] little-endian f32 (the engine's REXL3_INKLING2_DUMP_LOGITS_ONLY dump). LABELS.i32 holds, per row, the global vocabulary id of
// the token the prompt has next (-1: none, the last row); VOCAB_BASE is the global id of the shard's column 0. OUT gets, per row, 12 little-endian f64:
//   mA, ZA, SA, iA, a2, mB, ZB, iB, bAt, aAt, aLab, bLab
// mA/mB the shard maximum of A/B, ZA = sum exp(a - mA), SA = sum exp(a - mA) * (a - b), iA/iB its index in the shard, a2 the shard's second largest A,
// bAt the B logit at A's argmax, aAt the A logit at B's argmax, aLab / bLab the logits at the row's label when the label is in this shard (else NaN).
// Rows combine across the four shards in logit_fidelity.py (log-sum-exp, KL(A||B), top-1, and the negative log-likelihood of the label in A and in B).
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s A B SHARD_N OUT LABELS.i32 VOCAB_BASE\n", argv[0]);
        return 2;
    }
    size_t n = strtoull(argv[3], 0, 10);
    long long base = strtoll(argv[6], 0, 10);
    FILE *fa = fopen(argv[1], "rb"), *fb = fopen(argv[2], "rb"), *fo = fopen(argv[4], "wb"), *fl = fopen(argv[5], "rb");
    if (!fa || !fb || !fo || !fl || n == 0) {
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
        int32_t label;
        if (fread(&label, 4, 1, fl) != 1) {
            fprintf(stderr, "the labels end before row %zu\n", rows);
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
        const long long local = (long long)label - base;
        const int mine = label >= 0 && local >= 0 && (size_t)local < n;
        double rec[12] = {ma, za, sa, (double)ia, a2, mb, zb, (double)ib, (double)b[ia], (double)a[ib], mine ? (double)a[local] : NAN, mine ? (double)b[local] : NAN};
        fwrite(rec, 8, 12, fo);
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
