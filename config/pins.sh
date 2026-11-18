# Pins for Inkling-2-Small-EXL3-4bpw-4x-DGX-Spark. Sourced by inkling2.sh; not tunable from .env.
MODEL_REPO=0xSero/Inkling-2-Small-EXL3-4bpw
MODEL_REV=996dc404c1feb3abd5330af8a3c52270ffe0008d
MODEL_NAME=Inkling-2-Small-EXL3-4bpw
MANIFEST_FILE=intake/manifest.tsv

# Engine: REXL3 fork branch `inkling2`, started from upstream main (v0.1.2). `build` with no argument builds
# REXL3_COMMIT; `build REV` is a lab build of any commit in the local engine repository.
REXL3_UPSTREAM=https://github.com/REXL3/REXL3.git
REXL3_UPSTREAM_BASE=06181b1d742912108361784257710806d9a75d18
REXL3_FORK=https://github.com/justincasey/REXL3.git
REXL3_COMMIT=b97ef4ee943067e4e9d4a1f15766a5fcfaf4de18   # `inkling2` head: the family host half (no forward yet)
RUST_TOOLCHAIN=1.89.0

# NCCL every rank dlopens (REXL3_NCCL_LIB): the 2.30.7 build REXL3's measured TP runs used, already at this path on all
# four nodes; doctor refuses any other bytes.
NCCL_LIB=/home/spark/rexl3/lib/libnccl.so.2
NCCL_SHA256=fc7ea66334edbc934aa25959b9907dbb2b91a1d2485beff18839afc45cbc08d0
