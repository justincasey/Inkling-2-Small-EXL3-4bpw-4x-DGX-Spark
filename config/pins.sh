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
# Segment B1 (2026-11-22): the dense core in MXINT8, branch `inkling2-b` (docs/exec-plans/active/autoresearch-segment-b.md). Segment A's pin, whose replies the old
# references held, was f4e27caec70da060df2a923097fb750a0880a653 (recipe tag segment-a-v1, engine tag inkling2-segment-a-v1 at a73ff7a, same replies).
REXL3_COMMIT=c4a3a2675dc82db76676dc2b962eb524fbe73e95   # `inkling2-b`: `inkling2` + Segment A + int8 dense core (REXL3_INKLING2_INT8=0 serves bf16)
RUST_TOOLCHAIN=1.89.0

# NCCL every rank dlopens (REXL3_NCCL_LIB): the 2.30.7 build REXL3's measured TP runs used, already at this path on all
# four nodes; doctor refuses any other bytes.
NCCL_LIB=/home/spark/rexl3/lib/libnccl.so.2
NCCL_SHA256=fc7ea66334edbc934aa25959b9907dbb2b91a1d2485beff18839afc45cbc08d0

# NCCL environment of every rank: one array, so every rank gets the same NCCL_* set (REXL3's TP boot gate compares every
# NCCL_* variable across ranks and refuses a mismatch). Values from the switched fabric (skill dgx-spark-cluster):
# - rail 0 netdev for NCCL's sockets, both rails' RoCE devices (exact match, port 1), RoCEv2 IPv4 GID index 3, and traffic
#   class 106 (DSCP 26: the switch's lossless PFC class; unmarked RoCE is best effort and drops under incast). REXL3's own
#   RoCE transport reads NCCL_IB_HCA and NCCL_IB_TC too. The device list and TC are REXL3's measured TP runs on this switch
#   (RED-SNOW-5.3-Flash-2x-DGX-Spark rexl3/cluster.sh) and DeepSeek-V4.1-Flash-1017's FABRIC=switched (`dsv41.sh`
#   fabric_env: both rails to NCCL, NCCL_IB_TC=ROCE_TC=106).
# - NCCL_MAX_NCHANNELS=4: REXL3's lockstep default, made explicit.
# - Transport selection shared by every fabric in dsv41.sh / spark-ds41f (sparknet COMMON_IB_ENV): IB/RoCE net, IPv4
#   GIDs, RoCE v2, no net plugin (a plugin present on one node only would change the transport on that rank alone).
# Not taken over: sparknet's vLLM tuning (NCCL_PROTO, buffer sizes, CUMEM, channel minimums): unmeasured with REXL3, and
# NCCL_PROTO is part of REXL3's NCCL fingerprint. /etc/nccl.conf on .123/.18 sets NCCL_SOCKET_IFNAME and NCCL_IB_HCA (rail
# 0 only); the environment overrides it, and doctor refuses a conf key this array does not override.
NCCL_ENV=(
  NCCL_SOCKET_IFNAME=enp1s0f1np1
  NCCL_IB_HCA==rocep1s0f1:1,roceP2p1s0f1:1
  NCCL_IB_GID_INDEX=3
  NCCL_IB_TC=106
  NCCL_MAX_NCHANNELS=4
  NCCL_NET=IB
  NCCL_IB_DISABLE=0
  NCCL_IB_ADDR_FAMILY=AF_INET
  NCCL_IB_ROCE_VERSION_NUM=2
  NCCL_NET_PLUGIN=none
  NCCL_DEBUG=WARN
)
