# Release pins. Sourced by the controller; not tuned from .env.
MODEL_REPO="${MODEL_REPO:-0xSero/Inkling-2-Small-EXL3-4bpw}"
MODEL_REV="${MODEL_REV:-996dc404c1feb3abd5330af8a3c52270ffe0008d}"
MODEL_NAME="${MODEL_NAME:-Inkling-2-Small-EXL3-4bpw}"

# TODO: resolve with scripts/recipe_intake.py --image <tag>, then pin the
# manifest-list digest and every runtime fork's full sha.
BASE_IMAGE="${BASE_IMAGE:-TODO}"
BASE_IMAGE_DIGEST="${BASE_IMAGE_DIGEST:-TODO}"
EXL3_REPO="${EXL3_REPO:-TODO}"
EXL3_REV="${EXL3_REV:-TODO}"
PLUGIN_REPO="${PLUGIN_REPO:-TODO}"
PLUGIN_REV="${PLUGIN_REV:-TODO}"
PATCH_FILE="${PATCH_FILE:-patches/0001-TODO.patch}"
OVERLAY_FILE="${OVERLAY_FILE:-}"
MANIFEST_FILE="${MANIFEST_FILE:-intake/manifest.tsv}"
