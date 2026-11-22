#!/bin/bash
# ab_screen.sh PHASES 'label|KEY=VAL KEY=VAL' ...   one autoresearch screen per variant, in the order given (repeat the list to interleave).
# EXTRA_ENV of .env.example is set per variant and restored from git afterwards. Run it under `safeloop.sh cmd` so the known-good engine
# comes back. The engine under test is the HEAD of CANDIDATE_SRC (exported by the caller).
cd "$(dirname "$0")/.." || exit 2
PH="$1"; shift
mkdir -p .work/ab
n=0
for v in "$@"; do
  n=$((n + 1))
  label=${v%%|*}; envs=${v#*|}; [[ $v == *'|'* ]] || envs=""
  git checkout -- .env.example
  python3 - "$envs" <<'PY'
import re, sys
p = '.env.example'
s = open(p).read()
s = re.sub(r'^EXTRA_ENV=.*$', 'EXTRA_ENV=%s' % sys.argv[1], s, count=1, flags=re.M)
open(p, 'w').write(s)
PY
  log=.work/ab/$(date +%H%M%S)-$n-$label.log
  echo "== [$n] $label  EXTRA_ENV='$envs'  PHASES=$PH  ($(date +%H:%M))"
  PHASES=$PH bash autoresearch.sh > "$log" 2>&1
  rc=$?
  run=$(grep -o -m1 'run=bench/runs/autoresearch-[0-9-]*' "$log" | head -1 | cut -d= -f2)
  echo "   rc=$rc run=$run"
  grep -E '^(METRIC (decode_(c1|prose|structured|json|code)_tok_s|decode_tokens_per_round|prefill_32k_tok_s|boot_s)|FAIL|PASS pin)' "$log" | sed 's/^/   /'
  git checkout -- .env.example
done
