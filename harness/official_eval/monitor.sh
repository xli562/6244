#!/usr/bin/env bash
# Poll the official-eval run every INTERVAL seconds until it finishes or dies.
# Usage: bash harness/official_eval/monitor.sh [interval_seconds]
cd /temp/xli56/6244 || exit 1
OUT=runs/official_eval_01
LOG=$OUT/test_50.log
TMP=$OUT/test_50.tmp
FINAL=$OUT/test_50.jsonl.gz
INTERVAL=${1:-300}

while true; do
  stamp=$(date '+%H:%M:%S')
  if [ -f "$FINAL" ]; then
    echo "[$stamp] DONE: $FINAL exists. Next: python harness/official_eval/plot.py --src $FINAL"
    exit 0
  fi
  if ! pgrep -f "official_eval/run.py test" >/dev/null; then
    echo "[$stamp] STOPPED: process gone without a final file. Check $LOG"
    exit 1
  fi
  procs=$(pgrep -fc "official_eval/run.py test")
  size=$(du -h "$TMP" 2>/dev/null | cut -f1)
  last=$(tail -n 1 "$LOG" 2>/dev/null)
  echo "[$stamp] RUNNING: $procs procs, partial output ${size:-none}, last: ${last:-no progress yet}"
  sleep "$INTERVAL"
done
