#!/usr/bin/env bash
# Fetch the third-party assets used by the simulation (sparse, pinned commits).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"

fetch() {  # fetch <repo-url> <dir> <commit> <paths...>
  local url=$1 dir=$2 commit=$3
  shift 3
  if [ ! -d "$TP/$dir" ]; then
    git clone -q --filter=blob:none --sparse "$url" "$TP/$dir"
  fi
  git -C "$TP/$dir" sparse-checkout set "$@"
  git -C "$TP/$dir" checkout -q "$commit"
}

fetch https://github.com/google-deepmind/mujoco_menagerie.git mujoco_menagerie \
  62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35 franka_emika_panda
fetch https://github.com/kevinzakka/mujoco_scanned_objects.git mujoco_scanned_objects \
  6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c models/Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White

echo "Assets ready in $TP"
