#!/usr/bin/env bash
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
checkout="$root/third_party/LIBERO"
envdir=${LIBERO_VENV:-"$root/third_party/libero-env"}
revision=8f1084e3132a39270c3a13ebe37270a43ece2a01
mkdir -p "$root/third_party"
if [ ! -e "$checkout" ]; then
  git init -q "$checkout"
  git -C "$checkout" remote add origin https://github.com/Lifelong-Robot-Learning/LIBERO.git
  git -C "$checkout" fetch --depth 1 origin "$revision"
  git -C "$checkout" checkout -q --detach FETCH_HEAD
fi
test "$(git -C "$checkout" rev-parse HEAD)" = "$revision" || { echo 'LIBERO revision mismatch' >&2; exit 1; }
if [ ! -x "$envdir/bin/python" ]; then
  "${PYTHON:-python3.11}" -m venv "$envdir"
fi
"$envdir/bin/python" -m pip install 'numpy==1.26.4' 'mujoco==3.3.0' 'robosuite==1.4.1' 'bddl==1.0.1' 'gym==0.26.2' easydict h5py imageio pyyaml matplotlib future
"$envdir/bin/python" "$root/sim/libero_long.py" prepare
