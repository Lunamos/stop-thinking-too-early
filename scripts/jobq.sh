#!/bin/bash
# file-driven job queue: jobq.sh GPU QUEUEFILE ; executes the lines of QUEUEFILE one by one from this directory, on GPU, and appends
# each finished line to QUEUEFILE.done (so the file can be extended while the queue runs). PY (the Python interpreter) and HF_HOME
# come from the environment or from local_env.sh next to this script (not tracked), e.g.
#   PY=/path/to/venv/bin/python; export HF_HOME=/path/to/hf_cache
GPU=$1; Q=$2; DONE=$Q.done
cd "$(dirname "$0")"
[ -f local_env.sh ] && source local_env.sh
: "${PY:?set PY (the Python interpreter) in the environment or in local_env.sh}"
export HF_HOME
touch $DONE
while true; do
  line=$(grep -v -x -F -f $DONE $Q | grep -v '^#' | grep -v '^\s*$' | head -1)
  if [ -z "$line" ]; then sleep 30; continue; fi
  echo "$(date) START $line"
  CUDA_VISIBLE_DEVICES=$GPU PY=$PY bash -c "$line"
  echo "$line" >> $DONE
  echo "$(date) END $line"
done
