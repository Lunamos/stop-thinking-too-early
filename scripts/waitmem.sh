#!/bin/bash
# waitmem.sh GPU MIB: block until physical GPU index GPU has at least MIB free memory (checked twice, a minute apart)
G=$1; NEED=$2
while true; do
  f1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G)
  if [ "$f1" -ge "$NEED" ]; then
    sleep 60
    f2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G)
    [ "$f2" -ge "$NEED" ] && break
  fi
  sleep 60
done
