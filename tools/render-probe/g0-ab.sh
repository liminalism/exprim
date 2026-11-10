#!/bin/bash
# ab.sh <A.exe> <B.exe> <out-prefix> <rounds> <id> <pdf> [bench args...]
A=$1; B=$2; OUT=$3; R=$4; ID=$5; PDF=$6; shift 6
export RAYON_NUM_THREADS=1 PDF_RENDERER_IMAGE_ROW_PAR=0 PDF_RENDERER_JPX_CONCURRENCY=serial
for i in $(seq 1 $R); do
  if [ $((i%2)) = 1 ]; then o="A B"; else o="B A"; fi
  for x in $o; do
    if [ $x = A ]; then e=$A; else e=$B; fi
    "$e" bench "$PDF" --id "$ID#$i" --hash "$@" --jsonl "$OUT-$x.jsonl"
  done
done
