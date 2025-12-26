#!/bin/bash
v=5000
lr=1e-4
logfile="log/filter-focus-init5-twoCrits-${v}-lr=${lr}.log"

for gamma in 0.95 0.99; do
  for tau in 0.001 0.005; do
    for end in 0.05; do
      for decay in 0.0001 0.0005 0.001 0.005; do
        for tf in 10 50 100 500 1000; do
          python3 train.py benchmarks/filter-focus-analysis.maude init twoCrits $v trained/filter-focus-init5-twoCrits-$v \
            sweep $lr $gamma $tau $end $decay $tf >> "$logfile" 2>&1
        done
      done
    done
  done
done