#!/usr/bin/env bash

# Shell script to run main_uncoupled with mpirun
# at processor counts: 1, then 2..48 in steps of 2.

EXECUTABLE="./build/main_uncoupled"

# Optional: Check the executable exists and is executable
if [ ! -x "$EXECUTABLE" ]; then
  echo "Error: $EXECUTABLE not found or not executable."
  exit 1
fi

echo "==== Running main_uncoupled with 1 process ===="
mpirun -np 1 "$EXECUTABLE"

echo "==== Running main_uncoupled with 2..48 processes (step=2) ===="
for procs in $(seq 2 2 48); do
  echo "Running with $procs processes..."
  mpirun -np "$procs" "$EXECUTABLE"
done
