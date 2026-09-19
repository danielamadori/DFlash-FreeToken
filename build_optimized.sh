#!/usr/bin/env bash
# ==============================================================================
# Script di Build Ottimizzata -O3 / -Ofast per Linux / macOS / Apple Silicon
# ==============================================================================
set -e

mkdir -p build bin

if [[ "$(uname)" == "Darwin" ]]; then
    # macOS / Apple Silicon Metal
    CFLAGS="-O3 -Ofast -flto -DNDEBUG -Wall -Wextra"
else
    # Linux x86_64 con AVX2 / FMA
    CFLAGS="-O3 -Ofast -march=native -mtune=native -mavx2 -mfma -ffast-math -flto -DNDEBUG -Wall -Wextra"
fi

echo "Compilazione FreeToken C Engine con flag: $CFLAGS"
clang $CFLAGS -c src/freetoken_engine.c -Iinclude -o build/freetoken_engine.o
clang $CFLAGS build/freetoken_engine.o bench/bench_freetoken_io.c -Iinclude -lm -o bin/freetoken-engine

echo "✔ Build completata con successo: bin/freetoken-engine"
