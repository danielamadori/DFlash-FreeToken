@echo off
REM ==============================================================================
REM Script di Build Ottimizzata con Flag Estremi (-O3 / -Ofast / Native Arch)
REM per l'Engine Ibrido C (FreeToken + mmap NVMe di antirez)
REM ==============================================================================

set CC=clang
set CFLAGS=-O3 -Ofast -march=native -mtune=native -mavx2 -mfma -ffast-math -flto -DNDEBUG -Wall -Wextra

echo [1/2] Compilazione di freetoken_engine.c con ottimizzazione massima...
clang %CFLAGS% -c src/freetoken_engine.c -Iinclude -o build/freetoken_engine.o

echo [2/2] Linking statico e generazione del binario standalone...
clang %CFLAGS% build/freetoken_engine.o bench/bench_freetoken_io.c -Iinclude -o bin/freetoken-engine.exe

echo.
echo ==============================================================================
echo  ✔ Build -O3 Completata con successo in bin/freetoken-engine.exe
echo ==============================================================================
