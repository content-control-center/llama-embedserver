#!/bin/sh
# Locate jemalloc and inject it via LD_PRELOAD so it replaces ptmalloc2 for
# all C/C++ allocations (llama.cpp, ggml). Go manages its own heap via mmap
# directly and is unaffected by LD_PRELOAD.
JEMALLOC=$(find /usr/lib -name "libjemalloc.so.*" 2>/dev/null | head -1)
if [ -n "$JEMALLOC" ]; then
    export LD_PRELOAD="$JEMALLOC"
fi
exec embedserver "$@"
