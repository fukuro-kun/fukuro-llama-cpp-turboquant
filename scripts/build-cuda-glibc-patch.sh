#!/bin/bash
# Build-Skript fuer llama.cpp mit CUDA auf Systemen mit glibc >= 2.43
# Problem: glibc 2.43 fuegt 'noexcept' zu math-Funktionen hinzu,
#          was CUDA 13.1 nicht verarbeiten kann (Konflikt bei rsqrt).
# Loesung: Temporaeres Entfernen von rsqrt aus mathcalls.h waehrend des Builds.
#
# Betroffene Kombination: CUDA 13.1 + glibc 2.43 (z.B. Ubuntu 26.04)
# Die System-Datei wird nach dem Build wiederhergestellt.
#
# Verwendung:
#   chmod +x scripts/build-cuda-glibc-patch.sh
#   ./scripts/build-cuda-glibc-patch.sh
#
# Alternativen (statt dieses Patches):
#   1. Docker/Container mit aelterer glibc bauen
#   2. Lokale Kopie der Header fuer NVCC verwenden
#   3. Auf CUDA 13.2+ warten (wird den Konflikt beheben)

set -e

LLAMA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MATHCALLS="/usr/include/x86_64-linux-gnu/bits/mathcalls.h"
BUILD_DIR="${LLAMA_DIR}/build"

# Pruefe ob Patch noetig ist
if grep -q "noexcept" "${MATHCALLS}" 2>/dev/null; then
    echo "[INFO] Patche mathcalls.h fuer CUDA 13.1 Kompatibilitaet..."
    sudo cp "${MATHCALLS}" "${MATHCALLS}.backup.build"
    sudo sed -i '/__MATHCALL_VEC (rsqrt/d' "${MATHCALLS}"
    PATCHED=1
else
    echo "[INFO] mathcalls.h benoetigt keinen Patch."
    PATCHED=0
fi

# Cleanup und Build
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
cmake -S "${LLAMA_DIR}" -B "${BUILD_DIR}" \
  -DLLAMA_CUDA=ON -DLLAMA_NATIVE=ON

cmake --build "${BUILD_DIR}" -j$(nproc)

# Wiederherstellung
if [ "${PATCHED}" -eq 1 ]; then
    echo "[INFO] Stelle mathcalls.h wieder her..."
    sudo cp "${MATHCALLS}.backup.build" "${MATHCALLS}"
fi

echo "[OK] Build abgeschlossen: ${BUILD_DIR}/bin/llama-server"
