#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Architecture Gate Checks ==="

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Check 1: No C/C++/Objective-C/assembly files in repo
echo "Check 1: Searching for forbidden source files (*.c, *.cpp, *.h, *.s, etc.)..."
FORBIDDEN_FILES=$(find . -type f \( \
    -name "*.c" -o -name "*.cc" -o -name "*.cpp" -o -name "*.cxx" \
    -o -name "*.h" -o -name "*.hpp" -o -name "*.hxx" \
    -o -name "*.m" -o -name "*.mm" \
    -o -name "*.s" -o -name "*.S" \
\) ! -path "*/target/*" ! -path "*/.git/*" ! -path "*/fixtures/*" ! -path "*/include/whisper.h" ! -path "*/whisper.cpp/*" || true)

if [ -n "$FORBIDDEN_FILES" ]; then
    echo "ERROR: Found forbidden C/C++/Assembly source files:" >&2
    echo "$FORBIDDEN_FILES" >&2
    exit 1
fi
echo "✓ Check 1 passed: No forbidden source files found."

# Check 2: No build.rs invoking cc, cmake, or bindgen
echo "Check 2: Checking build.rs files for cc/cmake/bindgen invocation..."
BUILD_SCRIPTS=$(find . -name "build.rs" ! -path "*/target/*" ! -path "*/.git/*" ! -path "*/whisper.cpp/*" || true)
if [ -n "$BUILD_SCRIPTS" ]; then
    MATCHES=$(grep -rnE '(cc::|cmake::|bindgen::)' $BUILD_SCRIPTS || true)
    if [ -n "$MATCHES" ]; then
        echo "ERROR: Found forbidden build dependencies in build.rs:" >&2
        echo "$MATCHES" >&2
        exit 1
    fi
fi
echo "✓ Check 2 passed: No C/C++ build tools invoked in build scripts."

# Check 3: Check cargo tree for *-sys, cc, cmake, bindgen
echo "Check 3: Inspecting cargo dependency tree..."
if [ -f "Cargo.lock" ] || [ -f "Cargo.toml" ]; then
    TREE_OUTPUT=$(cargo tree --prefix none || true)
    # Match crates ending in -sys, or C builders (cc, cmake, bindgen, ggml-rs)
    FORBIDDEN_DEPS=$(echo "$TREE_OUTPUT" | grep -E '(\b[a-zA-Z0-9_]+-sys\b|\bcc\s+v|\bcmake\s+v|\bbindgen\s+v|\bggml-rs\s+v)' || true)
    if [ -n "$FORBIDDEN_DEPS" ]; then
        echo "ERROR: Found forbidden *-sys or C-builder crate dependencies in cargo tree:" >&2
        echo "$FORBIDDEN_DEPS" >&2
        exit 1
    fi
fi
echo "✓ Check 3 passed: Dependency tree contains no *-sys or C-builder crates."

# Check 4: Check for extern "C" blocks importing non-std external symbols
echo "Check 4: Checking for extern \"C\" foreign function import blocks..."
RS_FILES=$(find crates src tests examples bindings -name "*.rs" 2>/dev/null ! -path "*/target/*" ! -path "*/whisper.cpp/*" || true)
if [ -n "$RS_FILES" ]; then
    EXTERN_C=$(grep -rnE 'extern\s+"C"\s*\{' $RS_FILES || true)
    if [ -n "$EXTERN_C" ]; then
        echo "ERROR: Found extern \"C\" foreign function import blocks in Rust files:" >&2
        echo "$EXTERN_C" >&2
        exit 1
    fi
fi
echo "✓ Check 4 passed: No extern \"C\" FFI blocks found."

echo "=== All Architecture Gate Checks PASSED! ==="
exit 0
