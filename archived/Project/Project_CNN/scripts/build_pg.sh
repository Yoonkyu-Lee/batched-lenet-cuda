#!/bin/bash

set -euo pipefail

echo "Building the project with gprof instrumentation..."
"$(dirname "$0")/../cleanfile.sh"
cmake -DCMAKE_CXX_FLAGS=-pg "$(dirname "$0")/../project/" && make -j8
"$(dirname "$0")/../cleanfile.sh"
