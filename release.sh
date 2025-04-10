#!/bin/bash
set -euo pipefail; IFS=$'\n\t'

NAME="xnlbd"
VER=$( python -c "from xnlbd._version import __version__; print(__version__)" )

echo "========================================================================"
echo "Tagging $NAME v$VER"
echo "========================================================================"

git tag -a v$VER
git push origin v$VER

echo "========================================================================"
echo "Releasing $NAME v$VER on PyPI"
echo "========================================================================"

python -m build --sdist
twine upload dist/*
rm -r dist/ *.egg-info

