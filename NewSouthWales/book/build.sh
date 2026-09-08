#!/usr/bin/env bash
# Build the NewSouthWales Jupyter Book for the GUI's "Report" tab.
#
# Wraps `jupyter-book build --html` with a required post-build fix: this
# version of jupyter-book/mystmd fetches the site theme's runtime JS/CSS
# bundle into `_build/templates/.../public/build/` but never copies it into
# the exported `_build/html/` — every page then references
# `/myst_assets_folder/...` paths that simply don't exist in the static
# export, so the report loads with no styling and dead navigation. Until
# that's fixed upstream, copy the bundle over ourselves.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BASE_URL=/case-book/newsouthwales jupyter-book build --html

theme_build=$(find _build/templates/site -maxdepth 4 -type d -path "*/public/build" | head -1)
if [ -z "$theme_build" ]; then
    echo "error: couldn't find the site theme's public/build/ under _build/templates/ — jupyter-book's output layout may have changed, check this script" >&2
    exit 1
fi
mkdir -p _build/html/myst_assets_folder
cp -r "$theme_build"/. _build/html/myst_assets_folder/

echo "Built. Report is under _build/html/ — refresh the GUI to see changes."
