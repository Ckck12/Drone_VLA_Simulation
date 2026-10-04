#!/usr/bin/env bash
# Phase 0 step 4 — save the full `glxinfo -B` output twice: once with the environment as
# it comes, once with the renderer forced, so the two can be compared later.
# Run from WSL:  bash ~/dronevla/docs/phase0/capture_glxinfo.sh
set -u
out_dir="$(cd "$(dirname "$0")" && pwd)"

capture() {
    label="$1"; file="$2"
    {
        echo "# glxinfo -B --- ${label}"
        echo "# captured      : $(date -Is)"
        echo "# shell         : non-interactive, wsl.exe -d Ubuntu-24.04 -- bash <this script>"
        echo "# mesa-utils    : $(dpkg-query -W -f='${Version}' mesa-utils 2>/dev/null)"
        echo "# libgl1-mesa-dri: $(dpkg-query -W -f='${Version}' libgl1-mesa-dri 2>/dev/null)"
        echo "# DISPLAY                     : ${DISPLAY:-<unset>}"
        echo "# WAYLAND_DISPLAY             : ${WAYLAND_DISPLAY:-<unset>}"
        echo "# GALLIUM_DRIVER              : ${GALLIUM_DRIVER:-<unset>}"
        echo "# LIBGL_ALWAYS_SOFTWARE       : ${LIBGL_ALWAYS_SOFTWARE:-<unset>}"
        echo "# MESA_LOADER_DRIVER_OVERRIDE : ${MESA_LOADER_DRIVER_OVERRIDE:-<unset>}"
        echo
        glxinfo -B 2>&1
        # Capture glxinfo's own status here. Reading $? after the blank echo below would
        # report the echo's status instead, which is always 0 and says nothing.
        rc=$?
        echo
        echo "# --- glxinfo -B exited with status ${rc}"
    } > "${out_dir}/${file}"
    echo "wrote ${file} ($(wc -l < "${out_dir}/${file}") lines, glxinfo rc recorded)"
}

capture "default environment, no renderer override" glxinfo-default.log
GALLIUM_DRIVER=d3d12 capture "GALLIUM_DRIVER=d3d12 forced" glxinfo-d3d12.log
