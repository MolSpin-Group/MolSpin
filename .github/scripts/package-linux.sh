#!/usr/bin/env bash
set -euo pipefail

artifact_name="${1:?artifact name is required}"

rm -rf package
mkdir -p package/lib package/LICENSES

exe="$(find build -type f -name molspin | head -n 1)"
if [[ -z "$exe" || ! -f "$exe" ]]; then
  echo "molspin executable not found" >&2
  exit 1
fi

cp "$exe" package/molspin
cp README.md package/README.md

declare -A seen_libs=()

should_bundle_dep() {
  local dep_name
  dep_name="$(basename "$1")"
  case "$dep_name" in
    libarmadillo*|libopenblas*|liblapack*|libarpack*|libgomp*|libgfortran*|libquadmath*|libiomp*|libomp*|libgcc_s*|libstdc++*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

bundle_deps() {
  local target="$1"
  local dep
  while IFS= read -r dep; do
    [[ -z "$dep" || "$dep" != /* ]] && continue
    [[ "${seen_libs[$dep]:-0}" == 1 ]] && continue
    seen_libs[$dep]=1
    should_bundle_dep "$dep" || continue
    [[ -f "$dep" ]] || continue
    cp "$dep" "package/lib/$(basename "$dep")"
    bundle_deps "$dep"
  done < <(ldd "$target" | awk '/=>/ {print $3}')
}

bundle_deps package/molspin

patchelf --set-rpath '$ORIGIN/lib' package/molspin

while IFS= read -r bundled_lib; do
  [[ -f "$bundled_lib" ]] || continue
  patchelf --set-rpath '$ORIGIN' "$bundled_lib"
done < <(find package/lib -type f -name 'lib*.so*' | sort)

for license in \
  /usr/share/doc/libarmadillo-dev/copyright:libarmadillo-licence.txt \
  /usr/share/doc/libopenblas-dev/copyright:libopenblas-licence.txt \
  /usr/share/doc/liblapack-dev/copyright:liblapack-licence.txt \
  /usr/share/doc/libgomp1/copyright:libgomp-licence.txt
do
  src="${license%%:*}"
  dest="${license##*:}"
  [[ -f "$src" ]] && cp "$src" "package/LICENSES/$dest"
done

tar -czf "$artifact_name" -C package .
