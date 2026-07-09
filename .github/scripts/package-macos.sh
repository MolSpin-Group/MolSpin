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

seen_deps_file="$(mktemp)"
trap 'rm -f "$seen_deps_file"' EXIT

should_bundle_dep() {
  local dep_name
  dep_name="$(basename "$1")"
  case "$dep_name" in
    libarmadillo*|libopenblas*|liblapack*|libarpack*|libomp*|libgomp*|libgfortran*|libiomp*|libquadmath*|libgcc*|libstdc++*|libmpi*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_dep_path() {
  local dep="$1"
  local dep_name
  dep_name="$(basename "$dep")"

  if [[ "$dep" == /usr/lib/* || "$dep" == /System/Library/* ]]; then
    return 1
  fi

  if [[ "$dep" == @rpath/* ]]; then
    if [[ -f "package/lib/$dep_name" ]]; then
      printf '%s\n' "package/lib/$dep_name"
      return 0
    fi
    find /opt/homebrew/opt /usr/local/opt -type f -name "$dep_name" 2>/dev/null | head -n 1
    return 0
  fi

  printf '%s\n' "$dep"
}

bundle_deps() {
  local target="$1"
  local dep raw_dep dep_path bundled_dep_path
  while IFS= read -r raw_dep; do
    [[ -z "$raw_dep" ]] && continue
    dep_path="$(resolve_dep_path "$raw_dep" || true)"
    [[ -n "$dep_path" && -f "$dep_path" ]] || continue
    if grep -Fqx "$dep_path" "$seen_deps_file"; then
      continue
    fi
    printf '%s\n' "$dep_path" >> "$seen_deps_file"
    should_bundle_dep "$dep_path" || continue
    bundled_dep_path="package/lib/$(basename "$dep_path")"
    if [[ ! -f "$bundled_dep_path" ]]; then
      cp "$dep_path" "$bundled_dep_path"
    fi
    bundle_deps "$dep_path"
  done < <(otool -L "$target" | sed '1d' | awk '{print $1}')
}

has_loader_rpath() {
  local binary="$1"
  otool -l "$binary" | grep -Fq 'path @loader_path '
}

bundle_deps package/molspin

for dep in package/lib/*; do
  [[ -f "$dep" ]] || continue
  chmod u+w "$dep"
  install_name_tool -id "@rpath/$(basename "$dep")" "$dep"
  if ! has_loader_rpath "$dep"; then
    install_name_tool -add_rpath "@loader_path" "$dep"
  fi
done

chmod u+w package/molspin
if ! otool -l package/molspin | grep -Fq 'path @executable_path/lib '; then
  install_name_tool -add_rpath "@executable_path/lib" package/molspin
fi

rewrite_deps() {
  local binary="$1"
  local dep dep_name
  while IFS= read -r dep; do
    [[ -z "$dep" ]] && continue
    [[ "$dep" == /usr/lib/* || "$dep" == /System/Library/* ]] && continue
    dep_name="$(basename "$dep")"
    if [[ -f "package/lib/$dep_name" ]]; then
      install_name_tool -change "$dep" "@rpath/$dep_name" "$binary"
    fi
  done < <(otool -L "$binary" | sed '1d' | awk '{print $1}')
}

rewrite_deps package/molspin

for dep in package/lib/*; do
  [[ -f "$dep" ]] || continue
  rewrite_deps "$dep"
done

if [[ -f "$(brew --prefix armadillo)/LICENSE.txt" ]]; then
  cp "$(brew --prefix armadillo)/LICENSE.txt" package/LICENSES/armadillo-license.txt
fi
if [[ -f "$(brew --prefix armadillo)/NOTICE.txt" ]]; then
  cp "$(brew --prefix armadillo)/NOTICE.txt" package/LICENSES/armadillo-notice.txt
fi
if [[ -f "$(brew --prefix openblas)/LICENSE" ]]; then
  cp "$(brew --prefix openblas)/LICENSE" package/LICENSES/openblas-license.txt
elif [[ -f "$(brew --prefix openblas)/LICENSE.txt" ]]; then
  cp "$(brew --prefix openblas)/LICENSE.txt" package/LICENSES/openblas-license.txt
fi
if [[ -f "$(brew --prefix libomp)/LICENSE.TXT" ]]; then
  cp "$(brew --prefix libomp)/LICENSE.TXT" package/LICENSES/libomp-license.txt
fi

tar -czf "$artifact_name" -C package .
