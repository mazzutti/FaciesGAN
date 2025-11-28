#!/usr/bin/env bash
set -euo pipefail

# setup-subrepo.sh
# Helper script to install git-subrepo (optional) and import/update
# specific subdirectories from a remote repository into `data/`.
#
# This script uses a safe, built-in `git subtree` method by default to
# import remote paths `.cache/images/dataset` -> `data/dataset` and
# `.cache/images/xz_slices` -> `data/xz_slices`. If `git subrepo` is
# installed and you prefer it, the script can detect it but the import
# step uses the subtree approach for portability.

REPO_URL="https://github.com/mazzutti/Stanford-VI-E.git"
REPO_BRANCH="master"

usage(){
  cat <<EOF
Usage: $0 <command>

Commands:
  install     Install git-subrepo (Homebrew on macOS if available)
  import      Import remote paths into your repo (creates/updates data/*)
  update      Update previously imported subtrees from upstream
  help        Show this message

Import maps remote paths to local prefixes:
  .cache/images/dataset    -> data/dataset
  .cache/images/xz_slices  -> data/xz_slices

Notes:
  - The import uses a temporary clone and `git subtree split` then
    `git subtree add --prefix=...` so the imported content is part of
    your superproject (history can be squashed).
  - Backups of any existing targets are stored under `.backup_data`.
  - This script does NOT automatically remove any existing submodule
    entries; ensure you have a clean state before running.
EOF
}

install_gitsubrepo(){
  if command -v git-subrepo >/dev/null 2>&1 || command -v git-subrepo >/dev/null 2>&1; then
    echo "git-subrepo already installed"
    return 0
  fi

  if command -v brew >/dev/null 2>&1; then
    echo "Installing git-subrepo via Homebrew..."
    brew install git-subrepo || {
      echo "Homebrew install failed. Please install git-subrepo manually: https://github.com/ingydotnet/git-subrepo"
      return 1
    }
    echo "git-subrepo installed"
  else
    echo "Homebrew not found. Please install git-subrepo manually: https://github.com/ingydotnet/git-subrepo"
    return 1
  fi
}

# Internal helper: import one remote subdir into superproject via subtree
# Arguments: <remote-subdir> <local-prefix>
import_subdir_via_subtree(){
  remote_subdir="$1"
  prefix="$2"

  tmpdir=$(mktemp -d /tmp/stanford-XXXX)
  echo "Cloning remote repository into temporary dir: $tmpdir"

  # Clone without checkout (faster) then sparse-checkout the desired subdir
  git clone --no-checkout "$REPO_URL" "$tmpdir"
  pushd "$tmpdir" >/dev/null
  git sparse-checkout init --no-cone
  git sparse-checkout set "$remote_subdir"
  git checkout "${REPO_BRANCH}" || true

  split_branch="tmp-split-$(echo "$remote_subdir" | tr '/.' '__')"
  echo "Creating split branch $split_branch for path $remote_subdir"
  git subtree split -P "$remote_subdir" -b "$split_branch" || true
  popd >/dev/null

  # Create a bundle from the temporary clone for the split branch and
  # import via the bundle to avoid using a local remote path. This is
  # more robust on systems where git may attempt to chdir into the
  # temporary path (which can race if the path is removed).
  bundle=$(mktemp /tmp/subtree-XXXX.bundle)
  git -C "$tmpdir" bundle create "$bundle" "$split_branch" || true

  echo "Adding subtree at prefix $prefix"
  # Back up any existing target
  mkdir -p .backup_data
  if [ -e "$prefix" ]; then
    echo "Backing up existing $prefix -> .backup_data/$(basename "$prefix").bak"
    mv "$prefix" ".backup_data/$(basename "$prefix").bak"
  fi

  # Use --squash to keep history compact. Remove --squash if you want full history.
  git subtree add --prefix="$prefix" "$bundle" "$split_branch" --squash || true

  rm -f "$bundle"
  rm -rf "$tmpdir"
  echo "Imported $remote_subdir -> $prefix"
}

import_all(){
  echo "Importing subdirectories from $REPO_URL (single temporary clone)"

  tmpdir=$(mktemp -d /tmp/stanford-XXXX)
  echo "Cloning remote repository into temporary dir: $tmpdir"
  git clone --no-checkout "$REPO_URL" "$tmpdir"

  # If anything fails after this point, keep the temporary clone for inspection
  trap 'echo "Import failed — temporary clone preserved at: $tmpdir"; echo "You can inspect or remove it manually."; exit 1' ERR

  pushd "$tmpdir" >/dev/null
  git sparse-checkout init --no-cone
  # request both subpaths in the single clone
  git sparse-checkout set .cache/images/dataset .cache/images/xz_slices
  git checkout "${REPO_BRANCH}" || true

  # create split branches for each path
  split_branch_dataset="tmp-split-cache_images_dataset"
  split_branch_xz="tmp-split-cache_images_xz_slices"
  echo "Creating split branch $split_branch_dataset for .cache/images/dataset"
  git subtree split -P .cache/images/dataset -b "$split_branch_dataset" || true
  echo "Creating split branch $split_branch_xz for .cache/images/xz_slices"
  git subtree split -P .cache/images/xz_slices -b "$split_branch_xz" || true
  popd >/dev/null

  # Create bundle files for each split branch so we can fetch without contacting network
  bundle_dataset=$(mktemp /tmp/dataset-XXXX.bundle)
  bundle_xz=$(mktemp /tmp/xz-XXXX.bundle)
  git -C "$tmpdir" bundle create "$bundle_dataset" "$split_branch_dataset" || true
  git -C "$tmpdir" bundle create "$bundle_xz" "$split_branch_xz" || true

  # Ensure superproject working tree is clean before modifying with subtree
  if [ -n "$(git status --porcelain)" ]; then
    echo "Error: working tree has uncommitted changes. Please commit or stash before running import."
    echo "Aborting import. Temporary clone retained at: $tmpdir for inspection."
    exit 1
  fi

  # Back up and add each subtree into the superproject using the bundles (no network)
  mkdir -p .backup_data
  if [ -e "data/dataset" ]; then
    mv data/dataset .backup_data/dataset.bak
  fi
  git subtree add --prefix=data/dataset "$bundle_dataset" "$split_branch_dataset" --squash || true

  if [ -e "data/xz_slices" ]; then
    mv data/xz_slices .backup_data/xz_slices.bak
  fi
  git subtree add --prefix=data/xz_slices "$bundle_xz" "$split_branch_xz" --squash || true

  # Cleanup (only remove the temporary clone after successful import)
  rm -f "$bundle_dataset" "$bundle_xz"
  rm -rf "$tmpdir"
  # Remove the trap now that we're done successfully
  trap - ERR
  echo "Import complete. Verify with 'ls -la data' and 'git status'"
}

update_all(){
  echo "Updating imported subtrees from upstream ($REPO_URL -> data/*)"
  # Add remote if missing
  if ! git remote get-url stanford-upstream >/dev/null 2>&1; then
    git remote add stanford-upstream "$REPO_URL" || true
  fi
  git fetch stanford-upstream || true

  # Pull updates for each prefix. This uses subtree pull which expects the
  # upstream branch to be available on the remote.
  git subtree pull --prefix=data/dataset stanford-upstream "$REPO_BRANCH" --squash || echo "Failed to pull data/dataset (may be no changes)"
  git subtree pull --prefix=data/xz_slices stanford-upstream "$REPO_BRANCH" --squash || echo "Failed to pull data/xz_slices (may be no changes)"

  echo "Update complete. Run 'git status' to inspect changes."
}

case ${1:-help} in
  install)
    install_gitsubrepo
    ;;
  import)
    import_all
    ;;
  update)
    update_all
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: ${1:-}" >&2
    usage
    exit 2
    ;;
esac
