#!/bin/bash
# Rebrand AnythingLLM to RuyaLLM across the repository.
# This script will search and replace common variations of the name.
# Usage: bash scripts/rebrand.sh

set -e

# Directories to process
DIRECTORIES=(
  "browser-extension"
  "collector"
  ".github"
  ".devcontainer"
  "embed"
  "frontend"
  "locales"
  ".vscode"
  ".gitmodules"
  "cloud-deployments"
  "docker"
  "extras"
  "images"
  "server"
  "BARE_METAL.md"
  "Prg-ReadME.md"
  "README.md"
  "SECURITY.md"
)

# Exclude virtual environments or other large directories
EXCLUDES=("server/python-memory/venv")

# Build the grep exclude arguments
EXCLUDE_ARGS=()
for ex in "${EXCLUDES[@]}"; do
  EXCLUDE_ARGS+=(--exclude-dir="$ex")
done

replace_text() {
  local findstr="$1"
  local replacestr="$2"
  grep -Ilr "${findstr}" ${EXCLUDE_ARGS[@]} -- ${DIRECTORIES[@]} | xargs -r sed -i "s/${findstr}/${replacestr}/g"
}

# Lowercase and dashed
replace_text "anything-llm" "ruya-llm"
replace_text "anythingllm" "ruyallm"

# Title case
replace_text "AnythingLLM" "RuyaLLM"

# Uppercase
replace_text "ANYTHING-LLM" "RUYA-LLM"
replace_text "ANYTHINGLLM" "RUYALLM"

# With space
replace_text "Anything LLM" "Ruya LLM"

echo "Rebranding complete."
