#!/usr/bin/env bash
# SessionStart hook for superpowers skills

set -euo pipefail

# Determine the root directory (where .claude is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
CLAUDE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Read using-superpowers content
using_superpowers_content=$(cat "${CLAUDE_ROOT}/skills/using-superpowers/SKILL.md" 2>&1 || echo "Error reading using-superpowers skill")

# Escape output for JSON
using_superpowers_escaped=$(echo "$using_superpowers_content" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | awk '{printf "%s\\n", $0}')

# Output context injection as JSON
cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "<EXTREMELY_IMPORTANT>\nYou have superpowers.\n\n**Below is the full content of your 'using-superpowers' skill - your introduction to using skills. For all other skills, use the 'Skill' tool:**\n\n${using_superpowers_escaped}\n\n</EXTREMELY_IMPORTANT>"
  }
}
EOF

exit 0
