# Claude Superpowers Configuration

This directory contains Claude Code skills, commands, and hooks from the [superpowers](https://github.com/obra/superpowers) project.

## Structure

- **skills/** - Comprehensive skills library including:
  - Testing skills (TDD, async testing, anti-patterns)
  - Debugging skills (systematic debugging, root cause tracing, verification)
  - Collaboration skills (brainstorming, planning, code review, parallel agents)
  - Development skills (git worktrees, finishing branches, subagent workflows)
  - Meta skills (creating, testing, and sharing skills)

- **commands/** - Slash commands that activate specific skills:
  - `/brainstorm` - Interactive design refinement
  - `/write-plan` - Create implementation plan
  - `/execute-plan` - Execute plan in batches

- **hooks/** - SessionStart hook that loads the using-superpowers skill automatically

- **lib/** - Shared library files and utilities

## Usage

Claude will automatically:
1. Load the using-superpowers skill at session start via the SessionStart hook
2. Discover and make available all slash commands in the commands/ directory
3. Use relevant skills automatically when they match the current task

## Skills Available

### Testing
- test-driven-development - RED-GREEN-REFACTOR cycle
- condition-based-waiting - Async test patterns
- testing-anti-patterns - Common pitfalls to avoid

### Debugging
- systematic-debugging - 4-phase root cause process
- root-cause-tracing - Find the real problem
- verification-before-completion - Ensure it's actually fixed
- defense-in-depth - Multiple validation layers

### Collaboration
- brainstorming - Socratic design refinement
- writing-plans - Detailed implementation plans
- executing-plans - Batch execution with checkpoints
- dispatching-parallel-agents - Concurrent subagent workflows
- requesting-code-review - Pre-review checklist
- receiving-code-review - Responding to feedback
- using-git-worktrees - Parallel development branches
- finishing-a-development-branch - Merge/PR decision workflow
- subagent-driven-development - Fast iteration with quality gates

### Meta
- writing-skills - Create new skills following best practices
- sharing-skills - Contribute skills back via branch and PR
- testing-skills-with-subagents - Validate skill quality
- using-superpowers - Introduction to the skills system

## Source

These skills are from the superpowers project:
- Repository: https://github.com/obra/superpowers
- Version: 3.4.1
- License: MIT
- Author: Jesse Vincent

## Updating

To update the superpowers skills:
1. Clone the latest version: `git clone https://github.com/obra/superpowers.git /tmp/superpowers`
2. Copy the updated files: `cp -r /tmp/superpowers/skills/* .claude/skills/`
3. Copy updated commands: `cp -r /tmp/superpowers/commands/* .claude/commands/`
4. Copy updated lib: `cp -r /tmp/superpowers/lib/* .claude/lib/`
