# Project rules for Claude Code

## Simplicity first
- Write the simplest code that correctly solves the task. Prefer plain 
  functions and NumPy over classes, abstractions, or design patterns.
- Do not add features, parameters, config options, or flexibility that 
  the current task does not need.
- No new dependencies beyond requirements.txt without asking first.
- Prefer fewer files and fewer lines, as long as readability does not 
  suffer. Readable beats clever.
- If two functions share most of their logic, factor out the shared 
  part instead of duplicating it.
- Comments explain WHY, not what. No comments restating the code.

## Deliberate design decisions (do not "simplify" these away)
- simulate_paths takes Z as an input and stores full paths. Full paths 
  are needed later for American options; Z as input is needed for 
  variance reduction and deterministic testing.
- Pricers return MCResult (price, std_error, n_paths), not a bare float.
- Antithetic standard error is computed over pair averages, not 
  individual payoffs.
- Random numbers come from np.random.default_rng, never the legacy 
  np.random API.

## Workflow
- Before making changes that touch more than one file, state a short 
  plan and wait for approval.
- After any change, run pytest. Do not commit if tests fail.
- Never change numerical results silently. If a change alters output 
  for a fixed seed, say so and explain why.
- Commit messages describe what changed and why, in one line.
