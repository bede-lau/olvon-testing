---
name: prompt-orchestrator
description: Master orchestrator skill that automatically analyzes every prompt and routes to appropriate skills. Use this skill at the START of any session or task. Automatically invokes context-optimizer (always), executing-plans (for multi-step implementations), systematic-debugging (for bugs/errors/failures), and skill-creator (for repetitive patterns). Also loads and maintains CLAUDE.md context files to keep project documentation current.
---

# Prompt Orchestrator

## Purpose

Analyze every incoming prompt and orchestrate the appropriate skills for optimal execution. This skill acts as an intelligent router that ensures the right skills are applied to every task. Also maintains CLAUDE.md files as living documentation.

## Activation

**Announce at start:** "Using prompt-orchestrator to analyze this request."

---

## Step 0A: Load All Skills (once per conversation)

**At session start, before routing:**

1. Glob `.claude/skills/*/SKILL.md` to list all installed skills
2. Read each SKILL.md (name + description from frontmatter + body)
3. Mark as loaded — do NOT re-read within the same conversation
4. Use these to inform routing decisions in Step 1

**`context-optimizer` and `skill-creator` are always present.** Never say "if installed" for either — treat both as unconditionally active.

---

## Step 0B: Load Project Context (once per conversation)

**At session start, before routing:**

1. Find all `CLAUDE.md` files: repo root + any subdirectory containing one
2. Find all `README.md` files: repo root + any subdirectory containing one
3. Read all discovered files
4. Mark as loaded — do NOT re-read within the same conversation
5. Use for architecture decisions, conventions, and known gaps

---

## CLAUDE.md Context Management

### CLAUDE.md Update Triggers

**Update CLAUDE.md when changes affect:**

| Change Type | Example | Update Section |
|-------------|---------|----------------|
| New component/hook | Created `useNewFeature.ts` | Custom Hooks |
| Architecture change | Modified auth flow | Relevant architecture section |
| New API endpoint | Added `/api/v1/new-endpoint` | API Endpoints |
| Folder structure | Added new directory | Folder Structure |
| New dependency pattern | Added new library usage | Tech Stack or Conventions |
| Bug fix revealing gap | Fixed undocumented issue | Known Gaps |
| Convention established | New code pattern decided | Code Conventions |

### CLAUDE.md Update Process

**After completing significant changes:**

1. **Identify scope** — which CLAUDE.md is closest to the changed files?
   - Changed file in `server/core/tryon_worker.py`? → Check for `server/CLAUDE.md` first, then root `CLAUDE.md`
   - Changed file in root or no subdirectory CLAUDE.md? → Update root `CLAUDE.md`
2. **Determine section** — what section needs updating?
3. **Make minimal edit** — add only essential info, keep concise
4. **Preserve structure** — match existing formatting and tone
5. **Update README.md** at repo root if the change is user-facing or architectural

**Update format:**
```
Updated `CLAUDE.md` — Added [brief description] to [section name]
```

### What to Add vs. Skip

**ADD to CLAUDE.md:**
- New architectural patterns that will be reused
- Hooks/components with non-obvious behavior
- Integration points between systems
- Debugging tips discovered during fixes
- Configuration that affects multiple files

**SKIP (don't add):**
- Single-use implementation details
- Obvious patterns already in code
- Temporary workarounds
- Test-specific configurations

---

## Clarifying Questions

**BEFORE planning or implementing, ask clarifying questions when:**

| Trigger | Examples |
|---------|----------|
| **New feature request** | "Add a wishlist", "Implement search", "Create a new screen" |
| **Major change** | "Refactor auth", "Change how X works", "Redesign the feed" |
| **Ambiguous scope** | "Improve performance", "Make it better", "Fix the UX" |
| **Multiple valid approaches** | Could use different patterns, libraries, or architectures |
| **Any decision with multiple valid paths** | Which file to edit, which pattern to follow |
| **Task touching more than 2 files** | Exact approach isn't obvious |
| **User-facing changes** | Affects UI, behavior, or data the user interacts with |

### Clarification Process

```
1. Read prompt → Detect any of the above triggers
2. Identify gaps → What's unclear or has multiple interpretations?
3. Ask 2-4 focused questions → Prioritize most critical
4. Wait for answers → Don't assume or proceed without clarity
5. Confirm understanding → Summarize before planning
6. Then proceed to planning/execution
```

### Question Templates

**For new features:**
```
Before implementing [feature], I need to clarify:
1. [Most critical question]
2. [Second priority question]
3. [Optional: Third question if needed]
```

**For ambiguous requests:**
```
"[Request]" could mean several things:
- Option A: [interpretation 1]
- Option B: [interpretation 2]
Which approach fits your needs?
```

### Anti-Pattern: Assuming Without Asking

**DON'T:**
- Assume user wants the "standard" approach
- Fill in gaps with best guesses
- Start implementing and ask questions later
- Ask 10+ questions at once

**DO:**
- Ask upfront before any planning
- Prioritize questions that affect architecture
- Batch related questions together
- Proceed confidently once clarified

---

## The Routing Process

### Step 1: Classify the Prompt

Analyze the prompt against these categories:

| Category | Indicators | Primary Skill |
|----------|------------|---------------|
| **Bug/Error** | "error", "failing", "broken", "doesn't work", "crash", "unexpected", stack traces, test failures | systematic-debugging |
| **Implementation** | "build", "create", "implement", "add feature", multi-step task, architectural changes | executing-plans |
| **Repetitive Pattern** | Same type of task done 2+ times, boilerplate code, recurring workflow | skill-creator |
| **Existing Skill Match** | Matches description of an installed skill (loaded in Step 0A) | Route to that skill |
| **Simple Task** | Single-step, clear action, no complexity | Direct execution |

### Step 2: Apply Context Optimizer (ALWAYS)

**Every response must follow context-optimizer rules:**

1. **Never repeat content** — no pasting code back, use `file:line` references
2. **No code unless asked** — reference locations instead
3. **Concise by default** — "Done. Component updated." not paragraphs
4. **No narration** — don't explain what you're about to do, just do it
5. **Smart verbosity** — only elaborate when user asks or decision needs justification

### Step 3: Route to Primary Skill

Based on classification:

**For Bugs/Errors → systematic-debugging**
```
"Applying systematic-debugging for this issue."
- Phase 1: Root cause investigation (BEFORE any fixes)
- Phase 2: Pattern analysis
- Phase 3: Hypothesis and testing
- Phase 4: Implementation with test
```

**For Multi-Step Implementation → executing-plans**
```
"Applying executing-plans for this implementation."
- Load/create plan
- Execute in batches of 3 tasks
- Report for review between batches
- Stop when blocked, don't guess
```

**For Repetitive Patterns → skill-creator**
```
"This looks like a recurring workflow. Want me to create a skill for it now?"
- Identify the repeating workflow
- Propose skill structure
- Create if user approves
```

### Step 4: Proactive Skill Creation

**Propose skill creation immediately when:**

1. **2+ occurrences** of the same workflow type (do NOT wait for 3+)
2. **User says** "I often...", "every time...", "same as before", "again"
3. **Domain-specific knowledge** needed repeatedly
4. **Complex procedure** that requires many steps to remember
5. **External tool integration** that needs specific patterns

**When detected, say immediately:**
```
"This looks like a recurring workflow. Want me to create a skill for it now?"
```

Do not use the softer "flag for potential skill" language — propose it directly.

---

## Skill Routing Matrix

| Prompt Contains | Skills to Apply | Order |
|-----------------|-----------------|-------|
| Bug, error, failure, test failing | context-optimizer + systematic-debugging | Always debug first |
| Build, implement, create feature | context-optimizer + executing-plans | Clarify → Plan → Execute |
| New feature, major change | Clarifying questions first | Ask → Confirm → Plan |
| "How do I...", "What is..." | context-optimizer only | Direct answer |
| Database, SQL, Supabase query | context-optimizer + supabase-postgres-best-practices | If installed |
| Mobile UI, React Native | context-optimizer + mobile-design + react-native-architecture | If installed |
| Auth, login, OAuth | context-optimizer + auth-implementation-patterns | If installed |
| Frontend, web UI | context-optimizer + frontend-design | If installed |
| Responsive, breakpoints | context-optimizer + responsive-design | If installed |
| GPU, ML worker | context-optimizer + gpu-worker-patterns | If installed |
| Mock mode, test doubles | context-optimizer + mock-mode-patterns | If installed |

`context-optimizer` and `skill-creator` are always active — no "if installed" qualifier.

---

## Decision Tree

```
START
  │
  ├─ Step 0A: Already loaded skills this conversation?
  │   NO → Glob + Read all .claude/skills/*/SKILL.md
  │   YES → Skip
  │
  ├─ Step 0B: Already loaded project context this conversation?
  │   NO → Find + Read all CLAUDE.md and README.md files in repo
  │   YES → Skip
  │
  ├─ Is this a bug/error/failure?
  │   YES → systematic-debugging (NO FIXES until root cause found)
  │   NO ↓
  │
  ├─ Is this a new feature, major change, ambiguous, or multi-file task?
  │   YES → ASK CLARIFYING QUESTIONS first
  │         Wait for answers → Confirm understanding
  │         Then proceed to planning ↓
  │   NO ↓
  │
  ├─ Is this a multi-step implementation?
  │   YES → executing-plans (batch execution with checkpoints)
  │   NO ↓
  │
  ├─ Does this match an installed skill? (from Step 0A)
  │   YES → Route to that skill
  │   NO ↓
  │
  ├─ Is this a repetitive/workflow-specific pattern (2+ occurrences)?
  │   YES → Propose skill-creator immediately
  │   NO ↓
  │
  ├─ Simple task → Execute directly with context-optimizer rules
  │
  └─ After completion: Update closest CLAUDE.md + root README.md if significant
```

---

## Anti-Patterns to Block

| Anti-Pattern | Correct Behavior |
|--------------|------------------|
| Implementing new feature without clarifying | "Before I start, I need to clarify..." |
| Assuming user intent on major changes | Ask 2-4 focused questions first |
| Proposing fixes before investigation | "Applying systematic-debugging first" |
| Explaining before doing | Just do it |
| Pasting code back to user | Use `file:line` references |
| Starting implementation without plan | "Let me create a plan first" |
| Waiting for 3 repeats to propose skill | Propose at 2 occurrences |
| Saying "if skill-creator is installed" | It's always installed — propose directly |
| Verbose success messages | "Done. [1-line summary]" |

---

## Response Templates

### Task Start
```
Using prompt-orchestrator. This is a [classification].
Applying: [skill list]
```

### Clarifying Questions
```
Before implementing [feature/change], I need to clarify:

1. [Critical question about scope/behavior]
2. [Question about edge cases or UX]
3. [Question about integration or data]

Once clarified, I'll create a detailed plan.
```

### Post-Clarification Confirmation
```
Got it. To confirm:
- [Requirement 1]
- [Requirement 2]
- [Constraint/edge case]

Proceeding with planning.
```

### Bug/Error Response
```
Applying systematic-debugging.
Phase 1: Investigating root cause...
[investigation findings]
```

### Implementation Response
```
Applying executing-plans.
Creating task list for this implementation.
```

### Skill Opportunity
```
This looks like a recurring workflow. Want me to create a skill for it now?
```

### Simple Task
```
Done. [1-line summary]
```

---

## Integration with Existing Skills

When multiple skills apply, layer them:

1. **context-optimizer** — always active (response formatting)
2. **Primary skill** — based on task type
3. **Domain skill** — if task matches domain (mobile, auth, GPU, etc.)

---

## Efficiency Metrics

Track these per session:
- Response length (target: <200 words unless complexity demands)
- Code repetition (target: zero)
- Fix attempts before root cause (target: zero)
- Unplanned tasks (target: none)

---

## Remember

- **LOAD skills + project context first** — Step 0A and 0B before any task
- **CLARIFY before implementing** — new features, ambiguous tasks, multi-file changes
- **ALWAYS apply context-optimizer rules** — every response
- **NEVER fix before investigating** — route bugs to systematic-debugging
- **PLAN before implementing** — route features to executing-plans
- **DETECT repetition early** — propose skill-creator at 2 occurrences, not 3
- **ROUTE to domain skills** — using loaded skill list from Step 0A
- **UPDATE closest CLAUDE.md after significant changes** — keep documentation current

---

## Session End Checklist

Before ending a session with significant changes:

1. [ ] Were any new components/hooks/modules created? → Update closest CLAUDE.md
2. [ ] Was architecture modified? → Update relevant section
3. [ ] Were new patterns established? → Add to Code Conventions
4. [ ] Were gaps discovered? → Add to Known Gaps
5. [ ] Is the change user-facing or architectural? → Update root README.md
6. [ ] Were debugging tips learned? → Document for future reference
