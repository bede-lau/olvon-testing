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

## CLAUDE.md Context Management

### Context Files

| File | Scope | When to Load |
|------|-------|--------------|
| `CLAUDE.md` (root) | Mobile app (Expo/React Native) | Work in `mobile-app/`, components, hooks, stores, app routes |
| `heavy-functions/CLAUDE.md` | GPU Worker (Python/FastAPI) | Work in `heavy-functions/`, API routes, services, ML pipeline |

### Step 0: Load Context (BEFORE routing)

**At session start or when switching domains:**

1. **Detect working directory** from the prompt or file paths mentioned
2. **Load relevant CLAUDE.md**:
   - `mobile-app/*`, `components/*`, `hooks/*`, `store/*`, `app/*` → Read root `CLAUDE.md`
   - `heavy-functions/*`, `api/*`, `services/*` → Read `heavy-functions/CLAUDE.md`
   - Both mentioned → Read both
3. **Use context** for architecture decisions, conventions, known gaps

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

1. **Identify scope** - Which CLAUDE.md is affected?
2. **Determine section** - What section needs update?
3. **Make minimal edit** - Add only essential info, keep concise
4. **Preserve structure** - Match existing formatting and tone

**Update format:**
```
Updated `CLAUDE.md` - Added [brief description] to [section name]
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

## Clarifying Questions (New Features & Major Changes)

**BEFORE planning or implementing new features/major changes, ask clarifying questions.**

### When to Ask

| Trigger | Examples |
|---------|----------|
| **New feature request** | "Add a wishlist", "Implement search", "Create a new screen" |
| **Major change** | "Refactor auth", "Change how X works", "Redesign the feed" |
| **Ambiguous scope** | "Improve performance", "Make it better", "Fix the UX" |
| **Multiple valid approaches** | Could use different patterns, libraries, or architectures |
| **User-facing changes** | Affects UI, behavior, or data the user interacts with |

### Question Categories

Ask questions from relevant categories:

**1. Scope & Requirements**
- What specific functionality is needed?
- What should happen when [edge case]?
- Are there any features explicitly NOT wanted?

**2. User Experience**
- How should the user interact with this?
- What feedback should the user see (loading, success, error)?
- Should this work offline?

**3. Data & State**
- Where should data come from (API, local, mock)?
- Should data persist across sessions?
- How does this relate to existing data/state?

**4. Integration**
- How does this connect to existing features?
- Are there dependencies on other systems?
- Should this trigger notifications/events elsewhere?

**5. Constraints**
- Any performance requirements?
- Must it match existing patterns or can it differ?
- Timeline or complexity constraints?

### Clarification Process

```
1. Read prompt → Detect new feature or major change
2. Identify gaps → What's unclear or has multiple interpretations?
3. Ask 2-4 focused questions → Don't overwhelm, prioritize most critical
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

**For major changes:**
```
This change affects [scope]. To ensure I implement correctly:
1. [Question about desired behavior]
2. [Question about edge cases]
3. [Question about integration points]
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
| **Repetitive Pattern** | Same type of task done 3+ times, boilerplate code, recurring workflow | skill-creator |
| **Existing Skill Match** | Matches description of an installed skill | Route to that skill |
| **Simple Task** | Single-step, clear action, no complexity | Direct execution |

### Step 2: Apply Context Optimizer (ALWAYS)

**Every response must follow context-optimizer rules:**

1. **Never repeat content** - No pasting code back, use `file:line` references
2. **No code unless asked** - Reference locations instead
3. **Concise by default** - "Done. Component updated." not paragraphs
4. **No narration** - Don't explain what you're about to do, just do it
5. **Smart verbosity** - Only elaborate when user asks or decision needs justification

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
"This pattern could benefit from a dedicated skill."
- Identify the repeating workflow
- Propose skill structure
- Create if user approves
```

### Step 4: Detect Skill Creation Opportunities

**Flag for potential new skill when:**

1. **Domain-specific knowledge** needed repeatedly (e.g., specific API, internal schema)
2. **Same workflow** executed 3+ times in similar form
3. **Complex procedure** that requires many steps to remember
4. **External tool integration** that needs specific patterns
5. **User explicitly mentions** doing something "often" or "regularly"

**When detected:**
```
"I notice this task involves [pattern]. Creating a dedicated skill would:
- Save tokens on future runs
- Ensure consistent execution
- Store reusable scripts/references

Should I create a skill for this?"
```

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

## Decision Tree

```
START
  │
  ├─ Step 0: Load CLAUDE.md context
  │   mobile-app/* → Read CLAUDE.md
  │   heavy-functions/* → Read heavy-functions/CLAUDE.md
  │   ↓
  │
  ├─ Is this a bug/error/failure?
  │   YES → systematic-debugging (NO FIXES until root cause found)
  │   NO ↓
  │
  ├─ Is this a new feature or major change?
  │   YES → ASK CLARIFYING QUESTIONS first
  │         Wait for answers → Confirm understanding
  │         Then proceed to planning ↓
  │   NO ↓
  │
  ├─ Is this a multi-step implementation?
  │   YES → executing-plans (batch execution with checkpoints)
  │   NO ↓
  │
  ├─ Does this match an existing skill?
  │   YES → Route to that skill
  │   NO ↓
  │
  ├─ Is this a repetitive pattern (3+ occurrences)?
  │   YES → Suggest skill-creator
  │   NO ↓
  │
  ├─ Simple task → Execute directly with context-optimizer rules
  │   ↓
  │
  └─ After completion: Update CLAUDE.md if significant changes made
```

## Anti-Patterns to Block

**Block these behaviors:**

| Anti-Pattern | Correct Behavior |
|--------------|------------------|
| Implementing new feature without clarifying | "Before I start, I need to clarify..." |
| Assuming user intent on major changes | Ask 2-4 focused questions first |
| Proposing fixes before investigation | "Applying systematic-debugging first" |
| Explaining before doing | Just do it |
| Pasting code back to user | Use `file:line` references |
| Starting implementation without plan | "Let me create a plan first" |
| Repeating same workflow manually | "This could be a skill" |
| Verbose success messages | "Done. [1-line summary]" |

## Response Templates

### Task Start
```
Using prompt-orchestrator. This is a [classification].
Applying: [skill list]
```

### Clarifying Questions (New Feature/Major Change)
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
This pattern repeats. Skill creation would save future tokens.
Create [skill-name] skill? (y/n)
```

### Simple Task
```
Done. [1-line summary]
```

## Integration with Existing Skills

When multiple skills apply, layer them:

1. **context-optimizer** - Always active (response formatting)
2. **Primary skill** - Based on task type
3. **Domain skill** - If task matches domain (mobile, auth, etc.)

## Efficiency Metrics

Track these per session:
- Response length (target: <200 words unless complexity demands)
- Code repetition (target: zero)
- Fix attempts before root cause (target: zero)
- Unplanned tasks (target: none)

## Remember

- **LOAD CLAUDE.md context first** - Before any task in mobile-app or heavy-functions
- **CLARIFY before implementing** - New features/major changes require 2-4 focused questions
- **ALWAYS apply context-optimizer rules** - Every response
- **NEVER fix before investigating** - Route bugs to systematic-debugging
- **PLAN before implementing** - Route features to executing-plans
- **DETECT repetition** - Suggest skill-creator when patterns emerge
- **ROUTE to domain skills** - When installed and matching
- **UPDATE CLAUDE.md after significant changes** - Keep documentation current

## Session End Checklist

Before ending a session with significant changes:

1. [ ] Were any new hooks/components created? → Update CLAUDE.md
2. [ ] Was architecture modified? → Update relevant section
3. [ ] Were new patterns established? → Add to Code Conventions
4. [ ] Were gaps discovered? → Add to Known Gaps
5. [ ] Were debugging tips learned? → Document for future reference
