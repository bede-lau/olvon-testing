# CLAUDE.md Maintenance Patterns

Reference for keeping CLAUDE.md files current and useful across any repository.

## Context File Discovery

At session start (Step 0B of prompt-orchestrator), discover all context files:

1. Find all `CLAUDE.md` files in the repo (root + all subdirectories)
2. Find all `README.md` files in the repo (root + all subdirectories)
3. Read all discovered files
4. Use them for architecture decisions, conventions, and known gaps

## Proximity Rule: Which CLAUDE.md to Update

After making changes, update the CLAUDE.md **closest** to the changed files:

```
Changed file path → Check for CLAUDE.md moving up the directory tree

Example: server/core/tryon_worker.py was modified
  → Does server/core/CLAUDE.md exist? → Update it
  → Does server/CLAUDE.md exist?      → Update it
  → Fall back to root CLAUDE.md
```

**Also update root README.md** if the change is user-facing or architectural.

## Section Update Patterns

### Tech Stack
**Update when:** Adding new major dependency or changing versions

```markdown
## Tech Stack
- **New Category:** library-name vX.X
```

### Folder Structure
**Update when:** Creating new directories or reorganizing

```markdown
## Folder Structure
  new-folder/       # Brief description of purpose
```

### Code Conventions
**Update when:** Establishing new patterns that should be followed

```markdown
## Code Conventions
- New convention: description of when/how to apply
```

### Custom Hooks / Modules
**Update when:** Creating hooks or modules with non-obvious behavior or dependencies

```markdown
## Custom Hooks (or relevant section)
- `useNewHook()` - Brief description, returns { data, loading, error, refresh }
```

### Architecture Sections
**Update when:** Modifying how systems communicate or are structured

```markdown
## [Feature] Architecture
- **Component:** `path/to/file.py` - What it does
- **State:** How state is managed
- **Flow:** Step 1 → Step 2 → Step 3
```

### Known Gaps
**Update when:** Discovering missing functionality or placeholder code

```markdown
## Known Gaps
- **New Gap:** Description — what needs to be done to resolve
```

### API Endpoints
**Update when:** Adding or modifying endpoints

```markdown
## API Endpoints
| `/api/v1/new-endpoint` | METHOD | Brief description |
```

## Update Examples

### Example 1: New Module Created

**Change:** Created `server/core/new_worker.py`

**Proximity check:** Does `server/core/CLAUDE.md` exist? → `server/CLAUDE.md`? → root `CLAUDE.md`

**Update to closest CLAUDE.md:**
```markdown
## Pipeline Components
- `server/core/new_worker.py` - Does X with Y, returns Z
```

### Example 2: Architecture Change

**Change:** Modified auth flow to use refresh tokens

**Update to closest CLAUDE.md:**
```markdown
## Auth Architecture
- **Token refresh:** AuthProvider now handles automatic token refresh via `refreshSession()`
- 15-minute token expiry with silent refresh
```

### Example 3: New Convention

**Change:** Established pattern for error handling

**Update to closest CLAUDE.md:**
```markdown
## Code Conventions
- All ML fallbacks logged via `diagnostics.py` with GPU state snapshot
```

### Example 4: Gap Discovery

**Change:** Found that image caching isn't implemented

**Update to closest CLAUDE.md:**
```markdown
## Known Gaps
- **Image caching:** Feed images re-download on each render — implement caching layer
```

## When NOT to Update

- Single-line bug fixes
- Formatting changes
- Test file additions
- Temporary debug code
- Changes that don't affect how future work should be done

## Quality Checklist

Before updating CLAUDE.md:

1. [ ] Is this information useful for future sessions?
2. [ ] Would another developer need to know this?
3. [ ] Is it concise (1-2 lines if possible)?
4. [ ] Does it match existing formatting?
5. [ ] Is it in the right section (closest CLAUDE.md to the changed files)?
6. [ ] Does the change also warrant a README.md update at repo root?
