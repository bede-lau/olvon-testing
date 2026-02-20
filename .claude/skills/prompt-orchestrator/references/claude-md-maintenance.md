# CLAUDE.md Maintenance Patterns

Reference for keeping CLAUDE.md files current and useful.

## File Locations

| File | Path | Scope |
|------|------|-------|
| Mobile App | `CLAUDE.md` (root) | Expo, React Native, components, hooks, stores |
| GPU Worker | `heavy-functions/CLAUDE.md` | FastAPI, Python services, ML pipeline |

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

### Custom Hooks
**Update when:** Creating hooks with non-obvious behavior or dependencies

```markdown
## Custom Hooks (Section)
- `useNewHook()` - Brief description, returns { data, loading, error, refresh }
```

### Architecture Sections
**Update when:** Modifying how systems communicate or are structured

```markdown
## [Feature] Architecture
- **Component:** `path/to/file.tsx` - What it does
- **State:** How state is managed
- **Flow:** Step 1 → Step 2 → Step 3
```

### Known Gaps
**Update when:** Discovering missing functionality or placeholder code

```markdown
## Known Gaps
- **New Gap:** Description — what needs to be done to resolve
```

### API Endpoints (heavy-functions)
**Update when:** Adding or modifying endpoints

```markdown
## API Endpoints
| `/api/v1/new-endpoint` | METHOD | Brief description |
```

## Update Examples

### Example 1: New Hook Created

**Change:** Created `hooks/useNewFeature.ts`

**Update to CLAUDE.md:**
```markdown
## Custom Hooks
- `useNewFeature()` - Fetches X data with caching, returns { data, loading, refetch }
```

### Example 2: Architecture Change

**Change:** Modified auth flow to use refresh tokens

**Update to CLAUDE.md:**
```markdown
## OAuth Flow Architecture
- **Token refresh:** AuthProvider now handles automatic token refresh via `refreshSession()`
- 15-minute token expiry with silent refresh
```

### Example 3: New Convention

**Change:** Established pattern for error boundaries

**Update to CLAUDE.md:**
```markdown
## Code Conventions
- Error boundaries wrap all route components in `app/_layout.tsx`
- Use `ErrorBoundary` from `components/ui/ErrorBoundary.tsx` for feature-level errors
```

### Example 4: Gap Discovery

**Change:** Found that image caching isn't implemented

**Update to CLAUDE.md:**
```markdown
## Known Gaps
- **Image caching:** Feed images re-download on each render — implement expo-image or react-native-fast-image
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
5. [ ] Is it in the right section?

## Path Detection Keywords

Use these to determine which CLAUDE.md to update:

**Root CLAUDE.md (mobile-app):**
- `mobile-app/`, `app/`, `components/`, `hooks/`, `store/`, `lib/`, `types/`
- Keywords: Expo, React Native, screen, component, hook, Zustand

**heavy-functions/CLAUDE.md:**
- `heavy-functions/`, `api/`, `services/`, `config/`
- Keywords: FastAPI, Python, endpoint, service, GPU, ML, Blender
