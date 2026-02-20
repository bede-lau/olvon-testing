# Keyword Detection Patterns

Quick reference for prompt classification based on keywords.

## Bug/Error Detection (→ systematic-debugging)

**Strong indicators:**
- "error", "Error", "ERROR"
- "failing", "failed", "fails"
- "broken", "doesn't work", "not working"
- "crash", "crashed", "crashing"
- "bug", "issue", "problem"
- "unexpected", "wrong", "incorrect"
- Stack traces or error messages in prompt
- Test failure output
- "TypeError", "ReferenceError", "SyntaxError", etc.
- "500", "404", "403" (HTTP errors)
- "null", "undefined" in error context

**Phrases:**
- "Why is this..."
- "This should..."
- "It was working but now..."
- "I'm getting an error"
- "Tests are failing"

## Implementation Detection (→ executing-plans)

**Strong indicators:**
- "build", "create", "implement", "add"
- "feature", "functionality"
- "integrate", "integration"
- "refactor", "restructure"
- "migrate", "migration"
- "set up", "setup", "configure"
- Multi-part requests (contains "and then", "after that", numbered lists)

**Complexity markers:**
- Multiple files mentioned
- Architecture terms (component, service, module, layer)
- Database changes mentioned
- API endpoints involved
- Auth/security requirements

## Skill Creation Detection (→ skill-creator)

**Strong indicators:**
- "I often...", "I regularly...", "I always..."
- "Every time I...", "Whenever I..."
- "Can you remember how to..."
- "Same as last time"
- Third+ occurrence of similar task type
- Mentions of internal tools/APIs/schemas
- Company-specific workflows

**Domain-specific patterns:**
- Specific API that needs documentation
- Internal database schemas
- Recurring report formats
- Deployment procedures

## Domain Skill Matching

### Mobile (→ mobile-design, react-native-architecture)
- "React Native", "Expo", "mobile", "iOS", "Android"
- "touch", "gesture", "navigation"
- "app", "screen", "tab"

### Database (→ supabase-postgres-best-practices)
- "query", "SQL", "Postgres", "Supabase"
- "index", "performance", "slow query"
- "RLS", "policy", "migration"

### Auth (→ auth-implementation-patterns)
- "auth", "login", "signup", "OAuth"
- "JWT", "token", "session"
- "permission", "role", "RBAC"

### Frontend (→ frontend-design)
- "UI", "component", "page", "layout"
- "design", "style", "CSS"
- "web", "browser", "responsive"

### GPU/ML (→ gpu-worker-patterns)
- "GPU", "CUDA", "inference"
- "model", "ML", "machine learning"
- "batch processing", "worker"

### Mock Mode (→ mock-mode-patterns)
- "mock", "fake", "stub"
- "test mode", "development mode"
- "without API", "offline"

## Simple Task Detection (→ direct execution)

**Indicators of simplicity:**
- Single action requested
- Clear file path given
- Specific line/function mentioned
- "just", "quickly", "simply"
- Question format: "What is...", "Where is..."
- Small scope: one file, one function, one config

## Complexity Escalation

Escalate from simple → planned when:
- User adds requirements mid-task
- Initial change reveals dependencies
- Multiple files need coordinated changes
- Tests start failing
- "Actually, also..." patterns
