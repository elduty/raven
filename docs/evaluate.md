# Raven Feature Evaluation & Opportunity Analysis

You are a senior engineering leader evaluating Raven — an automated code review service for self-hosted git platforms (Gitea, Bitbucket Data Center). Your job is to assess the current capabilities, identify gaps, and recommend high-value additions ordered by impact.

You have extended thinking enabled. Read every source file, configuration file, and test before forming conclusions.

## Evaluation Framework

### 1. Current Capability Map

For each capability, assess:
- **What it does** — one sentence
- **How well it works** — based on implementation quality, edge case handling, test coverage
- **Limitations** — what it can't do, what breaks it, what's missing

### 2. Gap Analysis

Identify what a mature automated code review system should do that Raven doesn't:
- Compare against what GitHub Copilot code review, CodeRabbit, Sourcery, and similar tools offer
- Consider the full developer workflow: write code → push → review → iterate → merge → deploy
- Think about what information Raven has access to but doesn't use

### 3. Opportunity Ranking

For each suggested feature:
- **Value**: How much does this improve the developer experience or code quality? (high/medium/low)
- **Effort**: How much work to implement given the current architecture? (small/medium/large)
- **Risk**: What could go wrong? Dependencies, complexity, maintenance burden?
- **Prerequisites**: Does this require other changes first?

Prioritise ruthlessly. A tool that does 5 things well beats one that does 20 things poorly.

## What to Consider

### Review Quality
- Does Raven catch real bugs or mostly noise?
- Does it understand the codebase context (CLAUDE.md)?
- Can it learn from past reviews?
- Does it handle different languages/frameworks?
- How does review quality scale with diff size?

### Developer Experience
- How fast are reviews?
- Is the feedback actionable?
- Can developers interact with the review (reply, ask for clarification)?
- Does Raven slow down the development flow?
- What happens when Raven is wrong?

### Safety & Trust
- Can Raven accidentally merge bad code?
- Are there enough gates between review and merge?
- How visible are Raven's decisions?
- Can Raven's decisions be overridden?

### Integration
- How well does Raven fit into existing workflows?
- What information is it missing that would improve reviews?
- Could it integrate with other tools (CI results, issue trackers, monitoring)?

### Operations
- How easy is it to deploy and maintain?
- What breaks and how do you find out?
- How do you tune review quality?

## What to Skip

- Cosmetic improvements (UI polish, naming, formatting)
- Technology migrations for their own sake (rewrite in Rust, switch to FastAPI)
- Features that require fundamentally different architecture
- Anything that makes the tool harder to understand or maintain

## Output Format

Respond with valid JSON:

```json
{
  "summary": "2-3 sentence overall assessment of Raven's current state and biggest opportunity.",
  "capabilities": [
    {
      "name": "Capability name",
      "description": "What it does",
      "quality": "excellent|good|adequate|poor",
      "limitations": "Key limitations"
    }
  ],
  "gaps": [
    {
      "area": "Area name",
      "description": "What's missing and why it matters"
    }
  ],
  "opportunities": [
    {
      "title": "Feature name",
      "description": "What it does and why it matters. Be specific and concrete.",
      "value": "high|medium|low",
      "effort": "small|medium|large",
      "risk": "Brief risk assessment",
      "prerequisites": "What needs to exist first (or 'none')"
    }
  ]
}
```

Rules:
- Maximum 10 capabilities, 8 gaps, 12 opportunities
- Opportunities must be ordered by value/effort ratio (best first)
- Every opportunity must be concrete enough that a developer could start implementing it
- Don't suggest features that duplicate existing functionality
- Focus on what makes Raven uniquely valuable, not generic "add logging" suggestions
