# Task conventions — applies to `lab/tasks/` only

- [3. Task document template](#3-task-document-template)
  - [Key rules for task documents](#key-rules-for-task-documents)
- [4.12. Commit message format](#412-commit-message-format)
- [4.19. Steps with sub-steps](#419-steps-with-sub-steps)
- [12. Task design principles](#12-task-design-principles)
  - [12.1. Progressive complexity](#121-progressive-complexity)
  - [12.2. Every task teaches something](#122-every-task-teaches-something)
  - [12.3. Step-by-step instructions](#123-step-by-step-instructions)
  - [12.4. Provide fallback methods](#124-provide-fallback-methods)
  - [12.5. Localize instructions](#125-localize-instructions)
  - [12.6. Git workflow integration](#126-git-workflow-integration)
  - [12.7. Acceptance criteria](#127-acceptance-criteria)
  - [12.8. Hints and solutions](#128-hints-and-solutions)
  - [12.9. Expected output](#129-expected-output)
  - [12.10. Notes explain "why"](#1210-notes-explain-why)
  - [12.11. Three kinds of task endings](#1211-three-kinds-of-task-endings)
  - [12.12. Cross-task references](#1212-cross-task-references)
  - [12.14. Placeholder-based implementation templates](#1214-placeholder-based-implementation-templates)
  - [12.15. Seed project design](#1215-seed-project-design)
  - [12.16. Holistic task design](#1216-holistic-task-design)
  - [12.17. LLM-independence](#1217-llm-independence)
  - [12.18. Multi-bug debugging tasks](#1218-multi-bug-debugging-tasks)
  - [12.19. Step checkpoints](#1219-step-checkpoints)
  - [12.20. Recovery guidance](#1220-recovery-guidance)
  - [12.21. Controlled task environment](#1221-controlled-task-environment)
- [13. Conceptual review dimensions](#13-conceptual-review-dimensions)
  - [D1. Learning objective clarity](#d1-learning-objective-clarity)
  - [D2. Step-by-step completeness](#d2-step-by-step-completeness)
  - [D3. Student navigation](#d3-student-navigation)
  - [D4. Checkpoints and feedback loops](#d4-checkpoints-and-feedback-loops)
  - [D5. Acceptance criteria alignment](#d5-acceptance-criteria-alignment)
  - [D6. Difficulty and progression](#d6-difficulty-and-progression)
  - [D7. Practical usability](#d7-practical-usability)
  - [D8. LLM-independence](#d8-llm-independence)
  - [D9. Git workflow coherence](#d9-git-workflow-coherence)
  - [D10. Conceptual gaps and misconceptions](#d10-conceptual-gaps-and-misconceptions)
  - [D11. Controlled AI steps](#d11-controlled-ai-steps)
- [15. Testing pattern](#15-testing-pattern)
- [17. Checklist before publishing](#17-checklist-before-publishing)

## 3. Task document template

Every task file (`task-N.md`) must follow this structure:

```markdown
# <Task title>

<h4>Time</h4>

~<estimate> min

<h4>Purpose</h4>

<One sentence: what the student will learn.>

<h4>Context</h4>

<1–3 sentences: why this task matters and what background the student needs.>

<h4>Table of contents</h4>

- [1. Steps](#1-steps)
  - [1.1. Follow the `Git workflow`](#11-follow-the-git-workflow)
  - [1.2. Create an issue](#12-create-an-issue)
  - [1.3. <Step title>](#13-step-title)
    - [1.3.1. <Sub-step title>](#131-sub-step-title)
    - [1.3.2. <Sub-step title>](#132-sub-step-title)
  - ...
- [2. Acceptance criteria](#2-acceptance-criteria)

## 1. Steps

### 1.1. Follow the `Git workflow`

Follow the [`Git workflow`](../git-workflow.md) to complete this task.

### 1.2. Create a `Lab Task` issue

Title: `[Task] <Task title>`

### 1.3. <Step title>

- [1.3.1. <Sub-step title>](#131-sub-step-title)
- [1.3.2. <Sub-step title>](#132-sub-step-title)

#### 1.3.1. <Sub-step title>

<Step-by-step instructions>

#### 1.3.2. <Sub-step title>

<Step-by-step instructions>

### ...

### 1.N. Finish the task

1. [Create a PR](../git-workflow.md#create-a-pr) with your changes.
2. [Get a PR review](../git-workflow.md#get-a-pr-review) and complete the subsequent steps in the `Git workflow`.

---

## 2. Acceptance criteria

- [ ] <Criterion 1>
- [ ] <Criterion 2>
- ...
```

### Key rules for task documents

- **Time, Purpose, Context, Table of contents** use `<h4>` HTML tags so they don't appear in the document's auto-generated ToC.
- **Top-level sections are numbered:** `## 1. Steps` and `## 2. Acceptance criteria`. Steps are numbered as `### 1.1.`, `### 1.2.`, etc. This matches the pattern used in `setup.md` and makes anchor links unambiguous.
- When a `###` step covers multiple distinct sub-goals, split it into `####` sub-sections with a deeper number (`#### 1.3.1.`, `#### 1.3.2.`, etc.) and a descriptive title for each. Reflect the hierarchy in the ToC with indented entries. Add an inline mini-ToC (a bullet list of links to the sub-sections) right after the `###` heading so readers see the structure without scrolling back to the document-level ToC. Use a flat numbered list only when all actions serve a single, unified goal within the same sub-section.
- **Step 1.1** ("Follow the Git workflow") is present in tasks that require a branch + PR. Omit for tasks that don't produce commits (e.g., "Run the web server").
- **Step 1.2** is always "Create an issue" (either a `Lab Task` or specific issue type). When step 1.1 is omitted, "Create an issue" becomes step 1.1.
- The **last step** is either "Finish the task" (create PR, get review) or "Write a comment for the issue" (close with evidence).
- **Acceptance criteria** use `- [ ]` checkboxes. Reviewers check them during PR review.
- Acceptance criteria are concrete and verifiable: issue titles, passing tests, merged PRs, specific comments.

---

## 4.12. Commit message format

Use [conventional commits](https://www.conventionalcommits.org/):

```text
<type>: <short description>

- <detail 1>
- <detail 2>
```

Common types:

- `fix:` — bug fixes
- `feat:` — new features or additions
- `docs:` — documentation changes

When a task specifies a commit message, provide it in a code block:

```markdown
   Use a multi-line message:

   \`\`\`text
   <type>: <short description>

   - <detail 1>
   - <detail 2>
   \`\`\`
```

---

## 4.19. Steps with sub-steps

When multiple actions serve a single logical goal, group them under one step. Write the step as a complete sentence followed by "Complete these steps:", then list the sub-steps as a nested ordered list:

```markdown
1. Configure the environment. Complete these steps:
   1. Open `.env.example`.
   2. Copy it to `.env.secret`.
   3. Fill in the values.
```

When sub-items describe the behavior of an artifact being created (a workflow, config file, script, etc.) rather than actions the student performs, use "does the following:" instead. Write sub-items in third person to reflect what the artifact does:

```markdown
1. Add a workflow that does the following on every push to `main`:
   1. Checks out the repository.
   2. Runs all back-end unit tests.
   3. Runs all end-to-end tests.
```

When actions don't share a logical goal, flatten them into separate top-level steps (see [Instructions wording](./common.md#41-instructions-wording)).

---

## 12. Task design principles

### 12.1. Progressive complexity

- **Required tasks** build on each other and increase in complexity. A typical progression:
  - Early tasks: Run/explore something locally (minimal setup).
  - Middle tasks: Find and fix a problem, add a feature (debugging, testing, Git workflow).
  - Later tasks: Add infrastructure, containerize, or deploy (Docker, CI/CD, VM, cloud).
- **Start with observation, not coding.** The first tasks should be observation-only: explore the system, query endpoints, record what you see. Students should understand the system from all angles before implementing anything.
- Adapt the progression to the lab's topic. Not every lab needs Docker or deployment — the principle is that complexity increases across tasks.
- **Optional tasks** extend the lab with independent challenges.

### 12.2. Every task teaches something

Each task has a clear **Purpose** (what the student learns) and **Context** (why it matters). Tasks are not busywork — they simulate real engineering workflows.

When the lab has multiple domain entities (e.g., resources, models, tables), assign each entity a distinct learning role:

- **Reference** — study existing implementation — `items` endpoints (fully implemented)
- **Debug** — enable and fix broken code — `interactions` endpoint (commented out, contains bugs)
- **Implement** — build from a template — `learners` endpoint (placeholder-based template)

This prevents overlap and ensures each task has a unique learning objective.

### 12.3. Step-by-step instructions

- Provide explicit step-by-step instructions in the task document. Each numbered step should be a single concrete action — "Open the file", "Click `Execute`", "Run in the `VS Code Terminal`".
- Link to wiki sections for reusable tool operations (e.g., how to run a command in the terminal, how to open a file). This keeps task steps focused on the domain while letting beginners follow tool-specific guides.
- When a step repeats a process from an earlier task, reference the earlier task step instead of repeating the full instructions (see [12.12. Cross-task references](#1212-cross-task-references)).

### 12.4. Provide fallback methods

When one method to complete a step may not work (e.g., OS-specific), provide alternatives:

```markdown
View the file using one of the following methods.

Method 1:

1. [Run using the `VS Code Terminal`](...)

Method 2:

1. [Open the file](...)
```

### 12.5. Localize instructions

Provide instructions where they're easy to keep in mind. Don't make students jump between 5 different files to understand one step.

### 12.6. Git workflow integration

- Tasks that produce code changes always start with "Follow the `Git workflow`" (Step 0).
- The first step is always "Create an issue" with a specific title format: `[Task] <title>`.
- The last step is always finishing via PR or closing the issue with a comment.
- This teaches students the real-world cycle: Issue → Branch → Commits → PR → Review → Merge.

### 12.7. Acceptance criteria

- Every task ends with `## Acceptance criteria`.
- Criteria are concrete, binary, and verifiable by a PR reviewer.
- Use `- [ ]` checkbox format.
- **Criteria must match the task content.** Every criterion must trace back to a specific step or deliverable in the task. Don't list criteria for work the task doesn't ask for, and don't leave task deliverables uncovered by criteria.
- Examples of good criteria:
  - `Issue has the correct title.`
  - `All tests pass after the fix.`
  - `PR is approved.`
  - `PR is merged.`
  - `The comment with the JSON response exists.`

### 12.8. Hints and solutions

For debugging/problem-solving tasks, provide collapsible hints and solutions using `<details>` tags. Let students try first, then peek if stuck.

### 12.9. Expected output

After commands that produce output, show what the student should expect to see:

~~~markdown
2. The output should be similar to this:

   ```terminal
   <expected output>
   ```
~~~

### 12.10. Notes explain "why"

Use `> [!NOTE]` blocks to explain concepts inline without breaking the step flow:

```markdown
> [!NOTE]
> The `.venv` directory contains the virtual environment.
> That is, files and dependencies that are necessary to run the web server.
```

### 12.11. Three kinds of task endings

**Tasks that produce code** (bug fixes, new features):

- End with "Finish the task" → create PR, get review, merge.
- Acceptance criteria include: PR approved, PR merged, all tests pass.

**Tasks that don't produce code** (run server, deploy):

- End with "Write a comment for the issue" → paste evidence (e.g., JSON response), close issue.
- Acceptance criteria include: issue has correct title, comment with evidence exists.

**Tasks with auto-checked deliverables** (exploration, questionnaires):

- End with "Commit the deliverable file" → student fills in a structured file (e.g., questionnaire with single-value answers), checked automatically by regex or a script.
- Acceptance criteria include: deliverable file exists, all answers match the expected format, auto-checker passes.

### 12.12. Cross-task references

Later tasks can reference steps from earlier tasks instead of repeating them:

```markdown
1. [Run the web server](./task-1.md#8-run-the-web-server).
```

### 12.14. Placeholder-based implementation templates

When a task requires students to implement new code (e.g., a new endpoint), provide commented-out placeholder templates in the seed project. Students uncomment the code and replace placeholders with correct values, using an existing reference implementation as a guide.

~~~markdown
Placeholder template (in `src/app/routers/learners.py`):

```python
# UNCOMMENT AND FILL IN

# @router.<method>("/<resource_name>", response_model=List[<resource_schema>])
# def <function_name>(<query_param>: <type> = None):
#     """<docstring>"""
#     return <db_read_function>(<query_param>)

# Reference:
# items -> items_table (in db), ItemModel
# learners -> learners_table (in db), LearnerModel
```
~~~

Key rules:

- Placeholders use `<angle_brackets>` to indicate values students must fill in.
- Each placeholder template includes a `# Reference:` comment mapping the new resource to its reference counterpart.
- The reference implementation (e.g., `items` endpoint) must be fully working so students can study it.
- Each placeholder template should be a separate commit when implemented.

### 12.15. Seed project design

The seed project is the starting codebase students receive. Design it with three tiers of completeness:

1. **Fully implemented (reference):** One resource is complete and working. Students study it to understand the pattern. Example: `items` endpoints with all CRUD operations.
2. **Commented out with bugs (debug):** Code exists but is disabled. Students uncomment it, discover it fails, and debug. Example: `interactions` endpoint with a schema-database mismatch.
3. **Placeholder templates (implement):** Commented-out code with `<placeholders>` that students fill in by following the reference. Example: `learners` endpoint with `<method>`, `<resource_name>`, `<resource_schema>`.

For each tier, both the route code and its router registration (e.g., `app.include_router(...)`) must be in the same state — commented out or active. Students uncomment both to enable the route.

### 12.16. Holistic task design

Combine related concerns into a single task when they share the same learning objective. A debugging task should include everything needed to understand the failure: reading code, examining the database, and fixing the bug — not three separate tasks.

Separate concerns into different tasks only when they produce fundamentally different artifacts or teach distinct skills. Example: API exploration via Swagger (produces a questionnaire) and database exploration via PgAdmin (produces a bug fix) belong in different tasks even though both involve the same system.

### 12.17. LLM-independence

Tasks must be completable without LLMs unless the task explicitly states that students must use an AI. This means:

- Provide placeholder templates, clear examples, and explicit step-by-step guidance.
- Use simple, direct language in student-facing materials.
- Provide fallback methods for every major operation.
- The "Learning advice" section encourages LLM use for understanding, but tasks must not require it.
- When a task explicitly requires AI use (e.g., "Generate tests with an AI agent"), mark it as a separate, clearly labeled part so students and reviewers can distinguish AI-required steps from AI-optional ones.

### 12.18. Multi-bug debugging tasks

When designing debugging tasks, include multiple bugs at different layers of the request path so students learn to trace failures across the stack:

- **Schema–database mismatch:** A field name in the Pydantic model doesn't match the database column. Students discover this by comparing the model to the table schema (e.g., in PgAdmin).
- **Logic error:** A variable name or condition is wrong in the data processing code. Students discover this by reading the function and tracing its behavior.

Structure the task so each bug is discovered sequentially: the first fix unblocks progress but reveals the next failure. Provide collapsible hints for each bug.

### 12.19. Step checkpoints

Every non-trivial step should include a checkpoint — a quick way for students to verify they completed the step correctly and are in the right environment. Without checkpoints, students may proceed through multiple steps before discovering something went wrong early on, making debugging much harder.

Checkpoints can take different forms:

- **Expected output:** Show the terminal output the student should see (see [12.9](#129-expected-output)).
- **Smoke test:** A quick command or action that confirms the change worked (e.g., "Refresh the page and verify the new endpoint appears").
- **Visual confirmation:** A screenshot or description of what the UI should look like after the step.
- **State check:** A command that shows the current state (e.g., "Run `git status` and verify you see the new file").

**Checkpoints are part of the step, not separate steps.** Indent checkpoint content (text, screenshots, expected output) under the action step it verifies. This keeps step counts meaningful (only real actions are numbered) and lets readers scan for what to *do* next without confusing verifications for actions.

Good:

~~~markdown
1. [Open `Swagger UI`](../../wiki/swagger.md#open-swagger-ui).

   You should see the `Swagger UI` page with the API documentation.

   <img alt="Swagger UI" src="../images/tasks/setup/swagger-ui.png" style="width:400px"></img>
~~~

Bad:

~~~markdown
1. [Open `Swagger UI`](../../wiki/swagger.md#open-swagger-ui).

2. You should see the `Swagger UI` page with the API documentation.

   <img alt="Swagger UI" src="../images/tasks/setup/swagger-ui.png" style="width:400px"></img>
~~~

### 12.20. Recovery guidance

Steps that involve infrastructure or environment-dependent operations (`Docker`, databases, services on ports) can fail for reasons outside the task's scope — port conflicts, stale containers, missing environment variables. Instead of directing students to "ask the TA," include a collapsible troubleshooting block so students can self-diagnose common failures.

Key rules:

- **Place after the checkpoint.** The troubleshooting block follows the "You should see…" checkpoint, because students only need it when the checkpoint fails.
- **Use the summary `Troubleshooting`.** This keeps a consistent label that students learn to look for.
- **Use `<h4>` for each symptom.** Start each entry with an `<h4>` tag containing the symptom (what the student sees), then the fix. `<h4>` renders as a visible heading but stays out of the auto-generated ToC — the same pattern used for Time, Purpose, and Context in the task template.
- **Keep it brief.** Cover only the 2–3 most common failures per block. Rare edge cases can still go to the TA.
- **Only add to infrastructure steps.** Steps involving external systems or environment-dependent operations where common failures are predictable. Simple file edits or `Git` commands don't need troubleshooting blocks.

Good:

~~~markdown
1. Start `Docker` containers.

   You should see all containers running.

   <details><summary>Troubleshooting</summary>

   <h4>Port conflict (<code>port is already allocated</code>)</h4>

   Stop the process that uses the port, then retry.

   <h4>Containers exit immediately</h4>

   To rebuild all containers from scratch,

   [run in the `VS Code Terminal`](...):

   ```terminal
   docker compose down && docker compose up --build
   ```

   </details>
~~~

Bad:

~~~markdown
4. Ask the TA if something doesn't work.
~~~

### 12.21. Controlled task environment

Every task must produce predictable, repeatable outcomes for all students — regardless of their setup, skill level, or tool choices. Instructions must eliminate free variables: specify exact files, commands, values, and expected outputs. Leave nothing to interpretation.

This principle extends to AI-assisted steps. When a task requires students to use an AI tool (e.g., generate code, write a prompt, analyze output), the step must constrain the AI interaction precisely enough that the result is reproducible:

- Provide the exact prompt to use, or a template with clearly marked `<placeholders>`.
- Specify what the AI output should contain or how to verify it (see [12.19. Step checkpoints](#1219-step-checkpoints)).
- If AI output varies, require students to adapt it to a concrete acceptance criterion rather than produce "something reasonable."

Together with [step checkpoints](#1219-step-checkpoints), a controlled environment guarantees that every student who follows the instructions reaches the same verified state at the end of each step — making lab completion predictable and reviewer-verifiable.

---

## 13. Conceptual review dimensions

Use these dimensions when reviewing a task file for conceptual and educational problems. Conceptual review is distinct from convention review (formatting, naming, structure) — it evaluates whether the task works as a learning experience.

For each problem found, record: the dimension, the line number(s) or section, a short description of the problem, and a suggested fix. Distinguish severity:

- **High** — student would be blocked or form a wrong understanding
- **Medium** — student would be confused or slowed down significantly
- **Low** — minor gap that is unlikely to cause real trouble

### D1. Learning objective clarity

- Does the **Purpose** state a concrete, single learning outcome (not a vague "learn about X")?
- Does the **Context** explain *why* this task matters to a working engineer?
- Does the task content actually deliver on the stated Purpose?

### D2. Step-by-step completeness

- Is every action a single, concrete instruction a beginner can execute?
- Are there compound instructions hiding multiple actions in one step? (e.g., "Open the file and change the value and save")
- Are there ambiguous verbs without a clear target? (e.g., "Update the configuration" with no file path or key name)
- Are there prerequisite assumptions not covered by earlier tasks or `setup.md`?

### D3. Student navigation

- Can a student follow the task linearly without jumping between files?
- When the task references another file or wiki section, is the link present and the referenced section relevant?
- Is the Table of Contents accurate and complete relative to the actual headings?

### D4. Checkpoints and feedback loops

- Does every non-trivial step include a checkpoint (expected output, smoke test, visual confirmation, state check)?
- Are checkpoints indented under the action step they verify, not numbered as separate steps?
- For infrastructure or environment-dependent steps (Docker, databases, port-bound services), is there a collapsible troubleshooting block?

### D5. Acceptance criteria alignment

- Is there a criterion for every deliverable produced by the task?
- Is there a criterion not backed by any step in the task?
- Are all criteria concrete and binary (pass/fail), not subjective ("looks correct")?

### D6. Difficulty and progression

- Is the task's complexity appropriate for its position in the task sequence (setup → observe → debug → implement → deploy)?
- Does the task jump to implementation without first building the student's mental model?
- Does the task repeat learning objectives already covered by a prior task without adding new depth?

### D7. Practical usability

- Would a student on a fresh setup be able to complete the task without TA help beyond documented troubleshooting?
- Are there steps that could silently fail (no output, no checkpoint) leaving the student unaware of a problem?
- Are hints or collapsible solutions provided for debugging/problem-solving steps where a student is expected to search for the answer?

### D8. LLM-independence

- Is the task completable without an LLM? If it requires AI use, is that stated explicitly?
- Are placeholders, examples, and step-by-step guidance sufficient for a student who doesn't use AI assistance?

### D9. Git workflow coherence

- If the task produces code changes: does it start with "Follow the `Git workflow`", include "Create an issue", and end with "Finish the task" (PR + review)?
- If the task does not produce code: is the ending appropriate (issue comment with evidence, or committed deliverable file)?
- Is the issue title format specified (`[Task] <title>`)?

### D10. Conceptual gaps and misconceptions

- Does the task ask students to do something without explaining why (missing `> [!NOTE]` where the reasoning isn't obvious)?
- Could any step lead a student to form a wrong mental model (e.g., always deleting and recreating containers instead of understanding state)?
- Does the task introduce a concept without any reference to learn more (wiki link, note, or pointer)?

### D11. Controlled AI steps

When a task includes AI-assisted steps:

- Is the prompt exact or templated with `<placeholders>`? (Not "ask AI to help you" but "use this prompt:")
- Is there a checkpoint specifying what correct AI output looks like or how to verify it?
- If AI output is variable, is there a concrete acceptance criterion the student must satisfy regardless of what the AI produced?

---

## 15. Testing pattern

> Include this section if the lab has application code with tests. Omit for labs that are purely documentation- or configuration-focused.

- Include a `tests/` directory with test files.
- Use the project's test runner (e.g., `pytest`, `jest`, `go test`) configured in the package manager config.
- At least one test should **intentionally fail** so students can practice debugging.
- Tasks should instruct students to run tests and interpret output.
- Guide students through reading test output step-by-step. Break down the failure message into its components (test file, test name, assertion, expected vs. actual values):

  ```markdown
  1. Look at the test summary.
  2. You should see `FAILED <test-file>::<test-name> - assert <actual> == <expected>`.

     This line means the following:
     - The test failed (`FAILED`).
     - The test is in the file `<test-file>`.
     - The name of the failing test is `<test-name>`.
     - The assert that failed is `<actual> == <expected>`.
  ```

- Adapt the output format to the lab's test runner. The principle is the same: teach students to read and understand test output.
- Acceptance criteria should include "All tests pass."
- **Vary bug types across the request path.** When a lab includes multiple bugs, place them at different layers (e.g., schema–database mismatch at the data layer, logic error at the processing layer). This teaches students to trace failures across the full stack, not just look for one kind of mistake.

---

## 17. Checklist before publishing

**Always required:**

- [ ] `README.md` has: story, learning advice, learning outcomes, task list.
- [ ] Every task file has: Time, Purpose, Context, ToC, Steps, Acceptance criteria.
- [ ] Every terminal command uses the "To…" intention pattern with a `` [run in the `VS Code Terminal`] `` link.
- [ ] Every Command Palette command has a `` [Run using the `Command Palette`] `` link prefix.
- [ ] All cross-references use relative paths and are valid.
- [ ] Wiki docs exist for every tool/concept linked from tasks.
- [ ] Issue templates (`01-task.yml`, `02-bug-report.yml`) are configured.
- [ ] PR template has a checklist.
- [ ] `.vscode/settings.json` and `.vscode/extensions.json` are configured.
- [ ] `.gitignore` excludes generated files and secrets for the lab's ecosystem.
- [ ] Ordered lists use `1. 2. 3.` (not `1. 1. 1.`).
- [ ] Non-trivial steps include a checkpoint (expected output, smoke test, visual confirmation, or state check).
- [ ] Infrastructure steps include a collapsible troubleshooting block (not "ask the TA").
- [ ] Compound instructions are split into separate steps.
- [ ] All sentences end with `.`.
- [ ] Options and steps are clearly differentiated.
- [ ] Tool/concept names are wrapped in backticks: `` `VS Code` ``, `` `Git` ``, `` `Docker` ``.
- [ ] `Git workflow` is referenced from tasks that produce code changes.
- [ ] Acceptance criteria are concrete and verifiable.
- [ ] Commit message format is documented (conventional commits).
- [ ] Setup instructions cover: fork, clone, install tools, configure environment.
- [ ] Branch protection rules are documented.
- [ ] Partner/collaborator setup is documented.
- [ ] `CONTRIBUTORS.md` exists with placeholder entry.
- [ ] Diagrams use `.drawio.svg` format.
- [ ] `<!-- TODO -->` markers exist for unfinished sections.

**Conditional (include when applicable):**

- [ ] `.env.example` files are provided; `.env.secret` files are gitignored (if the lab uses environment variables).
- [ ] `.dockerignore` excludes tests, docs, `.git/`, build caches, markdown files (if the lab uses Docker).
- [ ] At least one test intentionally fails for the debugging task (if the lab has a testing/debugging task).
- [ ] Task runner commands are documented in the config file (if the lab uses a task runner).
- [ ] Seed project has three tiers: reference (working), debug (commented out with bugs), implement (placeholder templates) (if the lab uses the seed project pattern).
- [ ] Placeholder templates include `# Reference:` comments mapping new resources to reference counterparts (if the lab uses placeholder-based implementation).
- [ ] All tasks are completable without LLMs, unless the task explicitly states that students must use an AI.
- [ ] Docker images use an institutional container registry (if the lab uses Docker in an institutional setting).
- [ ] API key or auth mechanism is set via environment variable and encountered naturally during exploration (if the lab includes security).
