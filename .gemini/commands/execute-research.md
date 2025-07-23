# Execute Research Mandate

## Purpose
To automate the data-gathering phase of a project. This command parses a `planning_prp.md` file (the "Research Mandate"), systematically scrapes the web pages listed in its `research_targets`, and saves the content into the `/research/` directory. It provides real-time feedback and updates the status of the mandate file to reflect its progress.

This command acts as the bridge between collaborative planning (`/initiate-discovery`) and detailed implementation planning (`/generate-prp`).

## Core Principles
1.  **Automation:** Eliminates the manual, error-prone task of scraping and saving documentation.
2.  **Reliability:** Incorporates retry logic for transient network errors and provides clear reporting for persistent failures.
3.  **Transparency:** Updates the `planning_prp.md` file in real-time, providing a persistent, auditable log of the research progress.
4.  **Idempotency:** If run again, it should ideally skip targets that are already marked as "success," preventing redundant work.

---

## Command Invocation
The command requires a single argument: the path to the `planning_prp.md` file to be executed.

**Example:**
`/execute-research planning_prp.md`

---

## Execution Flow

Upon invocation, the AI will perform the following steps:

### 1. Parse and Validate the Mandate
- **Read File:** The AI reads the specified `planning_prp.md` file.
- **Validate YAML:** It parses the YAML frontmatter and validates that required fields (`project_name`, `research_targets`, etc.) exist. If validation fails, the process stops with an error message.

### 2. Initiate Research
- **Update Main Status:** The AI immediately updates the `status` field in `planning_prp.md` from `pending_research` to `research_in_progress` and saves the file. This prevents duplicate runs.
- **Announce Start:** The AI informs the user that the research process has begun.

### 3. Iterate, Scrape, and Report
- The AI loops through each item in the `research_targets` list. For each target:
    - **Check Status:** If the target's `status` is already `success`, it is skipped.
    - **Announce Target:** "Scraping: [URL] for purpose: [Purpose]..."
    - **Create Directory:** It ensures the directory for the `output_file` exists (e.g., `mkdir -p research/fastapi/`).
    - **Execute Scrape:** It runs the Jina `curl` command.
    - **Handle Success:**
        - The output is saved to the specified `output_file`.
        - The target's `status` in `planning_prp.md` is updated to `success`.
        - A success message is displayed to the user.
    - **Handle Failure:**
        - The AI will retry the `curl` command up to 2 times for transient errors.
        - If it still fails, the target's `status` in `planning_prp.md` is updated to `failed`.
        - A clear error message is logged, explaining why the scrape failed for that specific target.

### 4. Finalize and Summarize
- **Update Final Status:** After the loop completes, the AI checks the status of all targets.
    - If all targets are `success`, the main `status` in `planning_prp.md` is set to `research_complete`.
    - If any target is `failed`, the main `status` is set to `research_failed`.
- **Generate Final Report:** The AI provides a summary to the user, listing:
    - All successfully created research files.
    - All targets that failed, along with the reason for failure.
- **Recommend Next Step:** If successful, the AI will recommend proceeding to the `/generate-prp` command, now that the context has been gathered.

## Output
- A `/research/` directory populated with Markdown files containing the scraped documentation.
- An updated `planning_prp.md` file with the final status of the research operation.

## Quality Checklist
- [ ] Did the command parse the `planning_prp.md` file correctly?
- [ ] Did the command attempt to scrape all research targets not already marked "success"?
- [ ] Were the output files created in the correct locations?
- [ ] Was the status of each target and the main status updated correctly in the `planning_prp.md` file?
- [ ] Did the final report accurately reflect the outcome of the operation?
