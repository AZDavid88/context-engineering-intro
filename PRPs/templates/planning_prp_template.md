name: "Planning PRP & Research Mandate Template v1"
description: |
  This document is the output of the /initiate-discovery command.
  It serves two purposes:
  1. A human-readable summary of the project's vision, features, and architecture.
  2. A machine-parsable manifest for the /execute-research command to scrape necessary documentation.

## Core Principles
1. **Clarity Before Code:** We define what we're building and why before planning the implementation.
2. **Research is Foundational:** We explicitly define and gather the required knowledge before attempting to generate a technical PRP.
3. **User-in-the-Loop:** The user's decisions at the discovery phase are captured here as the source of truth for the project's direction.

---
# YAML Frontmatter for Machine Parsing by /execute-research
#
# project_name: A clear, user-defined name for the project.
# status: Tracks the state of the research.
#   Values: pending_research, research_in_progress, research_complete, research_failed
# technology_stack: A list of technologies chosen during discovery.
# research_targets: The list of URLs to be scraped by Jina.
#   - url: The full URL for Jina to scrape.
#   - purpose: A brief, human-readable reason for scraping this URL.
#   - status: Tracks the state of the individual scrape.
#     Values: pending, success, failed
#   - output_file: The relative path where the scraped content will be saved.

project_name: "User-Defined Project Name"
date_generated: "YYYY-MM-DD"
status: "pending_research"

technology_stack:
  - name: "Python"
    version: "3.11"
    purpose: "Backend logic and API"
  - name: "FastAPI"
    version: "latest"
    purpose: "Web framework for the API"
  - name: "React"
    version: "18"
    purpose: "Frontend user interface"

research_targets:
  - url: "https://fastapi.tiangolo.com/"
    purpose: "Primary documentation for the FastAPI framework."
    status: "pending"
    output_file: "research/fastapi/docs_main.md"
  - url: "https://react.dev/learn"
    purpose: "Core concepts and tutorials for React."
    status: "pending"
    output_file: "research/react/learn_main.md"
---

# Human-Readable Section for Context
# This section is used by the AI during the final /generate-prp phase.

## 1. Project Vision
[A high-level summary of what the project aims to achieve, derived from the /initiate-discovery conversation. This should clearly state the problem being solved and the desired end-state.]

## 2. Core Features & User Stories
[A breakdown of the essential features identified during the discovery phase. This list defines the core scope of the project.]
- **User Story 1:** As a [user type], I want to [perform an action] so that [I can achieve a goal].
- **User Story 2:** ...

## 3. Architectural Decisions & Rationale
[A summary of the technology stack choices, explaining *why* each component was chosen. This provides crucial context for future decisions and for the AI when it generates the implementation PRP.]
- **Backend:** [e.g., Python with FastAPI was chosen for its performance and modern async capabilities.]
- **Frontend:** [e.g., React was chosen to create a dynamic, component-based user interface.]
- **Database:** [e.g., PostgreSQL was selected for its robustness and support for structured data.]

## 4. Open Questions & Risks to Guide Research
[A list of any unresolved questions or potential risks identified during the discovery conversation. This section is critical for focusing the research and ensuring the final PRP addresses these points directly.]
- **Question:** How will we handle user authentication and authorization between the React frontend and the FastAPI backend? The FastAPI security documentation should be a priority.
- **Question:** What is the best library for making API calls from the React application?
- **Risk:** The chosen external API has a strict rate limit. The implementation must include a robust rate-limiting client.
