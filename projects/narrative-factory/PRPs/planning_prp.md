name: "Planning PRP & Research Mandate: Multi-Agent Narrative Pipeline"
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

project_name: "Multi-Agent Narrative Generation Pipeline"
date_generated: "2025-07-22"
status: "research_pending"

technology_stack:
  - name: "Reflex.dev"
    purpose: "Full-stack application framework (UI and Backend) in pure Python."
  - name: "Qdrant Cloud"
    purpose: "Managed vector database for the personas' long-term memory."
  - name: "Pydantic AI"
    purpose: "To structure and validate LLM outputs for reliable tool use."
  - name: "Google Gemini API"
    purpose: "The LLM powering the persona chat interfaces."
  - name: "FastEmbed"
    purpose: "Recommended library for creating vector embeddings for Qdrant."

research_targets:
  # Primary Targets (for immediate PoC implementation)
  - url: "https://reflex.dev/docs/getting-started/introduction/"
    purpose: "To build the full-stack application and the chat UIs in pure Python."
    status: "pending"
    output_file: "projects/narrative-factory/research/reflex/docs_main.md"
  - url: "https://qdrant.tech/documentation/"
    purpose: "To learn how to connect to Qdrant Cloud and manage vector data."
    status: "completed"
    output_file: "projects/narrative-factory/research/qdrant/research_summary.md"
  - url: "https://qdrant.tech/documentation/fastembed/"
    purpose: "To learn how to efficiently create vector embeddings."
    status: "completed"
    output_file: "projects/narrative-factory/research/fastembed/research_summary.md"
  - url: "https://ai.pydantic.dev/"
    purpose: "To structure and validate LLM outputs for reliable tool use."
    status: "pending"
    output_file: "projects/narrative-factory/research/pydantic_ai/docs_main.md"
  - url: "https://ai.google.dev/gemini-api/docs/"
    purpose: "To learn how to connect to and interact with the Gemini LLM."
    status: "pending"
    output_file: "projects/narrative-factory/research/gemini_api/docs_main.md"
  - url: "https://docs.jina.ai/"
    purpose: "To understand its API for future integration with the Librarian persona."
    status: "pending"
    output_file: "projects/narrative-factory/research/jina/docs_main.md"
  # Secondary Targets (for future pipeline implementation)
  - url: "https://docs.prefect.io/v3/get-started"
    purpose: "To research for future workflow orchestration between personas."
    status: "pending"
    output_file: "projects/narrative-factory/research/prefect/docs_main.md"
  - url: "https://controlflow.ai/welcome"
    purpose: "To research as an alternative for future workflow orchestration."
    status: "pending"
    output_file: "projects/narrative-factory/research/controlflow/docs_main.md"
---

# Human-Readable Section for Context

## 1. Project Vision
To create a sophisticated pipeline of five distinct AI personas (Director, Tactician, Weaver, Canonist, Librarian) that collaborate to write long-form, serial stories. They work by passing outputs to one another, all while reading from and writing to a shared, persistent Qdrant Cloud vector database to maintain narrative consistency.

## 2. Proof of Concept: Core Features & User Stories
The goal of this PoC is to build and validate the individual agents in a supervised environment.

- **User Story 1: Persona-Specific Interaction:** As a user, I can open separate chat interfaces for each persona and receive responses consistent with their unique roles.
- **User Story 2: Supervised Knowledge Retrieval:** When I send a prompt, the system will automatically query the Qdrant database for relevant context and add it to the LLM prompt.
- **User Story 3: Supervised Knowledge Creation:** As a user, I can manually provide the Canonist or Librarian with new information to be embedded and saved to the Qdrant database.
- **User Story 4: Supervised Knowledge Modification:** As a user, I can manually instruct the Canonist or Librarian to update an existing lore entry in the Qdrant database.

## 3. Architectural Decisions & Rationale
- **Full-Stack Framework:** We chose **Reflex.dev** to enable rapid development of the entire application (UI and backend) in pure Python, eliminating the need for JavaScript/TypeScript.
- **Database:** We chose **Qdrant Cloud** to provide a managed, scalable vector database without requiring local Docker installation.
- **Development Environment:** We will use a standard **Python Virtual Environment** to manage project dependencies locally.
- **Future Vision:** The architecture will be designed with future integration of **Jina** (for the Librarian) and a workflow orchestrator like **Prefect 2** in mind.

## 4. Open Questions & Risks to Guide Research
- **Primary Question:** What is the most robust pattern within Reflex.dev for creating a stateful chat interface that can handle asynchronous calls to the Gemini API and Qdrant?
- **Primary Question:** How should we structure our data collections within Qdrant (e.g., one collection for all lore, or separate collections for characters, locations, style guides)?
- **Risk:** The reliability of the entire system depends on the quality of the embeddings generated by `fastembed`. We must validate this early.
- **Risk:** The LLM's ability to use tools effectively depends on the prompts and the reliability of Pydantic AI. This needs to be a core focus of our testing.