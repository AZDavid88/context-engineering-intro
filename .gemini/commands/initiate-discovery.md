# Initiate Discovery Process

## Purpose
To guide a user through a structured, collaborative dialogue to define the high-level requirements, features, and technology stack for a new project *before* any implementation planning begins. This command solves the problem of a user having a vision but not the specific technical details required to create a high-quality Product Requirement Prompt (PRP).

The output of this command is a **`planning_prp.md`** file, which serves as a "Research Mandate" for the AI.

## Core Principles
1.  **Collaboration over Assumption:** The AI acts as an architectural consultant, not a mind-reader. It proposes, and the user decides.
2.  **De-risk Upfront:** By settling on features and technology early, we prevent wasted effort on PRPs built on flawed or incomplete assumptions.
3.  **Structured Dialogue:** The process is not an open-ended chat. It is a stateful, multi-step interrogation designed to extract all necessary information logically.
4.  **Explicit Artifacts:** The process generates a formal `planning_prp.md`, creating a clear, auditable link between the initial vision and the final research.

---

## Command Flow: `/initiate-discovery`

This is a conversational command. The AI will prompt the user at each of the following stages.

### Step 1: Project Vision & Goals
- **AI Asks:** "What is the high-level vision for your project? What problem are you trying to solve?"
- **User Provides:** A 1-3 sentence description of the project's purpose.
- **AI Confirms:** The AI summarizes its understanding of the vision.

### Step 2: Core Features & User Stories
- **AI Asks:** "What are the 1-3 most critical features this project needs to have? Please describe them as user stories if possible (e.g., 'As a user, I want to...')".
- **User Provides:** A list of core features.
- **AI Confirms:** The AI lists the features it has captured and asks for confirmation before proceeding.

### Step 3: Technology Stack Recommendation
- **AI Analyzes:** Based on the features, the AI analyzes potential technology stacks.
- **AI Proposes:** "Based on your requirements, I recommend the following stack(s)..." The AI will present 1-2 options with a brief explanation of the pros and cons for each.
- **User Decides:** The user selects a stack or proposes an alternative.
- **AI Confirms:** The AI restates the chosen technology stack.

### Step 4: Research Target Identification
- **AI Proposes:** "To effectively build with [Chosen Stack], I recommend we research the official documentation and best practices. I suggest we scrape the following URLs..." The AI provides a list of essential URLs.
- **AI Asks:** "Are there any other documentation pages, tutorials, or GitHub repositories you believe are essential for this project?"
- **User Provides:** Additional URLs, if any.

### Step 5: Final Review & Generation
- **AI Summarizes:** The AI presents a final summary of the entire plan: Project Name, Core Features, Technology Stack, and the complete list of Research Targets.
- **AI Asks for Approval:** "If this plan is correct, I will generate the `planning_prp.md` file. Shall I proceed?"
- **User Approves:** The user gives the final confirmation.

## Output
- **File Created:** `planning_prp.md` in the project's root directory.
- **Next Step Recommendation:** The AI will advise the user to review the generated file and then run the `/execute-research` command.

## Quality Checklist
- [ ] Did the AI guide the user through all 5 steps?
- [ ] Is the final `planning_prp.md` populated with the user's choices?
- [ ] Are the research URLs valid and relevant to the chosen technology?
- [ ] Is the user clear on the next step (`/execute-research`)?
