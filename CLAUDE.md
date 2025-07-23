### 🔄 Project Awareness & Context & Research
- **Documentation is a source of truth** - Your knowledge is out of date, I will always give you the latest documentation before writing any files that use third party API's - that information was freshsly scraped and you should NOT use your own knowledge, but rather use the documentation as a source of absolute truth.
- **Docker & Selftesting** - You must use Docker and you must do unit tests so when I open up the dashboard to use, it just works. I don't have to worry about fixing anything. You can use Docker with Curl, or just by running test scripts etc. Everything must be confirmed to work perfectly through Docker, you have access to the Docker CLI.
- **check all jina scrapes** - some Jina scrapes fail and have very little content in them - if this happens, try scraping again until it works and you get the actual content of the file.
- **Always read `PLANNING.md`** at the start of a new conversation to understand the project's architecture, goals, style, and constraints.
- **Check `TASK.md`** before starting a new task. If the task isn’t listed, add it with a brief description and today's date.
- **Use consistent naming conventions, file structure, and architecture patterns** as described in `PLANNING.md`.
- **Use Docker commands** whenever executing Python commands, including for unit tests.
- **Set up Docker** Setup a docker instance for development and be aware of the output of Docker so that you can self improve your code and testing.
- **Agents** - Agents should be designed as intelligent human beings by giving them decision making, ability to do detailed research using Jina, and not just your basic propmts that generate absolute shit. This is absolutely vital. They should not use programmatic solutions to problems - but rather use reasoning and AI decision making to solve all problems. Every agent should have at least 5 prompts in an agentic workflow to create truly unique content. Each agent should also have the context of what its previous iterations have made.
- **Stick to OFFICIAL DOCUMENTATION PAGES ONLY** - For all research ONLY use official documentation pages. Use a r.jina scrape on the documentation page given to you in intitial.md and then create a llm.txt from it in your memory, then choose the exact pages that make sense for this project and scrape them using your internal scraping tool.
**Create full production ready code**
- **Ultrathink** - Use Ultrathink capabilities before every stage of the PRP generation and code generation, what informatoin to put into PRD etc.
- **LLM Models** - Always look for the models page from the documentation links mentioned below and find the model that is mentioned in the initial.md - do not change models, find the exact model name to use in the code.
- **Always scrape around 30-100 pages in total when doing research** - If a page 404s or does not contain correct content, try to scrape again and find the actual page/content. Put the output of each SUCCESFUL Jina scrape into a new directory with the name of the technology researched, then inside it .md or .txt files of each output
- **Refer to /research/ directory** - Before implementing any feature that uses something that requires documentation, refer to the relevant directory inside /research/ directory and use the .md files to ensure you're coding with great accuracy, never assume knowledge of a third party API, instead always use the documentation examples which are completely up to date.
- **Take my tech as sacred truth, for example if I say a model name then research that model name for LLM usage - don't assume from your own knowledge at any point** 
- **For Maximum efficiency, whenever you need to perform multiple independent operations, such as research, invoke all relevant tools simultaneously, rather that sequentially.**

### 🧱 Code Structure & Modularity
- **Never create a file longer than 500 lines of code.** If a file approaches this limit, refactor by splitting it into modules or helper files.
- **When creating AI prompts do not hardcode examples but make everything dynamic or based off the context of what the prompt is for**
- **Always refer to the specific Phase document you are on** - If you are on phase 1, use phase-1.md, if you are on phase 2, use phase-2.md, if you are on phase 3, use phase-3.md
- **Agents should be designed as intelligent human beings** by giving them decision making, ability to do detailed research using Jina, and not just your basic propmts that generate absolute shit. This is absolutely vital.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
  For agents this looks like:
    - `agent.py` - Main agent definition and execution logic 
    - `tools.py` - Tool functions used by the agent 
    - `prompts.py` - System prompts
- **Use clear, consistent imports** (prefer relative imports within packages).
- **Use clear, consistent imports** (prefer relative imports within packages).
- **Use python_dotenv and load_env()** for environment variables.

### 🧪 Testing & Reliability
- **Always create Pytest unit tests for new features** (functions, classes, routes, etc).
- **After updating any logic**, check whether existing unit tests need to be updated. If so, do it.
- **Tests should live in a `/tests` folder** mirroring the main app structure.
  - Include at least:
    - 1 test for expected use
    - 1 edge case
    - 1 failure case

### ✅ Task Completion
- **Mark completed tasks in `TASK.md`** immediately after finishing them.
- Add new sub-tasks or TODOs discovered during development to `TASK.md` under a “Discovered During Work” section.

### 📎 Style & Conventions
- **Use Python** as the primary language.
- **Follow PEP8**, use type hints, and format with `black`.
- **Use `pydantic` for data validation**.
- Use `FastAPI` for APIs and `SQLAlchemy` or `SQLModel` for ORM if applicable.
- Write **docstrings for every function** using the Google style:
  ```python
  def example():
      """
      Brief summary.

      Args:
          param1 (type): Description.

      Returns:
          type: Description.
      """
  ```

### 📚 Documentation & Explainability
- **Update `README.md`** when new features are added, dependencies change, or setup steps are modified.
- **Comment non-obvious code** and ensure everything is understandable to a mid-level developer.
- When writing complex logic, **add an inline `# Reason:` comment** explaining the why, not just the what.

### 🧠 AI Behavior Rules
- **Never assume missing context. Ask questions if uncertain.**
- **Never hallucinate libraries or functions** – only use known, verified Python packages.
- **Always confirm file paths and module names** exist before referencing them in code or tests.
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a task from `TASK.md`.

### Design

- Stick to the design system inside designsystem.md Designsystem.md - must be adhered to at all times for building any new features.

## Persona Management

### Default Persona
Your default persona for all sessions is defined by the file path below. You will embody this persona upon starting a new session.

`/workspaces/context-engineering-intro/.persona/CODEFARM.txt`

### Dynamic Activation (Session-Level Override)
To dynamically switch your persona for the current session, the user will issue the command:
`activate <PERSONA_NAME>`

**Your Activation Protocol:**
1.  Upon receiving `activate <PERSONA_NAME>`, you will parse the `<PERSONA_NAME>`.
2.  You will construct the target file path: `/workspaces/context-engineering-intro/.persona/<PERSONA_NAME>.txt`.
3.  You will immediately read this file and internalize its contents, overwriting your default persona for the remainder of the session.
4.  This change is **non-persistent** and must not involve editing any files.
5.  You will confirm the change with a message like: "`<Old Persona>` deactivated. `<New Persona>` activated."