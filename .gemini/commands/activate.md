# Activate Persona

## Description
This command activates a new persona for the AI agent for the current session. It takes one argument: the name of the persona to activate.

## Arguments
- `$ARG_1` or `$ARGUMENTS`: The name of the persona to activate (e.g., `NOVA`, `ARCHITECT`).

## Activation Protocol

Upon receiving the `activate` command with a persona name, you MUST follow these steps precisely:

1.  **Parse Persona Name:** Identify the persona name from the user's command. This will be the first and only argument.

2.  **Construct File Path:** Create the absolute file path for the persona file using the following template:
    `/workspaces/context-engineering-intro/.persona/<PERSONA_NAME>.txt`
    Replace `<PERSONA_NAME>` with the name you parsed in step 1. For example, if the user runs `activate NOVA`, the path is `/workspaces/context-engineering-intro/.persona/NOVA.txt`.

3.  **Read the Persona File:** Use the `read_file` tool to read the contents of the file at the path you just constructed.

4.  **Internalize and Embody:** Read and internalize the persona description from the file. You must adopt the specified personality, traits, and response style for the remainder of the session.

5.  **Confirm Activation:** After successfully reading and internalizing the file, output a confirmation message to the user. The message should be in the style of the *newly activated persona*. For example: "`<Old Persona>` deactivated. `<New Persona>` activated." or a creative equivalent that fits the new persona.

This protocol is not optional. You must execute the `read_file` tool to fulfill the command.
