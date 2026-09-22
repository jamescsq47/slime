"""Allow exactly the approved system-prompt example against the 27B reference."""

PROMPT_VERSION = 'openai_tools_shell_example_v1'
EXAMPLE = b'''

The only tool name is shell; its required argument is command. Commands such as
cat, grep, and python belong inside command, not in the tool name. Example:
<tool_call>
<function=shell>
<parameter=command>
cat /testbed/example.py
</parameter>
</function>
</tool_call>
When calling a tool, replace only the command text, keeping this outer structure.'''


def matches_reference(old, current, filename, version=''):
    if not version:
        return old == current
    if version != PROMPT_VERSION:
        return False
    if filename != 'harness.py':
        return old == current
    return current.count(EXAMPLE) == 1 and current.replace(EXAMPLE, b'', 1) == old
