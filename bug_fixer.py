import ast

def bug_fixer_pattern_example(buggy_node, tree):
    """
    This is the fix of the bug, we just replace the "units" to 128

    """
    for node in ast.walk(tree):
        # we need to check if it is the buggy node found by the bug finder
        if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(buggy_node):
            node.args[0].value = 128