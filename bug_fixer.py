import ast


def pattern_bug_fixer_example(buggy_node, tree):
    """
    This is the fix of the bug, we just replace the "units" to 128

    """
    for node in ast.walk(tree):
        # we need to check if it is the buggy node found by the bug finder
        if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(buggy_node):
            node.args[0].value = 128


def pattern_decode_png_no_resize_bug(buggy_node, tree):
    """
    This pattern deals with the common bug where tf.image.decode_jpeg() or tf.io.decode_jpeg()
    are used to decode files of type .png. This will not throw an error but will potentially cause
    issues when the model is trying to understand the decoded image.

    To solve this issue we will be checking for instances of decode_jpeg() and trying
    to replace them with decode_image() instead, which generally works for both .jpeg
    and .png files.
    """
    print("Fixing this API misuse where decode_jpeg is called for files of type PNG")

    for node in ast.walk(tree):
        # we need to check if it is the buggy node found by the bug finder
        if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(buggy_node):
            node.func.attr = "decode_image"

    return tree


def pattern_decode_png_with_resize_bug(buggy_node, tree):
    """
    There is one problem where decode_image() cannot be used in conjunction
    with tf.image.resize() or tf.image.resize_images(), therefore if either of those calls are also
    present we will make the appropriate change to those calls as well.
    """
    print("Fixing decode_png resize error")
