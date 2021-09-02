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

    for node in ast.walk(tree):
        # we need to check if it is the buggy node found by the bug finder

        if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(buggy_node):
            node.func.value.attr = "image"
            node.func.attr = "resize_image_with_crop_or_pad"
            node.args = [node.args[0], node.args[1].elts[0], node.args[1].elts[1]]

    return tree


def pattern_merge_summary_bug(buggy_node, tree):
    """
    As Tensorflow changes through versions, many API calls become deprecated. This
    is an example of an API call that is no longer supported with the update to
    Tensorflow 1.0.

    tf.merge_all_summary should now be tf.summary.merge_all
    """
    print("Fixing this API misuse where merge_summary ")

    for node in ast.walk(tree):
        # we need to check if it is the buggy node found by the bug finder
        if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(buggy_node):
            node.func.attr = "summary.merge"

    return tree


def pattern_merge_all_summaries_bug(buggy_node, tree):
    """
    As Tensorflow changes through versions, many API calls become deprecated. This
    is an example of an API call that is no longer supported with the update to
    Tensorflow 1.0.

    tf.merge_all_summaries should now be tf.summary.merge_all
    """
    print("Fixing this API misuse where merge_all_summaries is called")

    for node in ast.walk(tree):
        # we need to check if it is the buggy node found by the bug finder
        if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(buggy_node):
            node.func.attr = "summary.merge_all"

    return tree


def pattern_summary_writer_bug(buggy_node, tree):
    """
    As Tensorflow changes through versions, many API calls become deprecated. This
    is an example of an API call that is no longer supported with the update to
    Tensorflow 1.0.

    tf.train.SummaryWriter         --------->             tf.summary.FileWriter
    """
    for node in ast.walk(tree):
        # we need to check if it is the buggy node found by the bug finder
        if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(buggy_node):
            node.func.value.attr = "summary"
            node.func.attr = "FileWriter"

    return tree


def pattern_last_dense_binary_bug(buggy_node, tree):
    """
    As Tensorflow changes through versions, many API calls become deprecated. This
    is an example of an API call that is no longer supported with the update to
    Tensorflow 1.0.

    tf.train.SummaryWriter         --------->             tf.summary.FileWriter
    """
    for node in ast.walk(tree):
        # we need to check if it is the buggy node found by the bug finder
        if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(buggy_node):
            node.args[0].value = 2

    return tree
