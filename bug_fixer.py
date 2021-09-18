import ast
from ast import Call
from ff_util import get_assign_calls

'''
Changes:
Return list of lines to be altered and the string that needs to be replaced (or inserted)
[12, "time = tf.getTime(tensor)", INSERT]
'''

def pattern_a_bug_fixer_example(buggy_node, tree):
    """
    This is the fix of the bug, we just replace the "units" to 128

    """
    for node in ast.walk(tree):
        # we need to check if it is the buggy node found by the bug finder
        if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(buggy_node):
            node.args[0].value = 128


def pattern_b_decode_png_no_resize_bug(buggy_node, tree):
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


def pattern_c_decode_png_with_resize_bug(buggy_node, tree):
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
            if isinstance(node.args[1], ast.Name):
                Name_assign_nodes = get_assign_calls(node.args[1].id, tree)
                index = 0;
                for assignNode in Name_assign_nodes:
                    if assignNode.lineno > node.args[1].lineno:
                        break;
                    else:
                        index += 1
                variableArg = Name_assign_nodes[index-1]
                node.args = [node.args[0], variableArg.value.elts[0], variableArg.value.elts[1]]
            else:
                node.args = [node.args[0], node.args[1].elts[0], node.args[1].elts[1]]

    return tree


def pattern_d_merge_summary_bug(buggy_node, tree):
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


def pattern_e_merge_all_summaries_bug(buggy_node, tree):
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


def pattern_f_summary_writer_bug(buggy_node, tree):
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


def pattern_g_last_dense_binary_bug(buggy_node, tree):
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


def pattern_h_softmax_with_cross_entropy_bug(crossEntityObject, tree):
    """
    Checks for instances where tf.nn.softmax() is used in conjunction with cross_entropy(). In this case it is better
    to use the combined method tf.nn.softmax_cross_entropy_with_logits() as it covers numerically unstable corner cases
    in the mathematically right way.

    sm = tf.nn.softmax(x)         ------>         sm_ce = tf.nn.softmax_cross_entropy_with_logits()
    ce = tf.reduce_sum(sm)
    """
    # Note does not look into softmax name = and assumes name = None
    for node in ast.walk(tree):
        # we need to check if it is the buggy node found by the bug finder
        if isinstance(node, ast.Assign) and ast.dump(node) == ast.dump(crossEntityObject.cross_entropy_node):
            newNodeValue = ast.Call(ast.Name('tf.nn.softmax_cross_entropy_with_Logits', ast.Load()), [crossEntityObject.logits, crossEntityObject.nameLabel_var],[])
            ast.copy_location(newNodeValue, node.value)
            node.value = newNodeValue
            print("I am here")

    return tree


def pattern_i_tffunction_with_for_loop(buggy_node, tree):
    """
    Check for instances where @tf.function is used and the function is called in a for loop. In this case that the
    parameter needs to be of Tensor type. This can be achieved by either using tf.range and tf.cast together, or
    by using tf.convert_to_tensor(). In this case we will use tf.range and tf.cast as that provides faster iterations.

    for step in range(100):         ------->            for step in tf.range(100):
    my_func(step)                                           step = tf.cast(step, tf.int64)
                                                            my_func(step)
    """

    # we need to record where the for statement is located to insert the step function afterwards

    flag = False
    for node in ast.walk(tree):
        # we need to check if it is the buggy node found by the bug finder
        if isinstance(node, ast.For) and ast.dump(buggy_node) == ast.dump(node):
            stepVariableId = buggy_node.body[0].value.args[0].id
            node.iter.func.id = "tf.range"


            nameNode = ast.Name(stepVariableId, ast.Store())
            # callNode = ast.Call(ast.Name('tf.cast', ast.Load()), [ast.Num(0)], [])
            arg1 = ast.Name(stepVariableId, ast.Load())
            arg2 = ast.Name('tf.int64', ast.Load())
            callNode = ast.Call(ast.Name('tf.cast', ast.Load()), [arg1, arg2], [])
            assignNode = ast.Assign([nameNode], callNode)
            exprNode = ast.Expr(assignNode)
            node.body.insert(0, exprNode)

    return tree
