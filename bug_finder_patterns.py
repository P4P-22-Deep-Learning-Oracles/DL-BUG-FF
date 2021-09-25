import ast
from ff_util import get_func_calls, get_assign_nodes_using_func


def pattern_a_bug_finder_example(tree):
    """
    this is the bug finder function specific for the pattern that we want to find, you need to implement one function
    like this for each pattern

    As an example I invented the pattern that checks if activation='relu' and the "units" are not a multiple of 32

    For example, this

    tf.keras.layers.Dense(12, activation='relu')

    is an API misuses because the units are 12 and not 32, 64, 128, .....

    Note that we stop when we find the first API misuse, but in general that could be more than one misuse in the
    same Tensorflow program.

    """
    print("Searching for the API misuse that assign units to keras.layers.Dense that are not multiple of 32......  ")
    func_calls, arguments = get_func_calls('keras.layers.Dense', tree)
    # In case you want to print and see how it looks like
    # astpretty.pprint(func_calls[0])
    for i in range(len(func_calls)):
        args = arguments[i]
        if len(args) >= 2:  # we check if there are at least two arguments/keywords
            if isinstance(args[0], ast.Constant):  # the first should be a constants
                if int(args[0].value) % 32 != 0:  # if is not a multiple of 32 I check if the activation is relu
                    for arg in args:
                        if isinstance(arg, ast.keyword):
                            if arg.arg == 'activation' and arg.value.value == "relu":
                                print("API misuse found!")
                                print(ast.unparse(func_calls[i]), " ", int(args[0].value), "should be a multiple of 32")
                                # we return the first function call that represent this specific API misuse
                                return func_calls[i]
    print("did not find any API misuses!")
    return None


def pattern_b_decode_png_no_resize_bug(tree):
    """
    This pattern deals with the common bug where tf.image.decode_jpeg() or tf.io.decode_jpeg()
    are used to decode files of type .png. This will not throw an error but will potentially cause
    issues when the model is trying to understand the decoded image.

    To solve this issue we will be checking for instances of decode_jpeg() and trying
    to replace them with decode_image() instead, which generally works for both .jpeg
    and .png files.
    """
    print("Searching for API misuse where decode_jpeg is called for files of type PNG")
    decode_jpeg_list = []
    func_calls, arguments = get_func_calls('decode_jpeg', tree)
    for i in range(len(func_calls)):
        decode_jpeg_list.append(func_calls[i])
    return decode_jpeg_list


def pattern_c_decode_png_with_resize_bug(tree):
    """
    There is one problem where decode_image() cannot be used in conjunction
    with tf.image.resize() or tf.image.resize_images(), therefore if either of those calls are also
    present we will make the appropriate change to those calls as well.
    """
    decode_jpeg_with_resize_list = []
    jpeg_func_calls, jpeg_arguments = get_func_calls('decode_jpeg', tree)
    if len(jpeg_func_calls) == 0:
        return None

    resize_func_calls, resize_arguments = get_func_calls('resize', tree)
    for i in range(len(resize_func_calls)):
        decode_jpeg_with_resize_list.append(resize_func_calls[i])

    return decode_jpeg_with_resize_list


def pattern_d_merge_summary_bug(tree):
    """
    As Tensorflow changes through versions, many API calls become deprecated. This
    is an example of an API call that is no longer supported with the update to
    Tensorflow 1.0.

    tf.merge_summary should now be tf.summary.merge
    """
    merge_summary_list = []
    merge_func_calls, merge_all_arguments = get_func_calls('merge_summary', tree)
    for i in range(len(merge_func_calls)):
        merge_summary_list.append(merge_func_calls[i])

    return merge_summary_list


def pattern_e_merge_all_summaries_bug(tree):
    """
    As Tensorflow changes through versions, many API calls become deprecated. This
    is an example of an API call that is no longer supported with the update to
    Tensorflow 1.0.

    tf.merge_all_summaries should now be tf.summary.merge_all
    """
    merge_all_summaries_list = []
    merge_all_func_calls, merge_all_arguments = get_func_calls('merge_all_summaries', tree)
    for i in range(len(merge_all_func_calls)):
        merge_all_summaries_list.append(merge_all_func_calls[i])

    return merge_all_summaries_list


def pattern_f_summary_writer_bug(tree):
    """
    As Tensorflow changes through versions, many API calls become deprecated. This
    is an example of an API call that is no longer supported with the update to
    Tensorflow 1.0.

    tf.train.SummaryWriter         --------->             tf.summary.FileWriter
    """
    instance_list = []
    summary_writer_func_calls, summary_writer_arguments = get_func_calls('SummaryWriter', tree)
    for i in range(len(summary_writer_func_calls)):
        instance_list.append(summary_writer_func_calls[i])

    return instance_list


def pattern_g_last_dense_binary_bug(tree):
    """
    Checks that the final layer of Dense() is 2 if the class mode is set to binary. A common bug can be making this 3
    which wont work with binary.
    Tensorflow 1.0

    Dense(3, activation='softmax')      ------->           Dense(2, activation='softmax')
    """
    dense_bug_list = []
    last_dense_binary_func_calls, last_dense_binary_arguments = get_func_calls('Dense', tree)
    flow_from_directory_func_calls, flow_from_directory_arguments = get_func_calls('flow_from_directory', tree)

    for i in range(len(flow_from_directory_func_calls)):
        for arg in flow_from_directory_arguments[i]:
            if isinstance(arg, ast.keyword):
                if arg.arg == "class_mode":
                    if arg.value.value == "binary":
                        final_call_args = last_dense_binary_arguments[len(last_dense_binary_func_calls) - 1]
                        if isinstance(final_call_args[0], ast.Constant) and final_call_args[0].value != 2:
                            dense_bug_list.append(last_dense_binary_func_calls[len(last_dense_binary_func_calls) - 1])
                            return dense_bug_list
    return dense_bug_list


def pattern_h_softmax_with_cross_entropy_bug(tree):
    """
    Checks for instances where tf.nn.softmax() is used in conjunction with cross_entropy(). In this case it is better
    to use the combined method tf.nn.softmax_cross_entropy_with_logits() as it covers numerically unstable corner cases
    in the mathematically right way.

    sm = tf.nn.softmax(x)         ------>         sm_ce = tf.nn.softmax_cross_entropy_with_logits()
    ce = tf.reduce_sum(sm)
    """
    class CrossEntropyObject:
        def __init__(self, logits, nameLabel_var, entropy_node):
            self.logits = logits
            self.nameLabel_var = nameLabel_var
            self.cross_entropy_node = entropy_node

    softmax_with_entropy_list = []
    softmax_assign_calls = get_assign_nodes_using_func('softmax', tree)
    cross_entropy_assign_calls = get_assign_nodes_using_func('reduce_sum', tree)

    # Check that there are instances of both subparts of the pattern
    if len(softmax_assign_calls) == 0 or len(cross_entropy_assign_calls) == 0:
        return softmax_with_entropy_list

    for softmax_node in softmax_assign_calls:
        for cross_entropy_node in cross_entropy_assign_calls:
            if isinstance(cross_entropy_node.value, ast.UnaryOp):
                if isinstance(cross_entropy_node.value.operand.args[0], ast.BinOp):
                    if isinstance(cross_entropy_node.value.operand.args[0].left, ast.Call):
                        if cross_entropy_node.value.operand.args[0].left.args[0].id == softmax_node.targets[0].id:
                            bug_location = CrossEntropyObject(softmax_node.value.args[0], cross_entropy_node.value.operand.args[0].right, cross_entropy_node)
                            softmax_with_entropy_list.append(bug_location)
                    if isinstance(cross_entropy_node.value.operand.args[0].right, ast.Call):
                        if cross_entropy_node.value.operand.args[0].right.args[0].id == softmax_node.targets[0].id:
                            bug_location = CrossEntropyObject(softmax_node.value.args[0], cross_entropy_node.value.operand.args[0].left, cross_entropy_node)
                            softmax_with_entropy_list.append(bug_location)

    return softmax_with_entropy_list


def pattern_i_tffunction_with_for_loop(tree):
    """
    Check for instances where @tf.function is used and the function is called in a for loop. In this case that the
    parameter needs to be of Tensor type. This can be achieved by either using tf.range and tf.cast together, or
    by using tf.convert_to_tensor(). In this case we will use tf.range and tf.cast as that provides faster iterations.

    for step in range(100):         ------->            for step in tf.range(100):
    my_func(step)                                           step = tf.cast(step, tf.int64)
                                                            my_func(step)
    """
    tffunction_pattern_list = []
    tffunction_function_names = []

    # Search for all function definitions that have the decorator @tf.function
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Attribute):
                    if decorator.attr == 'function' and decorator.value.id == 'tf':
                        tffunction_function_names.append(node.name)

    if len(tffunction_function_names) == 0:
        return tffunction_pattern_list

    for func_name in tffunction_function_names:
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call) and node.iter.func.id == 'range':
                    for node_body in node.body:
                        if isinstance(node_body, ast.Expr):
                            if isinstance(node_body.value.func, ast.Name):
                                if node_body.value.func.id == func_name:
                                    for args in node_body.value.args:
                                        if isinstance(args, ast.Name) and args.id == node.target.id:
                                            tffunction_pattern_list.append(node)

    return tffunction_pattern_list


def pattern_j_historgram_summary_bug(tree):
    """
    As Tensorflow changes through versions, many API calls become deprecated. This
    is an example of an API call that is no longer supported with the update to
    Tensorflow 1.0.

    tf.histogram_summary        --------->             tf.summary.histogram
    """
    instance_list = []
    depr_func_calls, summary_writer_arguments = get_func_calls('histogram_summary', tree)
    for i in range(len(depr_func_calls)):
        instance_list.append(depr_func_calls[i])

    return instance_list


def pattern_k_scalar_summary_bug(tree):
    """
    As Tensorflow changes through versions, many API calls become deprecated. This
    is an example of an API call that is no longer supported with the update to
    Tensorflow 1.0.

    tf.scalar_summary        --------->             tf.summary.scalar
    """
    instance_list = []
    depr_func_calls, summary_writer_arguments = get_func_calls('scalar_summary', tree)
    for i in range(len(depr_func_calls)):
        instance_list.append(depr_func_calls[i])

    return instance_list






