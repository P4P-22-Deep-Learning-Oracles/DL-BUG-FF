import ast
from bug_finder_util import get_func_calls


def bug_finder_pattern_example(tree):
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


def decode_png_no_resize_bug_pattern(tree):
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


def decode_png_with_resize_bug_pattern(tree):
    """
    There is one problem where decode_image() cannot be used in conjunction
    with tf.image.resize() or tf.image.resize_images(), therefore if either of those calls are also
    present we will make the appropriate change to those calls as well.
    """
    decode_jpeg_with_resize_list = []
    resize_func_calls, resize_arguments = get_func_calls('resize', tree)
    for i in range(len(resize_func_calls)):
        decode_jpeg_with_resize_list.append(resize_func_calls[i])

    return decode_jpeg_with_resize_list
