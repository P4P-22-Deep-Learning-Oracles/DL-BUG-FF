# this is the library for ast analysis
import ast
from collections import deque


class FuncCallVisitor(ast.NodeVisitor):
    """
    This is the AST visitor that returns the fully qualified names of all function calls
    """

    def __init__(self):
        self._name = deque()

    @property
    def name(self):
        return '.'.join(self._name)

    @name.deleter
    def name(self):
        self._name.clear()

    def visit_Name(self, node):
        self._name.appendleft(node.id)

    def visit_Attribute(self, node):
        try:
            self._name.appendleft(node.attr)
            self._name.appendleft(node.value.id)
        except AttributeError:
            self.generic_visit(node)


def get_func_calls(target):
    """
    This function get all the function calls and arguments that matches a specific target
    for example 'keras.layers.Dense'

    it returns a list of all AST nodes that calls the specific targets with the list of arguments

    In Python, args is when you specify only the value, keywords is when you specify the argument name.

    For example, tf.keras.layers.Dense(12, activation='relu')

    there is on args (12)
    there is one keyword (activation='relu')
    """
    func_calls = []
    arguments = []
    for node in ast.walk(tree):  # this navigates all teh nodes in the AST
        if isinstance(node, ast.Call):
            call_visitor = FuncCallVisitor()
            call_visitor.visit(node.func)
            if call_visitor.name.endswith(target):  # check if the function call is matching the target
                func_calls.append(node)
                args = node.args
                keywords = node.keywords
                arguments.append(args + keywords)  # combines the args and keywords
    return func_calls, arguments


def bug_finder_pattern_example():
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
    func_calls, arguments = get_func_calls('keras.layers.Dense')
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




def bug_fixer_pattern_example(buggy_node):
    """
    This is the fix of the bug, we just replace the "units" to 128

    """
    for node in ast.walk(tree):
        # we need to check if it is the buggy node found by the bug finder
        if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(buggy_node):
            node.args[0].value = 128


if __name__ == '__main__':
    # this is just n example, we should load and write the Tensorflow programs from files

    # we assume this to be a misuse 12 should be replaced with 128 in
    # tf.keras.layers.Dense(12, activation='relu'),
    src = '''model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(12, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
    ])'''
    print("==============================")
    print("DL-BUG_FINDER")
    print("==============================")

    print("the Tensorflow program under analysis is")
    print(src)
    print()
    # parse the AST
    tree = ast.parse(src)
    # find the pattern
    buggy_node = bug_finder_pattern_example()
    if buggy_node is not None:
        print()
        print("==============================")
        print("DL-BUG_FIXER")
        print("==============================")
        # if I find it I fix it
        bug_fixer_pattern_example(buggy_node)
        print()
        print("the fixed version of the program is")
        print(ast.unparse(tree))
