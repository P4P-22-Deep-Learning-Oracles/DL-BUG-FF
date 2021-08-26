import ast
from collections import deque
import bug_finder_patterns
import bug_fixer


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


def get_func_calls(target, tree):
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


def load_source_code(filename):
    with open(filename, "r") as source:
        tree_ting = ast.parse(source.read())

    return tree_ting


if __name__ == '__main__':
    # this is just n example, we should load and write the Tensorflow programs from files

    # we assume this to be a misuse 12 should be replaced with 128 in
    # tf.keras.layers.Dense(12, activation='relu'),

    print("==============================")
    print("DL-BUG_FINDER")
    print("==============================")
    while True:
        try:
            filename = input("Please specify the file location of the source code:")
            # parse the AST
            tree = load_source_code(filename)
            break
        except FileNotFoundError:
            print("Oops, sorry that file does not exist! Try again...")

    print("Converting " + filename + " into AST")
    print()
    # find the pattern
    bugList = []
    # Search for everything in bug_finder_patterns and iterate through
    for i in dir(bug_finder_patterns):
        # Get the attributes
        pattern = getattr(bug_finder_patterns, i)
        # If it's a function then call that function
        if callable(pattern) and i.startswith('pattern'):
            # Store list of bugs and then append that list to the overall list
            patternBugs = pattern(tree)
            bugList.append(patternBugs)
    if bugList is not None:
        print()
        print("==============================")
        print("DL-BUG_FIXER")
        print("==============================")
        patternCount = 0
        for i in dir(bug_fixer):
            # Get the attributes
            pattern = getattr(bug_fixer, i)
            # If it's a function then call that function
            if callable(pattern) and i.startswith('pattern'):
                # Store list of bugs and then append that list to the overall list
                # if bugList is empty continue to the next element in bugList
                if bugList[patternCount] is None:
                    patternCount += 1
                    continue

                for j in bugList[patternCount]:
                    pattern(j, tree)
                patternCount += 1
        print()
        print("the fixed version of the program is")
        print(ast.unparse(tree))
    else:
        print("No known bugs found")

