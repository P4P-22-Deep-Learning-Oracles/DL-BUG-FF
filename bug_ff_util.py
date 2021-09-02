import ast
from collections import deque
import bug_finder_patterns
import bug_fixer
import os

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


def file_iterator(file):
    for filename in os.listdir(file):
        if os.path.isdir(file + "/" + filename):
            file_iterator(file + "/" + filename)
            continue
        if filename.endswith(".py"):
            tree = load_source_code(file + "/" + filename)
            print("==============================")
            print("ORIGINAL CODE")
            print("==============================")
            print(ast.unparse(tree))
            bug_list = bug_finder(tree)
            bug_fixer_func(tree, bug_list)



def bug_finder(tree):
    print("==============================")
    print("DL-BUG_FINDER")
    print("==============================")
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

    return bugList


def bug_fixer_func(tree, bugList):
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
                if bugList[patternCount] is None or len(bugList[patternCount]) == 0:
                    patternCount += 1
                    continue

                for j in bugList[patternCount]:
                    tree = pattern(j, tree)
                patternCount += 1
        print()
        print("the fixed version of the program is")
        print(ast.unparse(tree))
    else:
        print("No known bugs found")


if __name__ == '__main__':
    print("==============================")
    print("INPUT")
    print("==============================")
    while True:
        filename = input("Please specify the directory location of the project code:")
        # Check directory exists
        if os.path.isdir(filename):
            break
        print("Oops, sorry that file does not exist or isn't a directory! Try again...")

    # Iterate through all .py files in directory and sub-directories
    file_iterator(filename)




