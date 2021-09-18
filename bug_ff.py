import ast
import astor
import bug_finder_patterns
import bug_fixer
import os

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
            print(astor.to_source(tree))
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
        fixList = []
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
        print(astor.to_source(tree))
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




