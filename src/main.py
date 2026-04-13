import ast
from os import walk


def main():

    with open('src/main.py', 'r') as f:
        code: str = f.read()
    for (root, dirs, files) in walk('vllm-0.10.1/vllm-0.10.1/'):
        print("Directory path: ", root)
        print("Directory Names: ", dirs)
        print("Files Names: ", files)
    parser = ast.parse(code)
    # for node in ast.walk(parser):
    #     if isinstance(node, ast.FunctionDef):
    #         print(node.name)
    #     if isinstance(node, ast.Call):
    #         print(node.args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
