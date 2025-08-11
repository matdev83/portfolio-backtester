import argparse
import ast
import os
from collections import defaultdict


def analyze_directory(directory, output_file=None):
    """
    Analyzes a directory for class dependencies and SOLID principles violations.
    """
    print(f"Analyzing directory: {directory}")

    all_dependencies = defaultdict(set)
    all_classes = {}

    files_to_analyze = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                files_to_analyze.append(os.path.join(root, file))

    for file_path in files_to_analyze:
        print(f"Analyzing file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            try:
                tree = ast.parse(content)
                analyzer = SOLIDAnalyzer(file_path)
                analyzer.visit(tree)
                analyzer.report()
                for class_name, deps in analyzer.dependencies.items():
                    all_dependencies[class_name].update(deps)
                all_classes.update(analyzer.classes)

            except SyntaxError as e:
                print(f"Could not parse {file_path}: {e}")

    if output_file:
        generate_mermaid_graph(all_dependencies, all_classes, output_file)


def generate_mermaid_graph(dependencies, classes, output_file):
    """
    Generates a Mermaid class diagram and saves it to a file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("classDiagram\n")
        for class_name in classes:
            f.write(f"    class {class_name}\n")

        for class_name, deps in dependencies.items():
            for dep in deps:
                if dep in classes:
                    f.write(f"    {class_name} --|> {dep}\n")
    print(f"Mermaid graph saved to {output_file}")


class SOLIDAnalyzer(ast.NodeVisitor):
    """
    AST visitor to analyze a single Python file for SOLID principles.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.classes = {}
        self.dependencies = defaultdict(set)

    def visit_ClassDef(self, node):
        self.classes[node.name] = {
            "methods": [],
            "size": 0,
            "cohesion": 0,
            "init_dependencies": set(),
        }

        self.classes[node.name]["size"] = node.end_lineno - node.lineno

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self.classes[node.name]["methods"].append(item.name)
                if item.name == "__init__":
                    for sub_node in ast.walk(item):
                        if isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Name):
                            self.classes[node.name]["init_dependencies"].add(sub_node.func.id)

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        current_class = None
        for class_name, data in self.classes.items():
            if node.name in data["methods"]:
                current_class = class_name
                break

        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Call):
                if isinstance(sub_node.func, ast.Name):
                    if current_class:
                        self.dependencies[current_class].add(sub_node.func.id)
                    else:
                        self.dependencies[node.name].add(sub_node.func.id)

            # OCP Check
            if isinstance(sub_node, ast.If):
                if (
                    isinstance(sub_node.test, ast.Call)
                    and isinstance(sub_node.test.func, ast.Name)
                    and sub_node.test.func.id == "isinstance"
                ):
                    print(
                        f"  - OCP Violation?: 'isinstance' check in function '{node.name}'. Consider using polymorphism."
                    )

        self.generic_visit(node)

    def report(self):
        if not self.classes:
            return

        print(f"--- Report for {self.file_path} ---")
        for class_name, data in self.classes.items():
            print(f"Class: {class_name}")
            print(f"  - Methods: {', '.join(data['methods'])}")
            print(f"  - Size (lines): {data['size']}")

            if data["size"] > 100:
                print("  - SRP Violation?: Class is large, may have too many responsibilities.")

            # DIP Check
            if data["init_dependencies"]:
                print(
                    f"  - DIP Violation?: Class '{class_name}' depends on concrete classes in __init__: {', '.join(data['init_dependencies'])}"
                )

        if self.dependencies:
            print("\nDependencies:")
            for key, value in self.dependencies.items():
                print(f"  - {key} depends on: {', '.join(value)}")
        print("--- End of Report ---")


def main():
    parser = argparse.ArgumentParser(description="Analyze Python code for SOLID principles.")
    parser.add_argument("directory", help="The directory to analyze.")
    parser.add_argument("--output-file", help="The file to save the Mermaid graph to.")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory.")
        return

    analyze_directory(args.directory, args.output_file)


if __name__ == "__main__":
    main()
