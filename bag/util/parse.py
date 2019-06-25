# -*- coding: utf-8 -*-

"""This module defines parsing utility methods.
"""

import ast


class ExprVarScanner(ast.NodeVisitor):
    """
    This node visitor collects all variable names found in the
    AST, and excludes names of functions.  Variables having
    dotted names are not supported.
    """
    def __init__(self):
        self.varnames = set()

    # noinspection PyPep8Naming
    def visit_Name(self, node):
        self.varnames.add(node.id)

    # noinspection PyPep8Naming
    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            self.visit(node.func)
        for arg in node.args:
            self.visit(arg)

    # noinspection PyPep8Naming
    def visit_Attribute(self, node):
        # ignore attributes
        pass


def get_variables(expr):
    """Parses the given Python expression and return a list of all variables.

    Parameters
    ----------
    expr : str
        An expression string that we want to parse for variable names.

    Returns
    -------
    var_list : list[str]
        Names of variables from the given expression.
    """
    root = ast.parse(expr, mode='exec')
    scanner = ExprVarScanner()
    scanner.visit(root)
    return list(scanner.varnames)
