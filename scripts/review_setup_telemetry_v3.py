"""Read-only AST and identity audit. Does not import or execute preparation tools."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

EXPECTED_STAGES = [
    "legacy_code",
    "runtime_code",
    "interpreter_environment",
    "distributions",
    "native_binaries",
    "input_inventory",
    "storage",
]


def function(tree, name):
    return copy.deepcopy(
        next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    )


def same(a, b):
    return ast.dump(a, include_attributes=False) == ast.dump(
        b, include_attributes=False
    )


def is_call(node, owner, name):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
        and node.func.attr == name
    )


class RemoveTelemetry(ast.NodeTransformer):
    def visit_Expr(self, node):
        if is_call(node.value, "telemetry", "completed"):
            if node.value.args or node.value.keywords:
                raise ValueError("unexpected progress arguments")
            return None
        return self.generic_visit(node)

    def visit_With(self, node):
        ctx = node.items[0].context_expr
        if is_call(ctx, "telemetry", "stage"):
            if len(node.items) != 1 or len(ctx.args) != 1 or ctx.keywords:
                raise ValueError("stage contract")
            name = ctx.args[0].value
            if name not in EXPECTED_STAGES:
                raise ValueError("unexpected stage")
            if name == "legacy_code":
                return ast.parse("verify_legacy_files()").body[0]
            result = []
            for statement in node.body:
                item = self.visit(statement)
                if item is not None:
                    result.append(item)
            return result
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == "file_sha":
            return None
        if node.name == "preflight":
            node.args.args = node.args.args[:1]
        return self.generic_visit(node)


class RemovePolicyQualifier(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "policy":
            return ast.Name(id=node.attr, ctx=ast.Load())
        return self.generic_visit(node)


def audit(root):
    paths = {
        "v2": "scripts/run_readonly_budget_v2_trial.py",
        "v3": "scripts/preflight_setup_telemetry_v3.py",
        "policy": "diagnostic_storage/validation_budget_policy_v2.py",
    }
    trees = {k: ast.parse((root / p).read_text()) for k, p in paths.items()}
    old = function(trees["v2"], "preflight")
    new = function(trees["v3"], "preflight")
    stages = [
        n.items[0].context_expr.args[0].value
        for n in new.body
        if isinstance(n, ast.With)
        and is_call(n.items[0].context_expr, "telemetry", "stage")
    ]
    if stages != EXPECTED_STAGES:
        raise ValueError("stage order mismatch")
    # Check the helper removed by the normalization: delegation only, no decision.
    delegate = next(
        n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "file_sha"
    )
    expected = ast.parse(
        "def file_sha(path):\n return hash_file(path, telemetry)"
    ).body[0]
    if not same(delegate, expected):
        raise ValueError("hash delegate changed")
    # Check legacy expansion independently before replacing the stage wrapper.
    legacy = function(trees["policy"], "verify_legacy_files")
    legacy.body = legacy.body[:-1]  # unused original return dict, not a gate
    block = next(
        n
        for n in new.body
        if isinstance(n, ast.With)
        and is_call(n.items[0].context_expr, "telemetry", "stage")
    )
    expanded = ast.FunctionDef(
        name="verify_legacy_files",
        args=copy.deepcopy(legacy.args),
        body=copy.deepcopy(block.body),
        decorator_list=[],
    )
    expanded = RemovePolicyQualifier().visit(RemoveTelemetry().visit(expanded))
    if not same(legacy, expanded):
        raise ValueError("legacy checks changed")
    normalized = RemoveTelemetry().visit(new)
    if not same(old, normalized):
        raise ValueError("identity checks/order/deadline locations differ")
    # Check complete hashing algorithm after removing ONLY byte-counter statements.
    hash_new = function(trees["v3"], "hash_file")
    hash_old = function(trees["policy"], "file_sha")
    hash_new.name = "file_sha"
    hash_new.args.args = hash_new.args.args[:1]

    class CountOnly(ast.NodeTransformer):
        def visit_Assign(self, n):
            if same(n, ast.parse("n = 0").body[0]):
                return None
            return self.generic_visit(n)

        def visit_AugAssign(self, n):
            if same(n, ast.parse("n += len(chunk)").body[0]):
                return None
            return self.generic_visit(n)

        def visit_Expr(self, n):
            if same(n, ast.parse("telemetry.hashed(n)").body[0]):
                return None
            return self.generic_visit(n)

    hash_new = CountOnly().visit(hash_new)
    if not same(hash_old, hash_new):
        raise ValueError("hash read algorithm changed")
    calls = [n for n in ast.walk(trees["v3"]) if isinstance(n, ast.Call)]
    if any(
        isinstance(n.func, ast.Name)
        and n.func.id
        in ("run_validation", "compare_captures", "validate_capture", "authorize")
        for n in calls
    ):
        raise ValueError("execution/authorization dispatch found")
    return {
        "status": "static_structure_checks_pass",
        "source_files_sha256": {
            p: hashlib.sha256((root / p).read_bytes()).hexdigest()
            for p in paths.values()
        },
        "stage_order": stages,
        "preflight_identity_AST_after_declared_telemetry_removal_equal": True,
        "legacy_expansion_AST_equal": True,
        "single_hash_algorithm_AST_equal": True,
        "library_or_launch_dispatch_found": False,
        "server_calls": 0,
        "scope": "structural checks plus explicit source review; not a runtime side-effect proof, independent third-party signature or target acceptance",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.root)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
