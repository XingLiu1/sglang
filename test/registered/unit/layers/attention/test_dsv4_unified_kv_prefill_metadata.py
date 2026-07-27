"""Source-level contracts for DSV4 unified-KV prefill metadata."""

import ast
import unittest
from pathlib import Path

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:
    # Keep this AST contract runnable in source-only checkouts where the
    # SGLang Python package (and GPU dependencies such as Triton) is absent.
    CustomTestCase = unittest.TestCase

    def register_cpu_ci(**_kwargs):
        return None


register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[5]
_BACKEND_PATH = (
    _REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
    / "deepseek_v4_backend_hip_radix.py"
)


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one {name} definition, found {len(matches)}")
    return matches[0]


def _is_torch_repeat_interleave(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "repeat_interleave"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    )


class TestDsv4UnifiedKvPrefillMetadata(CustomTestCase):
    def test_repeat_interleave_uses_cpu_known_output_size_on_both_paths(self):
        tree = ast.parse(_BACKEND_PATH.read_text(encoding="utf-8"))
        function = _find_function(tree, "_attach_unified_kv_prefill_meta")
        branch = next(
            (
                node
                for node in function.body
                if isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "need_compress"
            ),
            None,
        )
        self.assertIsNotNone(branch, "need_compress branch is missing")

        for label, statements in (
            ("need_compress=True", branch.body),
            ("need_compress=False", branch.orelse),
        ):
            calls = [
                node
                for statement in statements
                for node in ast.walk(statement)
                if _is_torch_repeat_interleave(node)
            ]
            self.assertEqual(
                len(calls),
                1,
                f"{label} must contain exactly one torch.repeat_interleave call",
            )
            output_size = next(
                (
                    keyword.value
                    for keyword in calls[0].keywords
                    if keyword.arg == "output_size"
                ),
                None,
            )
            self.assertIsInstance(
                output_size,
                ast.Name,
                f"{label} must pass output_size=num_tokens",
            )
            self.assertEqual(
                output_size.id,
                "num_tokens",
                f"{label} must pass output_size=num_tokens",
            )


if __name__ == "__main__":
    unittest.main()
