import ast
import unittest
from pathlib import Path


class BgeExportContractTests(unittest.TestCase):
    def test_dynamic_axes_opset17_uses_explicit_legacy_exporter(self):
        source = Path(__file__).with_name("export_bge.py")
        calls = [node for node in ast.walk(ast.parse(source.read_text()))
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and ast.unparse(node.func) == "torch.onnx.export"]
        self.assertEqual(len(calls), 1)
        keywords = {kw.arg: kw.value for kw in calls[0].keywords}
        self.assertEqual(ast.literal_eval(keywords["opset_version"]), 17)
        self.assertIn("dynamic_axes", keywords)
        self.assertIn("dynamo", keywords)
        self.assertIs(ast.literal_eval(keywords["dynamo"]), False)


if __name__ == "__main__":
    unittest.main()
