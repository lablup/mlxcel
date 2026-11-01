# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Negative coverage for scripts/ci/check_native_env_writes.py (#2803)."""

import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "check_native_env_writes.py"

_spec = importlib.util.spec_from_file_location("check_native_env_writes", SCRIPT)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)

DIRECT = 'pub fn f() { unsafe { std::env::set_var("MLX_X", "1") }; }\n'


class CheckNativeEnvWrites(unittest.TestCase):
    def findings(self, files: dict[str, str]) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for relative, text in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            return checker.scan(root)

    def test_the_repository_is_clean(self):
        self.assertEqual(checker.scan(ROOT), [])

    def test_direct_writes_are_refused_in_every_spelling(self):
        for line in [
            DIRECT,
            'pub fn f() { unsafe { std::env::remove_var("MLX_X") }; }\n',
            'use std::env;\npub fn f(n: &str) { unsafe { env::set_var(n, "1") }; }\n',
            'use std::env::set_var;\npub fn f() { unsafe { set_var("MLXCEL_X", "1") }; }\n',
            "pub fn f() { unsafe { std::env::set_var(\n    NAME,\n    value,\n) }; }\n",
            "pub fn f() { unsafe { _wputenv_s(a, b) }; }\n",
            "pub fn f() { unsafe { SetEnvironmentVariableW(a, b) }; }\n",
        ]:
            with self.subTest(line=line):
                self.assertTrue(self.findings({"src/lib.rs": line}))

    def test_a_write_after_a_test_module_is_still_production(self):
        text = "#[cfg(test)]\nmod tests {\n    fn t() {}\n}\n" + DIRECT
        found = self.findings({"src/lib.rs": text})
        self.assertEqual(len(found), 1)
        self.assertIn("src/lib.rs:5:", found[0])

    def test_a_file_named_like_a_test_is_production_unless_declared(self):
        self.assertEqual(len(self.findings({"src/foo_tests.rs": DIRECT})), 1)

    def test_a_cfg_not_test_item_is_production(self):
        text = '#[cfg(not(test))]\n' + DIRECT
        self.assertEqual(len(self.findings({"src/lib.rs": text})), 1)

    def test_the_helper_and_its_callers_pass(self):
        self.assertEqual(
            self.findings(
                {
                    "src/lib/mlxcel-core/src/native_env.rs": DIRECT,
                    "src/main.rs": 'fn f() { unsafe { set_native_var("MLX_X", "1") }; }\n',
                }
            ),
            [],
        )

    def test_comments_and_strings_do_not_count(self):
        text = (
            "// std::env::set_var is not used here\n"
            "/* nor set_var( /* nested */ here */\n"
            'const A: &str = "std::env::set_var(x)";\n'
            'const B: &str = r#"remove_var("x")"#;\n'
            "const C: char = '\"';\n"
            "fn f<'a>(x: &'a str) -> &'a str { x }\n"
        )
        self.assertEqual(self.findings({"src/lib.rs": text}), [])

    def test_test_code_is_exempt(self):
        cases = {
            "inline module": {
                "src/lib.rs": "#[cfg(test)]\nmod tests {\n    " + DIRECT + "}\n"
            },
            "attribute with more items": {
                "src/lib.rs": "#[cfg(test)]\n#[ctor::ctor(unsafe)]\n" + DIRECT
            },
            "cfg all": {"src/lib.rs": '#[cfg(all(test, feature = "cuda"))]\n' + DIRECT},
            "path attribute": {
                "src/lib.rs": '#[cfg(test)]\n#[path = "foo_tests.rs"]\nmod foo_tests;\n',
                "src/foo_tests.rs": DIRECT,
            },
            "plain declaration": {
                "src/lib.rs": "#[cfg(test)]\nmod helpers;\n",
                "src/helpers.rs": DIRECT,
            },
            "directory module and child": {
                "src/lib.rs": "#[cfg(test)]\nmod helpers;\n",
                "src/helpers/mod.rs": "mod child;\n",
                "src/helpers/child.rs": DIRECT,
            },
            "non-root parent": {
                "src/a.rs": "#[cfg(test)]\nmod t;\n",
                "src/a/t.rs": DIRECT,
            },
        }
        for name, files in cases.items():
            with self.subTest(name):
                self.assertEqual(self.findings(files), [])


if __name__ == "__main__":
    unittest.main()
