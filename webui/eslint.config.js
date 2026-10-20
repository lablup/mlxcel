import js from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  js.configs.recommended,
  ...tseslint.configs.strict,
  {
    ignores: ['dist/**', '.playwright-dist/**', 'playwright-report/**', 'test-results/**', '../src/webui/assets/**', 'node_modules/**'],
  },
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  {
    // Astryx is reached only through @lablup/ui-common (which mirrors @astryxdesign/core/<X> as
    // @lablup/ui-common/<X> and lab as @lablup/ui-common/lab), so one copy of core serves the
    // whole app and its theme and layer contexts.
    files: ['**/*.{ts,tsx,js,jsx,mjs,cjs}'],
    rules: {
      'no-restricted-imports': ['error', { patterns: [{ group: ['@astryxdesign/*'], message: 'Import Astryx through @lablup/ui-common/<Component> (or @lablup/ui-common/lab).' }] }],
      'no-restricted-syntax': [
        'error',
        { selector: "ImportExpression[source.value=/^@astryxdesign\\//]", message: 'Import Astryx through @lablup/ui-common.' },
        { selector: "CallExpression[callee.name='require'][arguments.0.value=/^@astryxdesign\\//]", message: 'Import Astryx through @lablup/ui-common.' },
        { selector: "TSImportType[argument.literal.value=/^@astryxdesign\\//]", message: 'Import Astryx through @lablup/ui-common.' },
      ],
    },
  },
);
