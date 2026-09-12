import { spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('canonical WebUI contract fixtures', () => {
  it('validates every shared fixture with the repository schema gate', () => {
    const repo = join(process.cwd(), '..');
    const exampleCount = readdirSync(join(repo, 'tests/fixtures/webui/examples')).filter((name) => name.endsWith('.json')).length;
    const scenarioCount = readdirSync(join(repo, 'tests/fixtures/webui/scenarios')).filter((name) => name.endsWith('.json')).length;
    const rootCount = ['identity-vectors.json', 'requirement-map.json', 'strings.json'].length;
    const expectedCount = exampleCount + scenarioCount + rootCount;
    const python = process.env.WEBUI_CONTRACT_PY ?? 'python3';
    const result = spawnSync(python, ['scripts/ci/check_webui_contract.py', '--self-test'], { cwd: repo, encoding: 'utf8' });
    expect(result.status, `${result.stdout}\n${result.stderr}`).toBe(0);
    expect(result.stdout).toContain(`validated ${expectedCount} WebUI contract fixtures`);
    expect(expectedCount).toBe(32);
  });
});
