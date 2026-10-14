// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import assert from 'node:assert/strict';
import test from 'node:test';
import { capturePerformanceDiagnostics } from './activity-performance-diagnostics.mjs';

test('captures fixed-argv bounded GPU and host diagnostics', async () => {
  const calls = [];
  const snapshot = await capturePerformanceDiagnostics('/usr/bin/nvidia-smi', {
    now: () => '2026-10-14T00:00:00.000Z',
    readFile: async (path, encoding) => {
      assert.equal(path, '/proc/loadavg');
      assert.equal(encoding, 'utf8');
      return '0.10 0.20 0.30 2/100 1234\n';
    },
    execFile: async (...args) => {
      calls.push(args);
      return { stdout: '2026/10/14 00:00:00.000, 0, P0, 98, 12, 67, 44.5, 1200, 900\n', stderr: '' };
    },
  });
  assert.deepEqual(calls, [[
    '/usr/bin/nvidia-smi',
    ['--query-gpu=timestamp,index,pstate,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory', '--format=csv,noheader,nounits'],
    { encoding: 'utf8', timeout: 10000, maxBuffer: 64 * 1024, windowsHide: true },
  ]]);
  assert.deepEqual(snapshot.host, { status: 'captured', load_one: 0.1, load_five: 0.2, load_fifteen: 0.3, running: 2, total: 100 });
  assert.equal(snapshot.gpu.status, 'captured');
  assert.equal(snapshot.gpu.graphics_clock_mhz, '1200');
});

test('records diagnostic failures without throwing or inventing samples', async () => {
  const snapshot = await capturePerformanceDiagnostics('/usr/bin/nvidia-smi', {
    readFile: async () => { throw new Error('loadavg unavailable'); },
    execFile: async () => { throw new Error('unsupported query\nprivate trailing detail'.repeat(30)); },
  });
  assert.equal(snapshot.host.status, 'unavailable');
  assert.equal(snapshot.gpu.status, 'unavailable');
  assert.match(snapshot.gpu.error, /^unsupported query private trailing detail/);
  assert.ok(snapshot.gpu.error.length <= 300);
});
