// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { execFile as execFileCallback } from 'node:child_process';
import { readFile as readFileDefault } from 'node:fs/promises';
import { promisify } from 'node:util';

const execFileDefault = promisify(execFileCallback);
const queryFields = [
  'timestamp',
  'index',
  'pstate',
  'utilization.gpu',
  'utilization.memory',
  'temperature.gpu',
  'power.draw',
  'clocks.current.graphics',
  'clocks.current.memory',
];
const valueNames = [
  'timestamp',
  'index',
  'pstate',
  'gpu_utilization_percent',
  'memory_utilization_percent',
  'temperature_c',
  'power_w',
  'graphics_clock_mhz',
  'memory_clock_mhz',
];

function compactError(error) {
  const message = error instanceof Error ? error.message : String(error);
  return message.replaceAll(/\s+/g, ' ').slice(0, 300);
}

function hostState(text) {
  const fields = text.trim().split(/\s+/);
  if (fields.length < 4) throw new Error('unexpected /proc/loadavg shape');
  const [running, total] = fields[3].split('/');
  return {
    load_one: Number(fields[0]),
    load_five: Number(fields[1]),
    load_fifteen: Number(fields[2]),
    running: Number(running),
    total: Number(total),
  };
}

function gpuState(stdout) {
  const lines = stdout.trim().split('\n').filter(Boolean);
  if (lines.length !== 1) throw new Error(`expected one GPU row, got ${lines.length}`);
  const values = lines[0].split(',').map((value) => value.trim());
  if (values.length !== valueNames.length) throw new Error(`expected ${valueNames.length} GPU fields, got ${values.length}`);
  return Object.fromEntries(valueNames.map((name, index) => [name, values[index]]));
}

/** Capture bounded, best-effort runner diagnostics without affecting the acceptance verdict. */
export async function capturePerformanceDiagnostics(nvidiaSmi, dependencies = {}) {
  const execFile = dependencies.execFile ?? execFileDefault;
  const readFile = dependencies.readFile ?? readFileDefault;
  const now = dependencies.now ?? (() => new Date().toISOString());
  const snapshot = { observed_at: now() };
  try {
    snapshot.host = { status: 'captured', ...hostState(await readFile('/proc/loadavg', 'utf8')) };
  } catch (error) {
    snapshot.host = { status: 'unavailable', error: compactError(error) };
  }
  if (!nvidiaSmi) {
    snapshot.gpu = { status: 'not-requested' };
    return snapshot;
  }
  try {
    const { stdout } = await execFile(nvidiaSmi, [
      `--query-gpu=${queryFields.join(',')}`,
      '--format=csv,noheader,nounits',
    ], { encoding: 'utf8', timeout: 10000, maxBuffer: 64 * 1024, windowsHide: true });
    snapshot.gpu = { status: 'captured', ...gpuState(stdout) };
  } catch (error) {
    snapshot.gpu = { status: 'unavailable', error: compactError(error) };
  }
  return snapshot;
}
