// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { defineConfig, devices } from '@playwright/test';

const target = process.env.MLXCEL_WEBUI_CSP_URL;
if (!target) throw new Error('Set MLXCEL_WEBUI_CSP_URL to the real secured /webui/ URL; no preview server or copied headers are substituted.');
const url = new URL(target);
if (!['http:', 'https:'].includes(url.protocol) || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) || url.username || url.password) throw new Error('The served-CSP harness must target an explicitly supplied loopback server without URL credentials.');
export default defineConfig({ testDir: './tests', testMatch: 'csp.spec.ts', fullyParallel: false, use: { ...devices['Desktop Chrome'] } });
