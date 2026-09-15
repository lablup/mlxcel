// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { defineConfig, devices } from '@playwright/test';

const target = process.env.MLXCEL_WEBUI_ROUTER_URL;
const keyFile = process.env.MLXCEL_WEBUI_ROUTER_KEY_FILE;
const artifacts = process.env.MLXCEL_WEBUI_ROUTER_ARTIFACTS;
if (!target || !keyFile || !artifacts) throw new Error('Set MLXCEL_WEBUI_ROUTER_URL, MLXCEL_WEBUI_ROUTER_KEY_FILE and MLXCEL_WEBUI_ROUTER_ARTIFACTS; missing real-router setup is not a skipped pass.');
const url = new URL(target);
if (!['http:', 'https:'].includes(url.protocol) || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) || url.username || url.password || !url.pathname.endsWith('/webui/')) throw new Error('The Rust-router harness must target a loopback /webui/ URL without URL credentials.');

export default defineConfig({
  testDir: './tests',
  testMatch: 'router-real.spec.ts',
  fullyParallel: false,
  reporter: 'list',
  outputDir: process.env.MLXCEL_WEBUI_ROUTER_ARTIFACTS,
  use: {
    ...devices['Desktop Chrome'],
    baseURL: target,
    trace: 'off',
    screenshot: 'off',
    video: 'off',
  },
});
