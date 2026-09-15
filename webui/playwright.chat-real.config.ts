import { defineConfig, devices } from '@playwright/test';
export default defineConfig({testDir:'./tests',testMatch:'**/chat-real.spec.ts',fullyParallel:false,workers:1,retries:0,timeout:120000,reporter:'list',use:{...devices['Desktop Chrome']}});
