// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import type { MeasuredValue, Operation, WebUiSnapshot } from '../../api/types';

export function metricValue(metric: MeasuredValue): string {
  if (metric.value === null || !Number.isFinite(metric.value)) return 'N/A';
  return `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(metric.value)} ${metric.unit}`;
}

export function operationProgress(operation: Operation): number | undefined {
  const progress = operation.progress;
  return !progress.indeterminate && progress.total_bytes !== null && progress.total_bytes > 0
    ? Math.min(100, 100 * progress.completed_bytes / progress.total_bytes) : undefined;
}

// Export an allowlist, never a redacted copy of the server snapshot: even debug
// slots, operation error messages, target repo names and paths are excluded.
export function diagnostics(snapshot: WebUiSnapshot): string {
  const runtime = snapshot.selectedModelId === null ? undefined : snapshot.runtimes.get(snapshot.selectedModelId);
  return JSON.stringify({
    format: 'mlxcel-webui-diagnostics-v1',
    connection: snapshot.connection,
    last_successful_at: snapshot.lastSuccessfulAt,
    operation_counts: Object.fromEntries(['queued', 'running', 'cancelling', 'succeeded', 'failed', 'cancelled'].map((state) => [state, [...snapshot.operations.values()].filter((op) => op.state === state).length])),
    catalog_count: snapshot.catalog.length,
    runtime_present: runtime !== undefined,
    history_samples: snapshot.runtimeHistory.length,
    // Export only numeric value and availability, not server-provided strings.
    measurements: runtime === undefined ? [] : Object.values(runtime.measurements).map((metric) => ({ value: metric.value, available: metric.value !== null })),
  }, null, 2);
}

export function downloadDiagnostics(snapshot: WebUiSnapshot): void {
  const url = URL.createObjectURL(new Blob([diagnostics(snapshot)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'mlxcel-diagnostics.json';
  link.click();
  URL.revokeObjectURL(url);
}
