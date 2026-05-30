// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Elastic pipeline repartitioning state machine.
//!
//! Drives the `Idle → Draining → Rebalancing → Resuming → Idle` flow behind
//! `--enable-elastic-pp`. The coordinator owns the state and emits
//! [`RepartitionEvent`] values for every transition so the metrics/obervability
//! layer can surface them. All RPC / network fan-out (drain fan
//! out, probe polling, reload) is performed by [`ElasticRuntimeDriver`]
//! implementations — this module is intentionally transport-agnostic so it
//! can be unit-tested deterministically with fake drivers.
//!
//! Concurrency: the coordinator is `Send + Sync` through an internal `Mutex`.
//! External callers are expected to drive it from a single operator-command
//! task — the mutex is a safety net against accidental reentrancy, not a
//! throughput optimisation.
//!
//! Used by: `server_runtime` (when `--enable-elastic-pp` is set),
//! observability (`metrics.rs` sink for [`RepartitionEvent`]).

use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};

use super::partition::{LayerAdjacencyGroup, ModelProfile};
use super::partition_balance::balance_layers;

/// Configuration for elastic repartitioning.
///
/// All durations use monotonic time via [`Instant`]. The defaults were chosen
/// to behave conservatively on a clean cluster: the cool-down debounces
/// pressure oscillation, and the drain timeout is long enough for a 30-second
/// prompt to finish generating tokens without tripping the watchdog.
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticPpConfig {
    /// Whether the elastic flow is enabled. When `false`, the coordinator
    /// rejects every trigger. Callers should usually skip constructing one.
    pub enabled: bool,
    /// Maximum time the coordinator waits for `in_flight_requests` to reach
    /// zero across all stages after issuing `BeginDrain`.
    pub drain_timeout: Duration,
    /// Minimum time between two consecutive memory-pressure triggers on the
    /// same stage. Explicit operator triggers bypass this debounce.
    pub cool_down: Duration,
    /// Memory usage fraction above which a memory-pressure trigger fires.
    /// Clamped to `(0.0, 1.0]`.
    pub trigger_memory_fraction: f64,
}

impl Default for ElasticPpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            drain_timeout: Duration::from_secs(120),
            cool_down: Duration::from_secs(30),
            trigger_memory_fraction: 0.92,
        }
    }
}

impl ElasticPpConfig {
    /// Construct an enabled configuration with default timeouts.
    #[must_use]
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    /// Override the drain timeout.
    #[must_use]
    pub fn with_drain_timeout(mut self, drain_timeout: Duration) -> Self {
        self.drain_timeout = drain_timeout;
        self
    }

    /// Override the cool-down window.
    #[must_use]
    pub fn with_cool_down(mut self, cool_down: Duration) -> Self {
        self.cool_down = cool_down;
        self
    }

    /// Override the memory-pressure trigger fraction. Values outside
    /// `(0.0, 1.0]` are clamped.
    #[must_use]
    pub fn with_trigger_memory_fraction(mut self, fraction: f64) -> Self {
        self.trigger_memory_fraction = fraction.clamp(f64::MIN_POSITIVE, 1.0);
        self
    }
}

/// State of the elastic repartitioning state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RepartitionState {
    /// No repartition is active. New triggers are accepted (subject to
    /// cool-down for memory-pressure triggers).
    Idle,
    /// Drain has been issued; the coordinator is polling stage lifecycle
    /// snapshots waiting for `in_flight_requests == 0`.
    Draining,
    /// Drain completed; the coordinator is computing the new plan and
    /// asking stages to reload.
    Rebalancing,
    /// Reload finished; coordinator is waiting for stages to clear their
    /// `draining` flag and report healthy.
    Resuming,
    /// A prior repartition failed. New triggers are accepted only via
    /// explicit operator command.
    Failed,
}

impl fmt::Display for RepartitionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Idle => "idle",
            Self::Draining => "draining",
            Self::Rebalancing => "rebalancing",
            Self::Resuming => "resuming",
            Self::Failed => "failed",
        };
        f.write_str(s)
    }
}

/// Why a repartition was initiated.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum RepartitionTrigger {
    /// Operator explicitly asked for a repartition.
    Explicit,
    /// A stage exceeded the configured memory-pressure threshold.
    MemoryPressure {
        /// Stage index that reported the pressure.
        stage_index: u32,
        /// Memory usage fraction on that stage at the moment of triggering.
        fraction: f64,
    },
}

impl fmt::Display for RepartitionTrigger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Explicit => write!(f, "explicit"),
            Self::MemoryPressure {
                stage_index,
                fraction,
            } => write!(
                f,
                "memory_pressure(stage={stage_index}, frac={:.2})",
                fraction
            ),
        }
    }
}

impl RepartitionTrigger {
    /// Short label suitable for a metrics `trigger` dimension (low-cardinality).
    pub fn kind_label(&self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::MemoryPressure { .. } => "memory_pressure",
        }
    }
}

/// Final outcome of a repartition attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RepartitionOutcome {
    /// Drain, rebalance, and resume all succeeded.
    Completed,
    /// Drain or rebalance timed out or was cancelled by the operator.
    Aborted,
    /// The planner or a peer reported an unrecoverable error.
    Failed,
}

impl fmt::Display for RepartitionOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Completed => "completed",
            Self::Aborted => "aborted",
            Self::Failed => "failed",
        };
        f.write_str(s)
    }
}

/// Observability event emitted for every meaningful state transition.
///
/// The metrics subsystem consumes these via a [`RepartitionEventSink`]
/// implementation. One event per transition keeps the counter math simple —
/// downstream dashboards filter by `outcome` / `trigger_kind` as needed.
#[derive(Debug, Clone)]
pub struct RepartitionEvent {
    /// What caused the repartition.
    pub trigger: RepartitionTrigger,
    /// Final state the coordinator transitioned to.
    pub to_state: RepartitionState,
    /// Duration spent in `Draining` (zero when the event fires before drain
    /// starts — for example when a trigger is debounced).
    pub drain_duration: Duration,
    /// Wall-clock time from trigger acceptance until the event was emitted.
    pub total_duration: Duration,
    /// Final outcome, if the repartition has concluded. `None` for
    /// intermediate transitions.
    pub outcome: Option<RepartitionOutcome>,
    /// Per-stage layer range before the repartition (empty until drain is
    /// initiated).
    pub ranges_before: Vec<Range<usize>>,
    /// Per-stage layer range after the repartition (empty until the planner
    /// produces a new plan).
    pub ranges_after: Vec<Range<usize>>,
}

/// Sink that receives [`RepartitionEvent`] values as they are emitted.
///
/// The default wiring forwards events to the pipeline metrics subsystem
/// (adds counters + histograms). Tests can plug in a
/// [`RecordingEventSink`] to capture emitted events for assertion.
pub trait RepartitionEventSink: Send + Sync {
    fn record_event(&self, event: &RepartitionEvent);
}

/// No-op sink used when elastic repartitioning is disabled or when callers
/// have not wired a real sink yet.
#[derive(Debug, Default)]
pub struct NoopEventSink;

impl RepartitionEventSink for NoopEventSink {
    fn record_event(&self, _event: &RepartitionEvent) {}
}

/// Test/diagnostic sink that keeps every event in a `Vec`.
#[derive(Debug, Default)]
pub struct RecordingEventSink {
    events: Mutex<Vec<RepartitionEvent>>,
}

impl RecordingEventSink {
    /// Create a fresh recorder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot of all events recorded so far.
    pub fn events(&self) -> Vec<RepartitionEvent> {
        self.events
            .lock()
            .expect("recording event sink poisoned")
            .clone()
    }
}

impl RepartitionEventSink for RecordingEventSink {
    fn record_event(&self, event: &RepartitionEvent) {
        self.events
            .lock()
            .expect("recording event sink poisoned")
            .push(event.clone());
    }
}

/// Transport-facing hook that the coordinator uses to drive a running
/// cluster. The production implementation wraps
/// [`super::runtime::RemotePipelineRuntime`]; unit tests swap in a fake.
pub trait ElasticRuntimeDriver: Send + Sync {
    /// Fan out a `BeginDrain` command and flip the local runtime's
    /// `draining` flag.
    fn begin_drain(&self) -> Result<()>;

    /// Poll each stage and return the total number of in-flight requests.
    fn in_flight_requests(&self) -> Result<usize>;

    /// Apply the new layer ranges. Implementations reload the partial
    /// weights via `partial_loading` and flush the KV cache. Returns
    /// `Err` on the first peer that fails so the coordinator can transition
    /// to `Failed` without partial commit.
    fn apply_new_plan(&self, new_ranges: &[Range<usize>]) -> Result<()>;

    /// Release the drain flag so new sequences can be admitted again.
    fn release_drain(&self) -> Result<()>;

    /// Layer ranges currently active on every stage, in stage order.
    fn current_ranges(&self) -> Result<Vec<Range<usize>>>;

    /// Per-stage memory budgets the planner should target. Implementations
    /// typically derive this from `StageHealth` memory reports plus the
    /// KV cache configuration.
    fn per_stage_budget_bytes(&self) -> Result<Vec<u64>>;

    /// Per-stage device identifier strings for diagnostic messages. The
    /// length must match `per_stage_budget_bytes`.
    fn device_ids(&self) -> Result<Vec<String>>;
}

/// Coordinator for the elastic repartition state machine.
///
/// Construction is cheap; the coordinator owns no transport resources
/// itself — it delegates to the driver for every network-facing action.
/// Embedding it in a long-running server is safe: state transitions are
/// serialised behind an internal mutex and the coordinator emits exactly
/// one event per transition.
pub struct RepartitionCoordinator {
    config: ElasticPpConfig,
    driver: Arc<dyn ElasticRuntimeDriver>,
    profile: ModelProfile,
    sink: Arc<dyn RepartitionEventSink>,
    inner: Mutex<CoordinatorInner>,
}

impl fmt::Debug for RepartitionCoordinator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RepartitionCoordinator")
            .field("config", &self.config)
            .field("state", &self.state())
            .field("profile_layers", &self.profile.num_layers)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct CoordinatorInner {
    state: RepartitionState,
    last_memory_trigger: HashMap<u32, Instant>,
}

impl RepartitionCoordinator {
    /// Create a new coordinator. Returns `Err` when the configuration is
    /// disabled — callers that want an always-disabled path should skip
    /// constructing one entirely.
    pub fn new(
        config: ElasticPpConfig,
        driver: Arc<dyn ElasticRuntimeDriver>,
        profile: ModelProfile,
        sink: Arc<dyn RepartitionEventSink>,
    ) -> Result<Self> {
        if !config.enabled {
            bail!("elastic repartitioning is disabled in this configuration");
        }
        Ok(Self {
            config,
            driver,
            profile,
            sink,
            inner: Mutex::new(CoordinatorInner {
                state: RepartitionState::Idle,
                last_memory_trigger: HashMap::new(),
            }),
        })
    }

    /// Current state of the state machine.
    pub fn state(&self) -> RepartitionState {
        self.inner
            .lock()
            .expect("elastic coordinator poisoned")
            .state
    }

    /// Run a repartition end-to-end and block until it terminates. This is
    /// the primary entry point for both operator-initiated commands and
    /// pressure-driven triggers.
    ///
    /// Errors:
    ///
    /// * Returns `Err` when the trigger is rejected (coordinator busy, stale
    ///   pressure trigger inside the cool-down window, or prior `Failed`
    ///   state requires explicit clearance).
    /// * Returns `Err` when the driver reports an unrecoverable error in
    ///   drain, rebalance, or reload. The state machine is driven to
    ///   `Failed`, an event is emitted, and the original error is wrapped.
    pub fn run(&self, trigger: RepartitionTrigger) -> Result<RepartitionOutcome> {
        let started = Instant::now();

        if let Err(err) = self.accept_trigger(&trigger) {
            // Emit an aborted event so operators see the rejection in metrics.
            let event = RepartitionEvent {
                trigger: trigger.clone(),
                to_state: self.state(),
                drain_duration: Duration::ZERO,
                total_duration: started.elapsed(),
                outcome: Some(RepartitionOutcome::Aborted),
                ranges_before: Vec::new(),
                ranges_after: Vec::new(),
            };
            self.sink.record_event(&event);
            return Err(err);
        }

        let ranges_before = self.driver.current_ranges().unwrap_or_default();

        // Transition: Idle → Draining.
        self.set_state(RepartitionState::Draining);
        self.emit_progress_event(
            &trigger,
            RepartitionState::Draining,
            Duration::ZERO,
            started,
            &ranges_before,
            &[],
        );

        // Drain step.
        let drain_started = Instant::now();
        let drain_result = self.drain_with_timeout();
        let drain_duration = drain_started.elapsed();
        if let Err(err) = drain_result {
            self.finalise_failure(
                trigger.clone(),
                drain_duration,
                started,
                &ranges_before,
                &[],
                &err,
            );
            return Err(err);
        }

        // Transition: Draining → Rebalancing.
        self.set_state(RepartitionState::Rebalancing);
        self.emit_progress_event(
            &trigger,
            RepartitionState::Rebalancing,
            drain_duration,
            started,
            &ranges_before,
            &[],
        );

        // Compute the new plan.
        let new_ranges = match self.compute_new_plan() {
            Ok(ranges) => ranges,
            Err(err) => {
                self.finalise_failure(
                    trigger.clone(),
                    drain_duration,
                    started,
                    &ranges_before,
                    &[],
                    &err,
                );
                return Err(err);
            }
        };

        // Apply it.
        if let Err(err) = self.driver.apply_new_plan(&new_ranges) {
            self.finalise_failure(
                trigger.clone(),
                drain_duration,
                started,
                &ranges_before,
                &new_ranges,
                &err,
            );
            return Err(err);
        }

        // Transition: Rebalancing → Resuming.
        self.set_state(RepartitionState::Resuming);
        self.emit_progress_event(
            &trigger,
            RepartitionState::Resuming,
            drain_duration,
            started,
            &ranges_before,
            &new_ranges,
        );

        if let Err(err) = self.driver.release_drain() {
            self.finalise_failure(
                trigger.clone(),
                drain_duration,
                started,
                &ranges_before,
                &new_ranges,
                &err,
            );
            return Err(err);
        }

        // Transition: Resuming → Idle.
        self.set_state(RepartitionState::Idle);
        let event = RepartitionEvent {
            trigger,
            to_state: RepartitionState::Idle,
            drain_duration,
            total_duration: started.elapsed(),
            outcome: Some(RepartitionOutcome::Completed),
            ranges_before,
            ranges_after: new_ranges,
        };
        self.sink.record_event(&event);
        Ok(RepartitionOutcome::Completed)
    }

    /// Clear a `Failed` state so the next trigger can be accepted again.
    /// Returns `Err` when the coordinator is not currently in `Failed`.
    pub fn clear_failure(&self) -> Result<()> {
        let mut inner = self.inner.lock().expect("elastic coordinator poisoned");
        if inner.state != RepartitionState::Failed {
            bail!(
                "clear_failure called while coordinator is in {} (expected failed)",
                inner.state
            );
        }
        inner.state = RepartitionState::Idle;
        Ok(())
    }

    fn accept_trigger(&self, trigger: &RepartitionTrigger) -> Result<()> {
        let mut inner = self.inner.lock().expect("elastic coordinator poisoned");
        match inner.state {
            RepartitionState::Idle => {}
            RepartitionState::Failed => match trigger {
                RepartitionTrigger::Explicit => {}
                RepartitionTrigger::MemoryPressure { .. } => {
                    bail!("coordinator is in failed state; only explicit triggers are accepted")
                }
            },
            other => bail!("coordinator busy: cannot accept trigger while in {other} state"),
        }

        if let RepartitionTrigger::MemoryPressure {
            stage_index,
            fraction,
        } = trigger
        {
            if *fraction < self.config.trigger_memory_fraction {
                bail!(
                    "memory-pressure trigger below threshold: observed={:.3} threshold={:.3}",
                    fraction,
                    self.config.trigger_memory_fraction
                );
            }
            if let Some(last) = inner.last_memory_trigger.get(stage_index)
                && last.elapsed() < self.config.cool_down
            {
                bail!(
                    "memory-pressure trigger on stage {stage_index} within cool-down window ({:?} remaining)",
                    self.config.cool_down.saturating_sub(last.elapsed())
                );
            }
            inner
                .last_memory_trigger
                .insert(*stage_index, Instant::now());
        }

        Ok(())
    }

    fn drain_with_timeout(&self) -> Result<()> {
        self.driver.begin_drain()?;

        let deadline = Instant::now() + self.config.drain_timeout;
        // Poll in small increments; in production the runtime is event-driven
        // via `probe_stages`, but a fixed interval keeps the control loop
        // simple and is testable with a fake driver.
        loop {
            let in_flight = self.driver.in_flight_requests()?;
            if in_flight == 0 {
                return Ok(());
            }
            if Instant::now() >= deadline {
                bail!(
                    "drain timeout exceeded ({:?}); {in_flight} request(s) still in flight",
                    self.config.drain_timeout
                );
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn compute_new_plan(&self) -> Result<Vec<Range<usize>>> {
        let per_layer_bytes = self.profile.effective_layer_bytes();
        let per_stage_budget = self.driver.per_stage_budget_bytes()?;
        let device_ids = self.driver.device_ids()?;
        if per_stage_budget.len() != device_ids.len() {
            bail!(
                "per_stage_budget_bytes and device_ids length mismatch ({} vs {})",
                per_stage_budget.len(),
                device_ids.len(),
            );
        }
        let forbidden = self.profile.forbidden_boundaries();
        let adjacency: &[LayerAdjacencyGroup] = &self.profile.adjacency;
        let (ranges, warnings) = balance_layers(
            &per_layer_bytes,
            &per_stage_budget,
            &forbidden,
            adjacency,
            &device_ids,
        )?;
        for warn in warnings {
            tracing::warn!(target: "elastic-pp", "{warn}");
        }
        Ok(ranges)
    }

    fn set_state(&self, state: RepartitionState) {
        self.inner
            .lock()
            .expect("elastic coordinator poisoned")
            .state = state;
    }

    fn emit_progress_event(
        &self,
        trigger: &RepartitionTrigger,
        to_state: RepartitionState,
        drain_duration: Duration,
        started: Instant,
        ranges_before: &[Range<usize>],
        ranges_after: &[Range<usize>],
    ) {
        let event = RepartitionEvent {
            trigger: trigger.clone(),
            to_state,
            drain_duration,
            total_duration: started.elapsed(),
            outcome: None,
            ranges_before: ranges_before.to_vec(),
            ranges_after: ranges_after.to_vec(),
        };
        self.sink.record_event(&event);
    }

    fn finalise_failure(
        &self,
        trigger: RepartitionTrigger,
        drain_duration: Duration,
        started: Instant,
        ranges_before: &[Range<usize>],
        ranges_after: &[Range<usize>],
        err: &anyhow::Error,
    ) {
        self.set_state(RepartitionState::Failed);
        // Best-effort drain release so the cluster does not get stuck.
        let _ = self.driver.release_drain();
        tracing::error!(
            target: "elastic-pp",
            trigger = %trigger,
            error = %err,
            "elastic repartition failed"
        );
        let event = RepartitionEvent {
            trigger,
            to_state: RepartitionState::Failed,
            drain_duration,
            total_duration: started.elapsed(),
            outcome: Some(RepartitionOutcome::Failed),
            ranges_before: ranges_before.to_vec(),
            ranges_after: ranges_after.to_vec(),
        };
        self.sink.record_event(&event);
    }
}

#[cfg(test)]
#[path = "elastic_tests.rs"]
mod tests;
