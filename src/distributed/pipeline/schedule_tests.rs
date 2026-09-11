// Copyright 2025-2026 Lablup Inc.
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

use super::*;

fn make_config(num_stages: u32, micro_batch_size: usize) -> PipelineConfig {
    PipelineConfig::new(num_stages, micro_batch_size).unwrap()
}

#[test]
fn pipeline_config_validation() {
    assert!(PipelineConfig::new(1, 2).is_err()); // need >= 2 stages
    assert!(PipelineConfig::new(2, 0).is_err()); // need > 0 micro-batch size
    assert!(PipelineConfig::new(2, 1).is_ok());
}

#[test]
fn gpipe_two_stages_one_micro_batch() {
    let config = make_config(2, 4);
    let mut sched = GPipeSchedule::new(config, 1).unwrap();

    // First action: forward mb0 through stage 0.
    let a1 = sched.next_action();
    assert_eq!(
        a1,
        ScheduleAction::Forward {
            stage_index: 0,
            micro_batch_id: 0,
        }
    );

    // No more initial actions; should be Idle until we notify completion.
    let a2 = sched.next_action();
    assert_eq!(a2, ScheduleAction::Idle);

    // Notify stage 0 complete for mb0 -> schedule Receive+Forward for stage 1.
    sched.notify_forward_complete(0, 0);

    let a3 = sched.next_action();
    assert_eq!(
        a3,
        ScheduleAction::Receive {
            stage_index: 1,
            micro_batch_id: 0,
        }
    );

    let a4 = sched.next_action();
    assert_eq!(
        a4,
        ScheduleAction::Forward {
            stage_index: 1,
            micro_batch_id: 0,
        }
    );

    // Notify stage 1 complete -> should get Flush.
    sched.notify_forward_complete(1, 0);
    let a5 = sched.next_action();
    assert_eq!(a5, ScheduleAction::Flush { micro_batch_id: 0 });

    // Notify flush complete.
    sched.notify_flush_complete(0);
    assert!(sched.is_complete());

    let a6 = sched.next_action();
    assert_eq!(a6, ScheduleAction::Done);
}

#[test]
fn gpipe_two_stages_four_micro_batches() {
    let config = make_config(2, 1);
    let mut sched = GPipeSchedule::new(config, 4).unwrap();

    // Collect all initial forward actions (all 4 micro-batches through stage 0).
    let mut forwards = Vec::new();
    for _ in 0..4 {
        let action = sched.next_action();
        forwards.push(action);
    }
    for (i, f) in forwards.iter().enumerate() {
        assert_eq!(
            *f,
            ScheduleAction::Forward {
                stage_index: 0,
                micro_batch_id: i as u32,
            }
        );
    }

    // Should be idle now.
    assert_eq!(sched.next_action(), ScheduleAction::Idle);

    // Complete stage 0 for all micro-batches and drain their stage-1 actions.
    for mb in 0..4u32 {
        sched.notify_forward_complete(0, mb);
    }

    // Now we should have Receive+Forward pairs for stage 1, then Flush.
    for mb in 0..4u32 {
        let recv = sched.next_action();
        assert_eq!(
            recv,
            ScheduleAction::Receive {
                stage_index: 1,
                micro_batch_id: mb,
            }
        );
        let fwd = sched.next_action();
        assert_eq!(
            fwd,
            ScheduleAction::Forward {
                stage_index: 1,
                micro_batch_id: mb,
            }
        );
    }

    // Complete stage 1 for all and flush.
    for mb in 0..4u32 {
        sched.notify_forward_complete(1, mb);
    }

    for mb in 0..4u32 {
        let flush = sched.next_action();
        assert_eq!(flush, ScheduleAction::Flush { micro_batch_id: mb });
        sched.notify_flush_complete(mb);
    }

    assert!(sched.is_complete());
    assert_eq!(sched.next_action(), ScheduleAction::Done);
}

#[test]
fn gpipe_three_stages() {
    let config = make_config(3, 2);
    let mut sched = GPipeSchedule::new(config, 2).unwrap();

    // Forward mb0 and mb1 through stage 0.
    assert_eq!(
        sched.next_action(),
        ScheduleAction::Forward {
            stage_index: 0,
            micro_batch_id: 0
        }
    );
    assert_eq!(
        sched.next_action(),
        ScheduleAction::Forward {
            stage_index: 0,
            micro_batch_id: 1
        }
    );

    // Complete both through stage 0, then stage 1, then stage 2.
    for mb in 0..2u32 {
        sched.notify_forward_complete(0, mb);
    }
    // Drain receive+forward for stage 1.
    for _ in 0..4 {
        sched.next_action();
    }
    for mb in 0..2u32 {
        sched.notify_forward_complete(1, mb);
    }
    // Drain receive+forward for stage 2.
    for _ in 0..4 {
        sched.next_action();
    }
    for mb in 0..2u32 {
        sched.notify_forward_complete(2, mb);
    }

    // Flush both.
    for mb in 0..2u32 {
        assert_eq!(
            sched.next_action(),
            ScheduleAction::Flush { micro_batch_id: mb }
        );
        sched.notify_flush_complete(mb);
    }

    assert!(sched.is_complete());
}

#[test]
fn mark_sequence_done() {
    let config = make_config(2, 1);
    let mut sched = GPipeSchedule::new(config, 2).unwrap();

    sched.mark_sequence_done(0);
    // Should not affect the schedule; the micro-batch still goes through
    // all stages normally, just gets flushed at the end.
    assert!(!sched.is_complete());
}

#[test]
fn create_gpipe_schedule_convenience() {
    let config = make_config(2, 3);
    let sched = create_gpipe_schedule(config, 10).unwrap();
    // 10 / 3 = 4 micro-batches (ceiling).
    assert_eq!(sched.num_micro_batches(), 4);
    assert_eq!(sched.num_stages(), 2);
}

#[test]
fn create_gpipe_schedule_zero_batch_errors() {
    let config = make_config(2, 1);
    assert!(create_gpipe_schedule(config, 0).is_err());
}

#[test]
fn schedule_action_display() {
    let action = ScheduleAction::Forward {
        stage_index: 1,
        micro_batch_id: 3,
    };
    assert_eq!(format!("{action}"), "Forward(stage=1, mb=3)");

    let flush = ScheduleAction::Flush { micro_batch_id: 0 };
    assert_eq!(format!("{flush}"), "Flush(mb=0)");
}

#[test]
fn gpipe_display() {
    let config = make_config(2, 2);
    let sched = GPipeSchedule::new(config, 4).unwrap();
    let display = format!("{sched}");
    assert!(display.contains("GPipeSchedule"));
    assert!(display.contains("stages=2"));
    assert!(display.contains("mbs=4"));
}

#[test]
fn in_flight_tracking() {
    let config = make_config(2, 1);
    let mut sched = GPipeSchedule::new(config, 2).unwrap();

    // Drain initial forward actions.
    sched.next_action();
    sched.next_action();

    // Start processing.
    sched.notify_forward_complete(0, 0);
    assert_eq!(sched.in_flight(), 1); // mb0 started, not flushed.

    sched.notify_forward_complete(0, 1);
    assert_eq!(sched.in_flight(), 2); // both started.

    // Drain stage 1 actions.
    for _ in 0..4 {
        sched.next_action();
    }

    sched.notify_forward_complete(1, 0);
    sched.next_action(); // flush
    sched.notify_flush_complete(0);
    assert_eq!(sched.in_flight(), 1); // mb0 flushed, mb1 still in flight.
}
