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

use super::super::ple_direct::{AlignedBuffer, DIRECT_ALIGN, coalesce};
use super::*;

fn seg(file: u32, offset: u64, len: u32) -> (u32, Segment) {
    (
        0,
        Segment {
            file,
            offset,
            len,
            within: 0,
        },
    )
}

#[test]
fn neighbours_coalesce_and_distant_rows_do_not() {
    let segments = [
        seg(0, 0, 100),
        seg(0, 200, 100),
        seg(0, 20000, 100),
        seg(1, 300, 100),
    ];
    let spans = coalesce(&segments);
    assert_eq!(spans.len(), 3);
    assert_eq!(
        (spans[0].start, spans[0].end, spans[0].first, spans[0].last),
        (0, 300, 0, 2)
    );
    assert_eq!(spans[1].start, 20000);
    assert_eq!(spans[2].file, 1);
}

#[test]
fn io_mode_parsing_names_the_three_modes() {
    assert_eq!(PleIoMode::parse("mmap"), Some(PleIoMode::Mmap));
    assert_eq!(PleIoMode::parse("direct"), Some(PleIoMode::Direct));
    assert_eq!(PleIoMode::parse("ram"), Some(PleIoMode::Ram));
    assert_eq!(PleIoMode::parse("o_direct"), None);
}

#[test]
fn the_read_buffer_is_page_aligned() {
    let mut buffer = AlignedBuffer::default();
    let slice = buffer.slice(8192);
    assert_eq!(slice.len(), 8192);
    assert_eq!(slice.as_ptr() as usize % DIRECT_ALIGN, 0);
}
