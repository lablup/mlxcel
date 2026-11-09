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

//! Unbuffered PLE row reads: open flags, span coalescing, aligned positional
//! reads.

use std::fs::File;
use std::path::Path;
use std::sync::Mutex;

use super::ple_io::Segment;

/// Direct-read alignment: the O_DIRECT requirement on common Linux filesystems,
/// and a whole-page span everywhere else.
pub(super) const DIRECT_ALIGN: usize = 4096;
/// Neighbouring segments within this gap are read as one span.
const COALESCE_GAP: u64 = DIRECT_ALIGN as u64;
const MAX_SPAN_BYTES: u64 = 1 << 20;

pub(super) fn open_direct(path: &Path) -> Result<File, String> {
    let open = |flags: i32| -> std::io::Result<File> {
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(flags);
        }
        let _ = flags;
        options.open(path)
    };
    #[cfg(target_os = "linux")]
    let file = open(libc::O_DIRECT).or_else(|_| open(0));
    #[cfg(not(target_os = "linux"))]
    let file = open(0);
    let file = file.map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: `file` is an open descriptor; F_NOCACHE takes an int flag.
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
    }
    Ok(file)
}

/// A run of segments to read at once. `first..last` indexes the sorted segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Span {
    pub(super) file: u32,
    pub(super) start: u64,
    pub(super) end: u64,
    pub(super) first: usize,
    pub(super) last: usize,
}

pub(super) fn coalesce(segments: &[(u32, Segment)]) -> Vec<Span> {
    let mut spans: Vec<Span> = Vec::new();
    for (index, (_, segment)) in segments.iter().enumerate() {
        let end = segment.offset + u64::from(segment.len);
        match spans.last_mut() {
            Some(span)
                if span.file == segment.file
                    && segment.offset <= span.end + COALESCE_GAP
                    && end - span.start <= MAX_SPAN_BYTES =>
            {
                span.end = span.end.max(end);
                span.last = index + 1;
            }
            _ => spans.push(Span {
                file: segment.file,
                start: segment.offset,
                end,
                first: index,
                last: index + 1,
            }),
        }
    }
    spans
}

/// A 4096-aligned read buffer, reusing the allocation between spans.
#[derive(Default)]
pub(super) struct AlignedBuffer {
    storage: Vec<u8>,
}

impl AlignedBuffer {
    pub(super) fn slice(&mut self, len: usize) -> &mut [u8] {
        self.storage.resize(len + DIRECT_ALIGN, 0);
        let shift = self.storage.as_ptr().align_offset(DIRECT_ALIGN);
        &mut self.storage[shift..shift + len]
    }
}

pub(super) fn read_span(
    files: &[File],
    span: &Span,
    segments: &[(u32, Segment)],
    stride: usize,
    buffer: &mut AlignedBuffer,
    out: &Mutex<&mut [u8]>,
) -> Result<usize, String> {
    let align = DIRECT_ALIGN as u64;
    let start = span.start / align * align;
    let end = span.end.div_ceil(align) * align;
    let chunk = buffer.slice((end - start) as usize);
    let file = files
        .get(span.file as usize)
        .ok_or("PLE span names an unknown file")?;
    let needed = (span.end - start) as usize;
    let got = read_full(file, chunk, start, needed)
        .map_err(|e| format!("PLE row read failed at byte {start}: {e}"))?;
    if got < needed {
        return Err(format!(
            "PLE row read came up short at byte {start}: got {got} bytes"
        ));
    }
    let mut out = out.lock().map_err(|_| "PLE output buffer lock poisoned")?;
    for (row, segment) in &segments[span.first..span.last] {
        let from = (segment.offset - start) as usize;
        let record = *row as usize * stride + segment.within as usize;
        out[record..record + segment.len as usize]
            .copy_from_slice(&chunk[from..from + segment.len as usize]);
    }
    Ok(got)
}

/// Read until `needed` bytes have arrived. The buffer is rounded up to the
/// direct-read alignment, so a span ending at the end of a file gets a short
/// read; retrying its unaligned remainder under O_DIRECT would fail with EINVAL.
fn read_full(file: &File, buffer: &mut [u8], offset: u64, needed: usize) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < needed {
        let read = read_at(file, &mut buffer[filled..], offset + filled as u64)?;
        if read == 0 {
            break;
        }
        filled += read;
    }
    Ok(filled)
}

#[cfg(unix)]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(file, buffer, offset)
}

#[cfg(windows)]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(file, buffer, offset)
}
