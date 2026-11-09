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

//! Reading PLE row records from disk.
//!
//! [`RowSource`] returns rows as interleaved records whatever the layout: the
//! `interleaved_q4` file already is one record per row, and the
//! `safetensors_ranges` layout assembles a record from the row's segments of
//! `weight`, `scales` and `biases`.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use memmap2::Mmap;

use super::ple_direct::{AlignedBuffer, coalesce, open_direct, read_span};
use super::ple_manifest::{PleLayout, PleManifest};

/// Environment override for the I/O mode.
pub const PLE_IO_ENV: &str = "MLXCEL_PLE_IO";

/// How rows reach host memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PleIoMode {
    /// Map the row file and copy the selected rows. Uses the page cache.
    #[default]
    Mmap,
    /// Unbuffered positional reads through a small thread pool.
    Direct,
    /// Read the whole table into anonymous memory once.
    Ram,
}

impl PleIoMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "mmap" => Some(Self::Mmap),
            "direct" => Some(Self::Direct),
            "ram" => Some(Self::Ram),
            _ => None,
        }
    }

    /// The mode selected by [`PLE_IO_ENV`].
    pub fn from_env() -> Result<Self, String> {
        match std::env::var(PLE_IO_ENV) {
            Ok(value) if value.is_empty() => Ok(Self::default()),
            Ok(value) => Self::parse(&value).ok_or_else(|| {
                format!("{PLE_IO_ENV}={value:?} is invalid; expected direct, mmap or ram")
            }),
            Err(_) => Ok(Self::default()),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Mmap => "mmap",
            Self::Direct => "direct",
            Self::Ram => "ram",
        }
    }
}

/// A contiguous piece of a row in one file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Segment {
    pub(crate) file: u32,
    pub(crate) offset: u64,
    pub(crate) len: u32,
    /// Byte offset of this segment within the row's record.
    pub(crate) within: u32,
}

/// Maps a row id to its on-disk segments.
#[derive(Debug, Clone)]
struct Geometry {
    files: Vec<PathBuf>,
    shards: Vec<GeometryShard>,
    row_starts: Vec<u64>,
    stride: u64,
    interleaved: bool,
}

#[derive(Debug, Clone)]
struct GeometryShard {
    /// `(file, base offset, row bytes)`.
    tensors: Vec<(u32, u64, u32)>,
}

impl Geometry {
    fn new(manifest: &PleManifest) -> Self {
        let stride = manifest.row_stride() as u64;
        match &manifest.layout {
            PleLayout::Interleaved { data_path } => Self {
                files: vec![data_path.clone()],
                shards: Vec::new(),
                row_starts: Vec::new(),
                stride,
                interleaved: true,
            },
            PleLayout::SafetensorsRanges { shards } => {
                let mut files: Vec<PathBuf> = Vec::new();
                let mut file_index = |path: &Path| -> u32 {
                    match files.iter().position(|known| known == path) {
                        Some(index) => index as u32,
                        None => {
                            files.push(path.to_path_buf());
                            (files.len() - 1) as u32
                        }
                    }
                };
                let mapped: Vec<GeometryShard> = shards
                    .iter()
                    .map(|shard| GeometryShard {
                        tensors: shard
                            .tensors
                            .iter()
                            .map(|tensor| {
                                (
                                    file_index(&tensor.path),
                                    tensor.offset,
                                    tensor.row_bytes as u32,
                                )
                            })
                            .collect(),
                    })
                    .collect();
                let row_starts = shards.iter().map(|shard| shard.row_start).collect();
                Self {
                    files,
                    shards: mapped,
                    row_starts,
                    stride,
                    interleaved: false,
                }
            }
        }
    }

    fn segments(&self, id: u64, out: &mut Vec<Segment>) {
        if self.interleaved {
            out.push(Segment {
                file: 0,
                offset: id * self.stride,
                len: self.stride as u32,
                within: 0,
            });
            return;
        }
        let shard_index = self.row_starts.partition_point(|&start| start <= id) - 1;
        let local = id - self.row_starts[shard_index];
        let mut within = 0;
        for &(file, offset, row_bytes) in &self.shards[shard_index].tensors {
            out.push(Segment {
                file,
                offset: offset + local * u64::from(row_bytes),
                len: row_bytes,
                within,
            });
            within += row_bytes;
        }
    }
}

/// The opened backing of a table.
pub(crate) struct RowSource {
    geometry: Geometry,
    backing: Backing,
}

enum Backing {
    Mmap(Vec<Mmap>),
    Direct(Vec<File>),
    Ram(Vec<u8>),
}

const DIRECT_THREADS: usize = 8;
/// Below this many spans the reads stay on the calling thread.
const DIRECT_PARALLEL_SPANS: usize = 16;
/// Rows copied per batch while loading a table into RAM.
const RAM_LOAD_BATCH_ROWS: usize = 1 << 20;

impl RowSource {
    pub(crate) fn open(manifest: &PleManifest, mode: PleIoMode) -> Result<Self, String> {
        let geometry = Geometry::new(manifest);
        let backing = match mode {
            PleIoMode::Direct => {
                let files = geometry
                    .files
                    .iter()
                    .map(|path| open_direct(path))
                    .collect::<Result<_, _>>()?;
                Backing::Direct(files)
            }
            PleIoMode::Mmap => Backing::Mmap(map_files(&geometry.files)?),
            PleIoMode::Ram => {
                let source = RowSource {
                    geometry: geometry.clone(),
                    backing: Backing::Mmap(map_files(&geometry.files)?),
                };
                Backing::Ram(load_table(
                    &source,
                    manifest.row_count,
                    geometry.stride as usize,
                )?)
            }
        };
        Ok(Self { geometry, backing })
    }

    /// Whether lookups need the row cache in front of this source.
    pub(crate) fn wants_cache(&self) -> bool {
        !matches!(self.backing, Backing::Ram(_))
    }

    /// Read records for the sorted `ids` into `out` (`ids.len() * stride` bytes),
    /// returning the bytes taken from storage.
    pub(crate) fn read(&self, ids: &[u64], out: &mut [u8]) -> Result<u64, String> {
        match &self.backing {
            Backing::Ram(table) => {
                let stride = self.geometry.stride as usize;
                for (id, dest) in ids.iter().zip(out.chunks_exact_mut(stride)) {
                    let start = *id as usize * stride;
                    dest.copy_from_slice(&table[start..start + stride]);
                }
                Ok(0)
            }
            Backing::Mmap(maps) => {
                let mut segments = Vec::with_capacity(ids.len());
                let mut bytes = 0;
                let stride = self.geometry.stride as usize;
                for (index, &id) in ids.iter().enumerate() {
                    segments.clear();
                    self.geometry.segments(id, &mut segments);
                    let dest = &mut out[index * stride..(index + 1) * stride];
                    for segment in &segments {
                        let range =
                            segment.offset as usize..segment.offset as usize + segment.len as usize;
                        let map = maps
                            .get(segment.file as usize)
                            .and_then(|map| map.get(range))
                            .ok_or_else(|| {
                                format!("PLE row {id}: mapped range is outside its file")
                            })?;
                        dest[segment.within as usize
                            ..segment.within as usize + segment.len as usize]
                            .copy_from_slice(map);
                        bytes += u64::from(segment.len);
                    }
                }
                Ok(bytes)
            }
            Backing::Direct(files) => self.read_direct(files, ids, out),
        }
    }

    fn read_direct(&self, files: &[File], ids: &[u64], out: &mut [u8]) -> Result<u64, String> {
        let stride = self.geometry.stride as usize;
        let mut segments: Vec<(u32, Segment)> = Vec::with_capacity(ids.len());
        let mut scratch = Vec::new();
        for (index, &id) in ids.iter().enumerate() {
            scratch.clear();
            self.geometry.segments(id, &mut scratch);
            segments.extend(scratch.iter().map(|&segment| (index as u32, segment)));
        }
        segments.sort_unstable_by_key(|(_, segment)| (segment.file, segment.offset));
        let spans = coalesce(&segments);

        let out = Mutex::new(out);
        let next = AtomicUsize::new(0);
        let total = AtomicUsize::new(0);
        let first_error = Mutex::new(None::<String>);
        let work = || {
            let mut buffer = AlignedBuffer::default();
            loop {
                if first_error.lock().is_ok_and(|error| error.is_some()) {
                    return;
                }
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(span) = spans.get(index) else { return };
                match read_span(files, span, &segments, stride, &mut buffer, &out) {
                    Ok(read) => {
                        total.fetch_add(read, Ordering::Relaxed);
                    }
                    Err(error) => {
                        if let Ok(mut slot) = first_error.lock() {
                            slot.get_or_insert(error);
                        }
                        return;
                    }
                }
            }
        };
        if spans.len() < DIRECT_PARALLEL_SPANS {
            work();
        } else {
            std::thread::scope(|scope| {
                for _ in 0..DIRECT_THREADS {
                    scope.spawn(work);
                }
            });
        }
        match first_error.into_inner().ok().flatten() {
            Some(error) => Err(error),
            None => Ok(total.into_inner() as u64),
        }
    }
}

fn map_files(paths: &[PathBuf]) -> Result<Vec<Mmap>, String> {
    paths
        .iter()
        .map(|path| {
            let file =
                File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
            // SAFETY: the map is read-only and the checkpoint files are not
            // modified while the model is loaded, the same assumption MLX's
            // own safetensors mapping makes.
            let map = unsafe { Mmap::map(&file) }
                .map_err(|e| format!("cannot map {}: {e}", path.display()))?;
            #[cfg(unix)]
            {
                let _ = map.advise(memmap2::Advice::Random);
            }
            Ok(map)
        })
        .collect()
}

fn load_table(source: &RowSource, row_count: u64, stride: usize) -> Result<Vec<u8>, String> {
    let total = usize::try_from(row_count)
        .ok()
        .and_then(|rows| rows.checked_mul(stride))
        .ok_or_else(|| "PLE table is too large for ram mode".to_string())?;
    let mut table = vec![0u8; total];
    let mut ids = Vec::with_capacity(RAM_LOAD_BATCH_ROWS);
    for (batch, dest) in table.chunks_mut(RAM_LOAD_BATCH_ROWS * stride).enumerate() {
        let first = (batch * RAM_LOAD_BATCH_ROWS) as u64;
        ids.clear();
        ids.extend(first..first + (dest.len() / stride) as u64);
        source.read(&ids, dest)?;
    }
    Ok(table)
}

#[cfg(test)]
#[path = "qwen4_exp_ple_io_tests.rs"]
mod tests;
