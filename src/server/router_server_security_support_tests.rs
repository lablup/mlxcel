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

use std::io;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub(super) struct SharedBufWriter(pub(super) Arc<Mutex<Vec<u8>>>);

pub(super) struct SharedBufGuard(Arc<Mutex<Vec<u8>>>);

impl io::Write for SharedBufGuard {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .expect("log buffer lock")
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedBufWriter {
    type Writer = SharedBufGuard;

    fn make_writer(&'a self) -> Self::Writer {
        SharedBufGuard(self.0.clone())
    }
}
