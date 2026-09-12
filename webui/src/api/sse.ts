// Copyright 2026 Lablup Inc.
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

export interface SseMessage {
  readonly event: string;
  readonly data: string;
  readonly id: string | null;
  readonly retry: number | null;
}

export interface SseParserOptions {
  readonly maxFrameBytes?: number;
  readonly onMessage: (message: SseMessage) => void;
  readonly onDone?: () => void;
}

const DEFAULT_MAX_FRAME_BYTES = 1024 * 1024;

export class SseParser {
  private readonly decoder = new TextDecoder('utf-8');
  private readonly maxFrameBytes: number;
  private readonly onMessage: (message: SseMessage) => void;
  private readonly onDone?: () => void;
  private buffered = '';
  private eventName = '';
  private dataLines: string[] = [];
  private id: string | null = null;
  private retry: number | null = null;
  private frameBytes = 0;
  private ended = false;

  constructor(options: SseParserOptions) {
    this.maxFrameBytes = options.maxFrameBytes ?? DEFAULT_MAX_FRAME_BYTES;
    this.onMessage = options.onMessage;
    this.onDone = options.onDone;
  }

  push(chunk: Uint8Array): void {
    if (this.ended) return;
    this.frameBytes += chunk.byteLength;
    if (this.frameBytes > this.maxFrameBytes) throw new Error('SSE frame exceeded the configured byte limit.');
    this.buffered += this.decoder.decode(chunk, { stream: true });
    this.drainLines(false);
  }

  close(): void {
    if (this.ended) return;
    const tail = this.decoder.decode();
    if (tail.length > 0) this.buffered += tail;
    this.drainLines(true);
    if (this.buffered.length > 0) {
      this.consumeLine(this.buffered);
      this.buffered = '';
    }
    this.dispatch();
  }

  private drainLines(final: boolean): void {
    while (this.buffered.length > 0) {
      const lf = this.buffered.indexOf('\n');
      const cr = this.buffered.indexOf('\r');
      let index: number;
      if (lf === -1) index = cr;
      else if (cr === -1) index = lf;
      else index = Math.min(lf, cr);
      if (index === -1) break;
      if (this.buffered[index] === '\r' && this.buffered[index + 1] === undefined && !final) break;
      const line = this.buffered.slice(0, index);
      const next = this.buffered[index] === '\r' && this.buffered[index + 1] === '\n' ? index + 2 : index + 1;
      this.buffered = this.buffered.slice(next);
      this.consumeLine(line);
    }
    if (final && this.buffered.length === 0) this.dispatch();
  }

  private consumeLine(line: string): void {
    if (line.length === 0) {
      this.dispatch();
      return;
    }
    if (line.startsWith(':')) return;
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    const raw = colon === -1 ? '' : line.slice(colon + 1);
    const value = raw.startsWith(' ') ? raw.slice(1) : raw;
    if (field === 'event') this.eventName = value;
    else if (field === 'data') this.dataLines.push(value);
    else if (field === 'id') this.id = value;
    else if (field === 'retry' && /^\d+$/.test(value)) this.retry = Number(value);
  }

  private dispatch(): void {
    if (this.dataLines.length === 0) {
      this.eventName = '';
      this.retry = null;
      this.frameBytes = 0;
      return;
    }
    const data = this.dataLines.join('\n');
    const event = this.eventName.length === 0 ? 'message' : this.eventName;
    this.eventName = '';
    this.dataLines = [];
    this.frameBytes = 0;
    if (data === '[DONE]') {
      this.ended = true;
      this.onDone?.();
      return;
    }
    this.onMessage({ event, data, id: this.id, retry: this.retry });
    this.retry = null;
  }
}

export function splitUtf8(text: string, chunkSizes: readonly number[]): Uint8Array[] {
  const bytes = new TextEncoder().encode(text);
  const chunks: Uint8Array[] = [];
  let offset = 0;
  let index = 0;
  while (offset < bytes.length) {
    const requested = chunkSizes[index % chunkSizes.length] ?? bytes.length;
    const size = Math.max(1, Math.min(requested, bytes.length - offset));
    chunks.push(bytes.slice(offset, offset + size));
    offset += size;
    index += 1;
  }
  return chunks;
}
