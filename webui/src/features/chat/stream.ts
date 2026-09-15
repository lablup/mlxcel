// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import type { ChatTurn } from './history';

export const MAX_TURN_CHARACTERS = 256_000;
export const MAX_PROMPT_CHARACTERS = 32_000;
const record = (value: unknown): Record<string, unknown> => {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new Error('Malformed chat stream.');
  return value as Record<string, unknown>;
};
function optionalText(value: unknown): string {
  if (value === undefined || value === null) return '';
  if (typeof value !== 'string') throw new Error('Malformed text delta.');
  return value;
}
function count(value: unknown): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) throw new Error('Malformed usage.');
  return value;
}

/** Accept one OpenAI frame. Never execute tool calls or infer token counts from words. */
export function appendFrame(turn: ChatTurn, data: string, elapsedMs: number): ChatTurn {
  if (data.length > MAX_TURN_CHARACTERS) throw new Error('Chat frame exceeds the display limit.');
  const frame = record(JSON.parse(data));
  if (frame.error !== undefined) throw new Error('The server reported a generation error. No automatic retry was made.');
  let next = { ...turn };
  if (frame.usage !== undefined && frame.usage !== null) {
    const usage = record(frame.usage);
    next = { ...next, usage: { prompt_tokens: count(usage.prompt_tokens), completion_tokens: count(usage.completion_tokens), total_tokens: count(usage.total_tokens) } };
  }
  if (!Array.isArray(frame.choices) || frame.choices.length > 1) throw new Error('Malformed chat choices.');
  for (const value of frame.choices) {
    const choice = record(value);
    if (choice.index !== 0) throw new Error('Unexpected chat choice index.');
    const delta = record(choice.delta);
    const content = optionalText(delta.content);
    const reasoning = optionalText(delta.reasoning_content ?? delta.reasoning);
    let tools = next.tools.map((tool) => ({ ...tool }));
    if (delta.tool_calls !== undefined && delta.tool_calls !== null) {
      if (!Array.isArray(delta.tool_calls) || delta.tool_calls.length > 32) throw new Error('Too many tool calls.');
      for (const item of delta.tool_calls) {
        const tool = record(item);
        const index = count(tool.index);
        if (index >= 32) throw new Error('Tool index exceeds the display limit.');
        const fn = tool.function === undefined ? {} : record(tool.function);
        const previous = tools.find((entry) => entry.index === index) ?? { index, id: '', name: '', arguments: '' };
        const merged = { index, id: previous.id + optionalText(tool.id), name: previous.name + optionalText(fn.name), arguments: previous.arguments + optionalText(fn.arguments) };
        tools = [...tools.filter((entry) => entry.index !== index), merged].sort((a, b) => a.index - b.index);
      }
    }
    const firstDelta = content.length > 0 || reasoning.length > 0 || tools.length > 0;
    next = { ...next, content: next.content + content, reasoning: next.reasoning + reasoning, tools, ttftMs: next.ttftMs ?? (firstDelta ? elapsedMs : null) };
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
      const reason = optionalText(choice.finish_reason);
      if (reason.length > 64) throw new Error('Invalid finish reason.');
      next = { ...next, finishReason: reason };
    }
  }
  if (next.content.length + next.reasoning.length + JSON.stringify(next.tools).length > MAX_TURN_CHARACTERS) throw new Error('Response exceeds the bounded display limit; generation stopped.');
  return next;
}

export function completeTurn(turn: ChatTurn, elapsedMs: number): ChatTurn {
  if (turn.finishReason === null || (!turn.content && !turn.reasoning && turn.tools.length === 0)) throw new Error('The server returned an empty or unfinished response.');
  return { ...turn, status: 'complete', elapsedMs };
}

export function buildMessages(systemPrompt: string, turns: readonly ChatTurn[]): unknown[] {
  const messages: unknown[] = systemPrompt ? [{ role: 'system', content: systemPrompt }] : [];
  for (const turn of turns) {
    messages.push({ role: 'user', content: turn.images.length ? [{ type: 'text', text: turn.prompt }, ...turn.images.map((image) => ({ type: 'image_url', image_url: { url: image.dataUrl } }))] : turn.prompt });
    // Do not replay partial turns, reasoning, or tool calls as fulfilled tool results.
    if (turn.status === 'complete' && turn.content) messages.push({ role: 'assistant', content: turn.content });
  }
  return messages;
}

export function decodeRate(turn: ChatTurn): number | null {
  if (turn.usage === null || turn.ttftMs === null || turn.elapsedMs === null || turn.elapsedMs <= turn.ttftMs || turn.usage.completion_tokens <= 1) return null;
  return (turn.usage.completion_tokens - 1) * 1000 / (turn.elapsedMs - turn.ttftMs);
}
