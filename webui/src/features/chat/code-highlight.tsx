// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { memo, type ReactNode } from 'react';
const SUPPORTED = new Set(['js', 'javascript', 'ts', 'typescript', 'json', 'py', 'python', 'rs', 'rust', 'sh', 'bash']);
const KEYWORDS = new Set('const let var function return if else for while class import from export default async await def with as try catch except raise fn pub use impl struct enum match mut self true false null None True False'.split(' '));
// Bounded lexical highlighting, not a language grammar. Lazy-loaded only for
// complete code blocks; memoization prevents retokenizing preceding messages.
export default memo(function HighlightedCode({ code, language }: { code: string; language: string }) {
  if (!SUPPORTED.has(language.toLowerCase()) || code.length > 32_768) return <code>{code}</code>;
  const parts: ReactNode[] = [];
  const tokens = /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\b[A-Za-z_]\w*\b|\b\d+(?:\.\d+)?\b)/g;
  let offset = 0;
  let count = 0;
  for (const match of code.matchAll(tokens)) {
    if (++count > 4096) break;
    parts.push(code.slice(offset, match.index));
    const token = match[0];
    const kind = /^['"]/.test(token) ? 'string' : /^\d/.test(token) ? 'number' : KEYWORDS.has(token) ? 'keyword' : undefined;
    parts.push(kind ? <span key={match.index} className={`chat-code-${kind}`}>{token}</span> : token);
    offset = match.index + token.length;
  }
  parts.push(code.slice(offset));
  return <code>{parts}</code>;
});
