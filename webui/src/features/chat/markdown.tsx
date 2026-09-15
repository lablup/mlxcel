// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { lazy, memo, Suspense, useState, type ReactNode } from 'react';
import { Button } from '../../design-system/primitives';
const HighlightedCode = lazy(() => import('./code-highlight'));
const MAX_MARKDOWN_CHARS = 65_536;
const MAX_BLOCKS = 512;

// A deliberately small Markdown subset, not CommonMark: paragraphs, headings,
// unordered lists, fenced code, inline code/emphasis and HTTP(S) links. HTML is
// always escaped text. Images never render, irrespective of their URL scheme.
function inline(text: string): ReactNode[] {
  // Bound each token scan independently: adversarial unmatched delimiters must
  // not turn a long generation into quadratic work across the whole message.
  const chunks: ReactNode[] = [];
  for (let offset = 0; offset < text.length; offset += 1024) chunks.push(<span key={offset}>{inlineChunk(text.slice(offset, offset + 1024))}</span>);
  return chunks;
}
function inlineChunk(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const pattern = /!\[([^\]\n]*)\]\(([^)\n]*)\)|\[([^\]\n]*)\]\(([^)\n]*)\)|`([^`\n]+)`|\*\*([^*\n]+)\*\*|\*([^*\n]+)\*/g;
  let offset = 0;
  for (const match of text.matchAll(pattern)) {
    parts.push(text.slice(offset, match.index));
    const key = match.index;
    if (match[1] !== undefined) parts.push(<span key={key}>[Image omitted: {match[1] || 'image'}]</span>);
    else if (match[3] !== undefined) {
      let href: string | undefined;
      try {
        const url = new URL(match[4]);
        if (/^https?:$/.test(url.protocol) && !url.username && !url.password) href = url.href;
      } catch { /* Unsafe, relative and malformed links remain inert text. */ }
      parts.push(href ? <a key={key} href={href} target="_blank" rel="noopener noreferrer">{match[3]}</a> : <span key={key}>{match[3]}</span>);
    } else if (match[5] !== undefined) parts.push(<code key={key}>{match[5]}</code>);
    else if (match[6] !== undefined) parts.push(<strong key={key}>{match[6]}</strong>);
    else parts.push(<em key={key}>{match[7]}</em>);
    offset = match.index + match[0].length;
  }
  parts.push(text.slice(offset));
  return parts;
}
const CodeBlock = memo(function CodeBlock({ code, language, complete }: { code: string; language: string; complete: boolean }) {
  const [copyState, setCopyState] = useState('Copy code');
  async function copy() {
    try { await navigator.clipboard.writeText(code); setCopyState('Copied'); }
    catch { setCopyState('Copy failed'); }
  }
  return <div className="chat-code-block"><div className="chat-code-toolbar"><span>{language || 'Plain text'}</span><Button tone="ghost" onClick={() => void copy()}>{copyState}</Button></div><pre>{complete ? <Suspense fallback={<code>{code}</code>}><HighlightedCode code={code} language={language} /></Suspense> : <code>{code}</code>}</pre></div>;
});
export const SafeMarkdown = memo(function SafeMarkdown({ text, streaming = false }: { text: string; streaming?: boolean }) {
  const lines = text.slice(0, MAX_MARKDOWN_CHARS).split('\n');
  const blocks: ReactNode[] = [];
  let cursor = 0;
  while (cursor < lines.length && blocks.length < MAX_BLOCKS) {
    const start = cursor;
    const line = lines[cursor++];
    const fence = /^ {0,3}(`{3,}|~{3,})([\w+-]*)\s*$/.exec(line);
    if (fence) {
      const code: string[] = [];
      let closed = false;
      while (cursor < lines.length) {
        const candidate = lines[cursor++];
        const trimmed = candidate.trim();
        if (trimmed.length >= fence[1].length && [...trimmed].every(char => char === fence[1][0])) { closed = true; break; }
        code.push(candidate);
      }
      blocks.push(<CodeBlock key={start} code={code.join('\n')} language={fence[2].slice(0, 40)} complete={closed || (!streaming && text.length <= MAX_MARKDOWN_CHARS)} />);
    } else if (/^#{1,6} /.test(line)) {
      blocks.push(<p className="chat-markdown-heading" key={start}><strong>{inline(line.replace(/^#{1,6} /, ''))}</strong></p>);
    } else if (/^[-*] /.test(line)) {
      const items = [line.slice(2)];
      while (cursor < lines.length && /^[-*] /.test(lines[cursor]) && items.length < MAX_BLOCKS) items.push(lines[cursor++].slice(2));
      blocks.push(<ul key={start}>{items.map((item, index) => <li key={index}>{inline(item)}</li>)}</ul>);
    } else if (line.trim()) blocks.push(<p key={start}>{inline(line)}</p>);
  }
  return <div className="chat-markdown">{blocks}{(text.length > MAX_MARKDOWN_CHARS || cursor < lines.length) && <p>Display truncated for performance. Copy the message to read its full text.</p>}</div>;
});
