// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useState } from 'react';

// A native disclosure whose content is built only while it is open. Activity
// re-renders on every 2 s runtime poll, so a closed disclosure must cost nothing:
// `children` is a render function that is not called until the reader opens it.
// The open state mirrors the native `toggle` event; the element stays uncontrolled.
export function Disclosure(props: { summary: React.ReactNode; className?: string; testId?: string; children: () => React.ReactNode }): React.JSX.Element {
  const [open, setOpen] = useState(false);
  return <details className={props.className} data-testid={props.testId} onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary>{props.summary}</summary>
    {open ? props.children() : null}
  </details>;
}
