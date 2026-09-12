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

import React from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

type Capability = {
  label: string;
  detail: string;
};

const capabilities: Capability[] = [
  {
    label: 'Model-free startup',
    detail: 'Start the control plane first, then discover, download, and load models explicitly.',
  },
  {
    label: 'Offline bundle',
    detail: 'Every script, style, and asset is served by mlxcel from the same local origin.',
  },
  {
    label: 'Native-feeling shell',
    detail: 'macOS-style glass navigation, keyboard focus, and readable content surfaces.',
  },
];

function App(): React.JSX.Element {
  return (
    <main className="shell" aria-labelledby="app-title">
      <aside className="sidebar" aria-label="Primary navigation">
        <div className="brand" aria-hidden="true">mlx</div>
        <nav>
          <a aria-current="page" href="#models">Models</a>
          <a href="#chat">Chat</a>
          <a href="#activity">Activity</a>
          <a href="#settings">Settings</a>
        </nav>
        <p className="status">Local WebUI shell ready</p>
      </aside>
      <section className="content">
        <header className="toolbar">
          <p>Bundled WebUI</p>
          <span>Waiting for server adapters</span>
        </header>
        <div className="hero">
          <p className="eyebrow">Issue #1836 foundation</p>
          <h1 id="app-title">mlxcel WebUI loads from a reproducible offline bundle.</h1>
          <p className="lead">This minimal shell verifies the React, TypeScript, Vite, and Rust embedding pipeline before model controls land in later issues.</p>
          <div className="cards" aria-label="Planned capabilities">
            {capabilities.map((item) => (
              <article key={item.label}>
                <h2>{item.label}</h2>
                <p>{item.detail}</p>
              </article>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}

const rootElement = document.getElementById('root');
if (rootElement === null) {
  throw new Error('Missing #root element for mlxcel WebUI.');
}

createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
