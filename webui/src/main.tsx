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
// ui-common 0.2 (Astryx): the layer order, reset and component styles. index.html links
// public/layer-order.css first.
import './ui-common-entry.css';
// The one theme stylesheet, directly after the ui-common entry (never imported by a
// component): the pre-built family themes and the product-only theme tokens.
import './design-system/themes/index.css';
// Every other product stylesheet, in @layer components and in a fixed order.
import './app.css';
import { App } from './app';
import { WebUiProvider } from './state';

const rootElement = document.getElementById('root');
if (rootElement === null) {
  throw new Error('Missing #root element for mlxcel WebUI.');
}

createRoot(rootElement).render(
  <React.StrictMode>
    <WebUiProvider>
      <App />
    </WebUiProvider>
  </React.StrictMode>,
);
