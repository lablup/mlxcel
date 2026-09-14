// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { createContext } from 'react';

// alpha.19 popups portal to body, outside native showModal's interactive top layer.
export const NativeModalContext = createContext(false);
