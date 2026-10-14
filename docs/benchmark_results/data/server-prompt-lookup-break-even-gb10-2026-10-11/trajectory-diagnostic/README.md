# Governor trajectory diagnostic

This is a non-performance diagnostic of source `a4bd791a` with the exact temporary environment-gated logging patch archived here. The coordinator built the warm `test-fast` workspace, ran the same copy-32 and chat-32 warmups as the acceptance driver, then one copy-256 request under gpu-lock. It reproduced the failing acceptance path exactly: 137 proposed tokens, 112 accepted tokens, and 21 draft rounds. Logging overhead and the test-fast profile mean the recorded timings are not acceptance measurements.

The temporary patch was reversed after capture; the two production files matched HEAD byte-for-byte. The environment file records the instrumented executable and patch hashes. The exact patch, launch script, warmups, client output, and server event log are preserved with checksums. The server was stopped by the owned launch script's EXIT trap.
