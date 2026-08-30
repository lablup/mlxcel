impl ModelProvider {
    /// Build a model-free provider that records the options dispatched by HTTP
    /// route tests and immediately returns an empty successful generation.
    pub(crate) fn recording_for_route_tests(
        options_tx: mpsc::Sender<ServerGenerateOptions>,
    ) -> Self {
        Self::recording_for_route_tests_with_admission(options_tx, false, usize::MAX)
    }

    pub(crate) fn recording_for_route_tests_with_admission(
        options_tx: mpsc::Sender<ServerGenerateOptions>,
        single_stream_admission: bool,
        max_queue_depth: usize,
    ) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<ModelRequest>();
        let loaded = Arc::new(AtomicBool::new(true));
        let batch_metrics = Arc::new(BatchMetrics::new());
        let batch_observability = Arc::new(BatchObservability::new());
        let worker_handle = thread::spawn(move || {
            while let Ok(request) = request_rx.recv() {
                match request {
                    ModelRequest::Generate {
                        options,
                        queue_reservation,
                        response_tx,
                        ..
                    } => {
                        drop(queue_reservation);
                        // A request carrying string stop sequences is answered
                        // as a generation that one of them ended (issue #1466),
                        // so route tests can assert the b10621
                        // `stop_type` / `stopping_word` mapping without a model.
                        // A request without `stop` keeps the empty canned
                        // generation every other route test depends on.
                        let stop_kind = options
                            .stop_sequences
                            .as_deref()
                            .and_then(<[String]>::first)
                            .map(|word| {
                                crate::server::model_provider::StopKind::Word(word.clone())
                            })
                            .unwrap_or(crate::server::model_provider::StopKind::Eos);
                        let _ = options_tx.send(options);
                        let _ = response_tx.send(GenerateEvent::Done(GenerationResult {
                            text: String::new(),
                            prompt_tokens: 0,
                            completion_tokens: 0,
                            generation_time_ms: 0,
                            prompt_eval_ms: 0,
                            generation_only_ms: 0,
                            finish_reason: "stop".to_string(),
                            stop_kind,
                            logprobs: None,
                            cached_tokens: 0,
                            // A canned id run so a route test can assert the
                            // `return_tokens` projection (#1477); the counts
                            // stay zero, as every other route test expects.
                            generated_token_ids: vec![9001, 9002],
                            structured_output: None,
                        }));
                    }
                    ModelRequest::PromptCacheWarmup { .. } => continue,
                    ModelRequest::Shutdown => break,
                }
            }
        });

        Self {
            request_tx,
            model_id: "route-test-model".to_string(),
            created_at: 0,
            loaded,
            snapshot_reuse_capable: Arc::new(AtomicBool::new(false)),
            chat_unavailable: Arc::new(AtomicBool::new(false)),
            batch_metrics,
            batch_observability,
            max_queue_depth,
            single_stream_queue_admission: Arc::new(AtomicBool::new(single_stream_admission)),
            sleeping: Arc::new(AtomicBool::new(false)),
            prompt_cache: None,
            prompt_tokenizer: None,
            decode_hang_timeout: DECODE_HANG_TIMEOUT,
            _worker_handle: worker_handle,
        }
    }

    pub(crate) fn chat_unavailable_for_route_tests() -> Self {
        Self::closed_worker_for_route_tests(true, false)
    }

    pub(crate) fn exited_chat_worker_for_route_tests() -> Self {
        Self::closed_worker_for_route_tests(false, true)
    }

    fn closed_worker_for_route_tests(chat_unavailable: bool, loaded: bool) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<ModelRequest>();
        drop(request_rx);
        let batch_metrics = Arc::new(BatchMetrics::new());
        Self {
            request_tx,
            model_id: "route-test-model".to_string(),
            created_at: 0,
            loaded: Arc::new(AtomicBool::new(loaded)),
            snapshot_reuse_capable: Arc::new(AtomicBool::new(false)),
            chat_unavailable: Arc::new(AtomicBool::new(chat_unavailable)),
            batch_metrics,
            batch_observability: Arc::new(BatchObservability::new()),
            max_queue_depth: usize::MAX,
            single_stream_queue_admission: Arc::new(AtomicBool::new(false)),
            sleeping: Arc::new(AtomicBool::new(false)),
            prompt_cache: None,
            prompt_tokenizer: None,
            decode_hang_timeout: DECODE_HANG_TIMEOUT,
            _worker_handle: thread::spawn(|| {}),
        }
    }

    /// Build a model-free provider whose streaming output is driven step by
    /// step from the test (#1444). Each `ModelRequest::Generate` consumes
    /// [`ScriptedStreamStep`]s from the returned handle until a `Finish`,
    /// then answers `GenerateEvent::Done`. The worker checks the request's
    /// cancellation token before every step, so a test can assert that
    /// `DELETE /v1/stream` (or a plain disconnect) aborts the generation.
    ///
    /// After cancellation the worker still consumes steps until the `Finish`
    /// that closes the scripted request, so a test must always send one.
    pub(crate) fn scripted_streaming_for_route_tests(
        options_tx: mpsc::Sender<ServerGenerateOptions>,
    ) -> (Self, ScriptedStreamHandle) {
        let (request_tx, request_rx) = mpsc::channel::<ModelRequest>();
        let (step_tx, step_rx) = mpsc::channel::<ScriptedStreamStep>();
        let cancellation_flags: Arc<std::sync::Mutex<Vec<Arc<AtomicBool>>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let flags_for_worker = cancellation_flags.clone();
        let loaded = Arc::new(AtomicBool::new(true));
        let batch_metrics = Arc::new(BatchMetrics::new());
        let batch_observability = Arc::new(BatchObservability::new());
        let worker_handle = thread::spawn(move || {
            while let Ok(request) = request_rx.recv() {
                match request {
                    ModelRequest::Generate {
                        options,
                        queue_reservation,
                        response_tx,
                        cancelled,
                        ..
                    } => {
                        drop(queue_reservation);
                        let _ = options_tx.send(options);
                        if let Ok(mut flags) = flags_for_worker.lock() {
                            flags.push(cancelled.clone());
                        }
                        let mut text = String::new();
                        let mut tokens_emitted = 0usize;
                        let mut aborted = false;
                        while let Ok(step) = step_rx.recv() {
                            if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                                aborted = true;
                            }
                            match step {
                                ScriptedStreamStep::Token(token) => {
                                    if aborted {
                                        continue;
                                    }
                                    text.push_str(&token);
                                    tokens_emitted += 1;
                                    let _ = response_tx.send(GenerateEvent::Token(token, TokenMeta::default()));
                                }
                                ScriptedStreamStep::Finish => break,
                            }
                        }
                        let _ = response_tx.send(GenerateEvent::Done(GenerationResult {
                            text,
                            prompt_tokens: 1,
                            completion_tokens: tokens_emitted,
                            generation_time_ms: 1,
                            prompt_eval_ms: 0,
                            generation_only_ms: 1,
                            finish_reason: "stop".to_string(),
                            stop_kind: crate::server::model_provider::StopKind::Eos,
                            logprobs: None,
                            cached_tokens: 0,
                            // A canned id run so a route test can assert the
                            // `return_tokens` projection (#1477); the counts
                            // stay zero, as every other route test expects.
                            generated_token_ids: vec![9001, 9002],
                            structured_output: None,
                        }));
                    }
                    ModelRequest::PromptCacheWarmup { .. } => continue,
                    ModelRequest::Shutdown => break,
                }
            }
        });
        let provider = Self {
            request_tx,
            model_id: "route-test-model".to_string(),
            created_at: 0,
            loaded,
            snapshot_reuse_capable: Arc::new(AtomicBool::new(false)),
            chat_unavailable: Arc::new(AtomicBool::new(false)),
            batch_metrics,
            batch_observability,
            max_queue_depth: usize::MAX,
            single_stream_queue_admission: Arc::new(AtomicBool::new(false)),
            sleeping: Arc::new(AtomicBool::new(false)),
            prompt_cache: None,
            prompt_tokenizer: None,
            decode_hang_timeout: DECODE_HANG_TIMEOUT,
            _worker_handle: worker_handle,
        };
        (
            provider,
            ScriptedStreamHandle {
                step_tx,
                cancellation_flags,
            },
        )
    }

}

/// One step of a scripted streaming generation (#1444).
pub(crate) enum ScriptedStreamStep {
    /// Emit one token to the live stream.
    Token(String),
    /// End the current request with a successful `Done`.
    Finish,
}

/// Test-side driver for [`ModelProvider::scripted_streaming_for_route_tests`].
pub(crate) struct ScriptedStreamHandle {
    step_tx: mpsc::Sender<ScriptedStreamStep>,
    /// One entry per dispatched `Generate`, in dispatch order: the request's
    /// scheduler cancellation token, so tests can assert whether a disconnect
    /// or a `DELETE /v1/stream` aborted the generation.
    cancellation_flags: Arc<std::sync::Mutex<Vec<Arc<AtomicBool>>>>,
}

impl ScriptedStreamHandle {
    pub(crate) fn token(&self, token: &str) {
        let _ = self.step_tx.send(ScriptedStreamStep::Token(token.to_string()));
    }

    pub(crate) fn finish(&self) {
        let _ = self.step_tx.send(ScriptedStreamStep::Finish);
    }

    /// The cancellation token of the `index`-th dispatched generation.
    pub(crate) fn cancellation_flag(&self, index: usize) -> Option<Arc<AtomicBool>> {
        self.cancellation_flags.lock().ok()?.get(index).cloned()
    }
}
