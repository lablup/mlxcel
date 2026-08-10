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
                        let _ = options_tx.send(options);
                        let _ = response_tx.send(GenerateEvent::Done(GenerationResult {
                            text: String::new(),
                            prompt_tokens: 0,
                            completion_tokens: 0,
                            generation_time_ms: 0,
                            prompt_eval_ms: 0,
                            generation_only_ms: 0,
                            finish_reason: "stop".to_string(),
                            logprobs: None,
                            cached_tokens: 0,
                            structured_output: None,
                        }));
                    }
                    ModelRequest::Shutdown => break,
                }
            }
        });

        Self {
            request_tx,
            model_id: "route-test-model".to_string(),
            created_at: 0,
            loaded,
            batch_metrics,
            batch_observability,
            max_queue_depth,
            single_stream_queue_admission: Arc::new(AtomicBool::new(single_stream_admission)),
            prompt_cache: None,
            prompt_tokenizer: None,
            decode_hang_timeout: DECODE_HANG_TIMEOUT,
            _worker_handle: worker_handle,
        }
    }
}
