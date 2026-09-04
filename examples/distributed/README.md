# Distributed config examples

`pipeline_remote_2node_tcp.toml` and `pipeline_remote_2node_thunderbolt.toml`
are checked-in `--distributed-config` templates for a two-node remote pipeline
(coordinator + two stages, over plain TCP and over a Thunderbolt Bridge
network respectively). Copy or edit one to match your node addresses before
starting the servers; see [`docs/distributed.md`](../../docs/distributed.md)
for the full workflow.

`scripts/benchmark_pipeline_remote_rollout.sh write-config` generates
`generated_*.toml` variants of the same shape in this directory at runtime,
filled from its `CLUSTER_NAME` / `TRANSPORT_BACKEND` / `*_ADDR` environment
variables. Those generated files are not checked in.
