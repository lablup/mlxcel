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

Both templates reference `secret_file = "cluster.secret"`: the pre-shared
cluster secret every pipeline role must hold (issue #2637). Create it next to
the TOML with `openssl rand -hex 32 > cluster.secret && chmod 600
cluster.secret`, copy it to every host with the TOML, or override the path with
`--cluster-secret-file`, `MLXCEL_CLUSTER_SECRET_FILE`, or `MLXCEL_CLUSTER_SECRET`.
Never commit a real secret; `cluster.secret` is a local file only.
