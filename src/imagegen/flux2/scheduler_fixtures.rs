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

//! Upstream `FlowMatchEulerDiscreteScheduler.get_timesteps_and_sigmas` tables
//! (mlx-vlm 0.7.8, git 73d366ec, mlx 0.32.4 on the CUDA backend) for 4 and 50
//! steps at image sequence lengths 1024 (interpolated `mu`) and 5000 (linear
//! `mu`). Regenerate with `get_timesteps_and_sigmas(image_seq_len=L,
//! num_inference_steps=N)` and store the f32 bit patterns.

/// `(steps, image_seq_len, timestep bits, sigma bits)`.
pub(super) type Case = (u32, i32, &'static [u32], &'static [u32]);

pub(super) const CASES: &[Case] = &[
    (
        4,
        1024,
        &[0x447a0000, 0x446f8577, 0x445cfed7, 0x44335fc8],
        &[0x3f800000, 0x3f754515, 0x3f624ca3, 0x3f37addb, 0x00000000],
    ),
    (
        4,
        5000,
        &[0x447a0000, 0x44653ce4, 0x444495e8, 0x4409bb21],
        &[0x3f800000, 0x3f6abd54, 0x3f494dba, 0x3f0d0959, 0x00000000],
    ),
    (
        50,
        1024,
        &[
            0x447a0000, 0x447912be, 0x44781d77, 0x44771fbf, 0x44761926, 0x44750933, 0x4473ef64,
            0x4472cb2b, 0x44719bf4, 0x4470611c, 0x446f19f3, 0x446dc5c0, 0x446c63b4, 0x446af2f2,
            0x4469728d, 0x4467e180, 0x44663ead, 0x446488e2, 0x4462becc, 0x4460defa, 0x445ee7d5,
            0x445cd7a0, 0x445aac70, 0x44586423, 0x4455fc62, 0x44537290, 0x4450c3c4, 0x444decbf,
            0x444ae9dd, 0x4447b706, 0x44444f9c, 0x4440ae62, 0x443ccd68, 0x4438a5df, 0x44342ffe,
            0x442f62c7, 0x442a33d0, 0x442496f1, 0x441e7df0, 0x4417d7fb, 0x4410910f, 0x4408913d,
            0x43ff7714, 0x43ebd932, 0x43d5f1a6, 0x43bd53a0, 0x43a17594, 0x4381a727, 0x433a047d,
            0x42c9542b,
        ],
        &[
            0x3f800000, 0x3f7f0d0c, 0x3f7e11e2, 0x3f7d0e13, 0x3f7c012d, 0x3f7aeab3, 0x3f79ca21,
            0x3f789ee4, 0x3f776866, 0x3f762600, 0x3f74d6fd, 0x3f737aa0, 0x3f721014, 0x3f709679,
            0x3f6f0cda, 0x3f6d722d, 0x3f6bc54d, 0x3f6a0500, 0x3f682fec, 0x3f664496, 0x3f64415d,
            0x3f62247b, 0x3f5febf8, 0x3f5d95a5, 0x3f5b1f1d, 0x3f5885b2, 0x3f55c66b, 0x3f52ddf3,
            0x3f4fc890, 0x3f4c8212, 0x3f4905be, 0x3f454e37, 0x3f415568, 0x3f3d1459, 0x3f388310,
            0x3f339859, 0x3f2e4989, 0x3f288a2e, 0x3f224bb6, 0x3f1b7ce8, 0x3f140947, 0x3f0bd84f,
            0x3f02cc54, 0x3ef1823f, 0x3edb141f, 0x3ec1ded9, 0x3ea55596, 0x3e84c3bd, 0x3e3e7b61,
            0x3dce2922, 0x00000000,
        ],
    ),
    (
        50,
        5000,
        &[
            0x447a0000, 0x44789f11, 0x4477338d, 0x4475bcf3, 0x44743ac6, 0x4472ac74, 0x4471116e,
            0x446f690f, 0x446db2b4, 0x446beda7, 0x446a1926, 0x44683465, 0x44663e87, 0x446436a0,
            0x44621bb3, 0x445fecaf, 0x445da86f, 0x445b4db7, 0x4458db34, 0x44564f74, 0x4453a8ea,
            0x4450e5e6, 0x444e0492, 0x444b02f1, 0x4447ded7, 0x444495e8, 0x44412589, 0x443d8ae6,
            0x4439c2dd, 0x4435ca03, 0x44319c8c, 0x442d3647, 0x4428928e, 0x4423ac35, 0x441e7d76,
            0x4418ffe0, 0x44132c32, 0x440cfa45, 0x440660e0, 0x43feab13, 0x43ef98a6, 0x43df6f2d,
            0x43ce0f43, 0x43bb5494, 0x43a714f1, 0x43911f03, 0x43727178, 0x433e3ac7, 0x4304f5be,
            0x428bb998,
        ],
        &[
            0x3f800000, 0x3f7e9699, 0x3f7d225b, 0x3f7ba2c4, 0x3f7a1752, 0x3f787f71, 0x3f76da8d,
            0x3f7527ff, 0x3f73671f, 0x3f719732, 0x3f6fb773, 0x3f6dc70f, 0x3f6bc526, 0x3f69b0c5,
            0x3f6788e8, 0x3f654c7a, 0x3f62fa4d, 0x3f60911e, 0x3f5e0f91, 0x3f5b742d, 0x3f58bd5a,
            0x3f55e95e, 0x3f52f658, 0x3f4fe23e, 0x3f4caad8, 0x3f494dba, 0x3f45c83a, 0x3f421773,
            0x3f3e382e, 0x3f3a26ed, 0x3f35dfcb, 0x3f315e7e, 0x3f2c9e44, 0x3f2799d0, 0x3f224b39,
            0x3f1cabe7, 0x3f16b46d, 0x3f105c70, 0x3f099a7f, 0x3f0263e1, 0x3ef558ba, 0x3ee4cbf5,
            0x3ed3014b, 0x3ebfd389, 0x3eab177e, 0x3e949aa3, 0x3e78430a, 0x3e42cb8c, 0x3e0826a6,
            0x3d8f1410, 0x00000000,
        ],
    ),
];
