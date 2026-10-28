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

//! Literals are upstream's own values (`mlx_vlm.models.ideogram4.scheduler`
//! and `statistics.NormalDist().inv_cdf`, Python 3.12), printed with `repr`.

use super::*;

/// Relative tolerance for an f64 port of f64 arithmetic.
const TOL: f64 = 1e-12;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= TOL * b.abs().max(1e-300) || a == b
}

#[test]
fn inverse_normal_cdf_matches_python() {
    for (p, want) in [
        (1e-6, -4.753424308822899),
        (0.5, 0.0),
        (0.999999, 4.753424308817089),
        (0.025, -1.9599639845400538),
        (0.3, -0.5244005127080407),
        (0.07, -1.4757910281791702),
        (0.93, 1.4757910281791708),
        (1e-12, -7.034483825301132),
        (1.0 - 1e-12, 7.0344869100478356),
        (1e-300, -37.0470962993612),
    ] {
        let got = inv_normal_cdf(p);
        assert!(close(got, want), "inv_cdf({p}) = {got:?}, want {want:?}");
    }
}

fn check(name: &str, width: u32, height: u32, want: &(f64, impl AsRef<[f64]>)) {
    let preset = get_preset(Some(name)).unwrap();
    let schedule = get_schedule_for_resolution(height, width, preset.mu, preset.std);
    assert!(
        close(schedule.mean, want.0),
        "{name} mean {}",
        schedule.mean
    );
    let intervals = make_step_intervals(preset.num_steps).unwrap();
    let want = want.1.as_ref();
    assert_eq!(intervals.len(), want.len());
    for (i, (&x, &w)) in intervals.iter().zip(want).enumerate() {
        let got = schedule.at(x);
        assert!(
            close(got, w),
            "{name} {width}x{height} [{i}] = {got:?}, want {w:?}"
        );
    }
}

#[test]
fn every_preset_schedule_matches_upstream_at_512_and_1024x768() {
    check("V4_DEFAULT_20", 512, 512, &V4_DEFAULT_20_512X512);
    check("V4_DEFAULT_20", 1024, 768, &V4_DEFAULT_20_1024X768);
    check("V4_QUALITY_48", 512, 512, &V4_QUALITY_48_512X512);
    check("V4_QUALITY_48", 1024, 768, &V4_QUALITY_48_1024X768);
    check("V4_TURBO_12", 512, 512, &V4_TURBO_12_512X512);
    check("V4_TURBO_12", 1024, 768, &V4_TURBO_12_1024X768);
}

#[test]
fn presets_match_upstream() {
    let quality = get_preset(Some("V4_QUALITY_48")).unwrap();
    assert_eq!((quality.num_steps, quality.mu, quality.std), (48, 0.0, 1.5));
    assert_eq!(quality.guidance_schedule[..4], [3.0, 3.0, 3.0, 7.0]);
    let default = get_preset(None).unwrap();
    assert_eq!(default.name, "V4_DEFAULT_20");
    assert_eq!(
        (default.num_steps, default.mu, default.std),
        (20, 0.0, 1.75)
    );
    assert_eq!(default.guidance_schedule[..3], [3.0, 3.0, 7.0]);
    let turbo = get_preset(Some("V4_TURBO_12")).unwrap();
    assert_eq!((turbo.num_steps, turbo.mu, turbo.std), (12, 0.5, 1.75));
    assert_eq!(turbo.guidance_schedule[..2], [3.0, 7.0]);
    for preset in PRESETS {
        assert_eq!(preset.guidance_schedule.len(), preset.num_steps as usize);
        assert_eq!(*preset.guidance_schedule.last().unwrap(), 7.0);
    }
}

#[test]
fn an_unknown_preset_lists_the_valid_names() {
    let err = get_preset(Some("V4_FAST")).unwrap_err().to_string();
    assert_eq!(
        err,
        "Unknown Ideogram 4 sampler preset 'V4_FAST'; expected one of \
         ['V4_DEFAULT_20', 'V4_QUALITY_48', 'V4_TURBO_12']"
    );
}

#[test]
fn schedule_endpoints_clamp_to_the_log_snr_range() {
    let s = LogitNormalSchedule::new(0.0, 1.0);
    assert!(close(s.at(0.0), 0.9994472213630764));
    assert!(close(s.at(1.0), 0.00012339457598623172));
    assert!(make_step_intervals(0).is_err());
    assert_eq!(make_step_intervals(2).unwrap(), vec![0.0, 0.5, 1.0]);
}

/// `V4_DEFAULT_20` at 512x512 (width x height): mean 0.0.
const V4_DEFAULT_20_512X512: (f64, [f64; 21]) = (
    0.0,
    [
        0.9994472213630764,
        0.9467730137798347,
        0.9040203116776014,
        0.8598155014954277,
        0.8134882372155796,
        0.7650119985395049,
        0.7145734161507626,
        0.6624677500192346,
        0.6090587477974414,
        0.5547563522400429,
        0.5,
        0.44524364775995684,
        0.3909412522025587,
        0.33753224998076536,
        0.28542658384923736,
        0.23498800146049514,
        0.1865117627844204,
        0.14018449850457226,
        0.09597968832239856,
        0.05322698622016553,
        0.00012339457598623172,
    ],
);

/// `V4_DEFAULT_20` at 1024x768 (width x height): mean 0.5493061443340549.
const V4_DEFAULT_20_1024X768: (f64, [f64; 21]) = (
    0.5493061443340549,
    [
        0.9994472213630764,
        0.9112656758933575,
        0.8446719631524287,
        0.779791592012149,
        0.7157610275908437,
        0.6527277165708631,
        0.5910710372692316,
        0.5312105367928567,
        0.4735375705926895,
        0.418386914646685,
        0.3660254037844386,
        0.31664926627197687,
        0.27038629289201466,
        0.22730055913951108,
        0.18739795520292113,
        0.1506306643090891,
        0.11689755057061058,
        0.08603291714146555,
        0.057756851151794075,
        0.03143785242650077,
        0.00012339457598623172,
    ],
);

/// `V4_QUALITY_48` at 512x512 (width x height): mean 0.0.
const V4_QUALITY_48_512X512: (f64, [f64; 49]) = (
    0.0,
    [
        0.9994472213630764,
        0.9550086950549566,
        0.930700290830805,
        0.9089747864940443,
        0.8883990232726837,
        0.8684407842520157,
        0.8488390039861626,
        0.829446372297313,
        0.8101726424482978,
        0.7909596795915816,
        0.7717689460566649,
        0.7525746112288307,
        0.733359480957885,
        0.7141124549056344,
        0.6948268641336233,
        0.6754993413955681,
        0.6561290270823624,
        0.6367169938254085,
        0.6172658175102249,
        0.5977792485220039,
        0.5782619527741416,
        0.5587193018409158,
        0.5391571977242908,
        0.5195819217900834,
        0.5,
        0.48041807820991655,
        0.4608428022757093,
        0.4412806981590842,
        0.4217380472258583,
        0.4022207514779961,
        0.3827341824897751,
        0.36328300617459164,
        0.3438709729176376,
        0.324500658604432,
        0.30517313586637673,
        0.28588754509436576,
        0.26664051904211516,
        0.24742538877116937,
        0.22823105394333498,
        0.20904032040841836,
        0.1898273575517021,
        0.170553627702687,
        0.15116099601383726,
        0.13155921574798424,
        0.11160097672731639,
        0.0910252135059556,
        0.06929970916919492,
        0.044991304945043664,
        0.00012339457598623172,
    ],
);

/// `V4_QUALITY_48` at 1024x768 (width x height): mean 0.5493061443340549.
const V4_QUALITY_48_1024X768: (f64, [f64; 49]) = (
    0.5493061443340549,
    [
        0.9994472213630764,
        0.9245575408208764,
        0.8857646460098649,
        0.8521890969440014,
        0.8213007364334304,
        0.7921503449608915,
        0.7642670955410288,
        0.7373814118356125,
        0.7113246436540124,
        0.6859847173964682,
        0.6612838110550499,
        0.6371660306748084,
        0.6135901209723884,
        0.5905249155060258,
        0.5679463720909702,
        0.5458355722017516,
        0.5241773310752151,
        0.5029592081885345,
        0.4821707879258589,
        0.461803147093531,
        0.4418484543271519,
        0.42229966416792797,
        0.4031502799607878,
        0.38439416718337616,
        0.3660254037844386,
        0.3480381574525807,
        0.33042658197765595,
        0.31318472632929806,
        0.29630645095212604,
        0.2797853461661277,
        0.2636146474930745,
        0.2477871421656226,
        0.23229505989224075,
        0.21712993890718213,
        0.2022824549996347,
        0.18774219583234808,
        0.1734973540759316,
        0.1595342982193556,
        0.14583695463610213,
        0.13238588912548277,
        0.11915689070569313,
        0.10611868969924232,
        0.09322907528416369,
        0.08042781393422105,
        0.06762247936413401,
        0.054656151047977164,
        0.04121744421719131,
        0.02647926005700385,
        0.00012339457598623172,
    ],
);

/// `V4_TURBO_12` at 512x512 (width x height): mean 0.5.
const V4_TURBO_12_512X512: (f64, [f64; 13]) = (
    0.5,
    [
        0.9994472213630764,
        0.8721651634598191,
        0.767274993358841,
        0.6638183854715942,
        0.5631048902155242,
        0.4671099845279746,
        0.3775406687981454,
        0.295618635591388,
        0.22204773693593916,
        0.15704819366852485,
        0.1003819620521953,
        0.051162062458179114,
        0.00012339457598623172,
    ],
);

/// `V4_TURBO_12` at 1024x768 (width x height): mean 1.049306144334055.
const V4_TURBO_12_1024X768: (f64, [f64; 13]) = (
    1.049306144334055,
    [
        0.9994472213630764,
        0.7975309452797447,
        0.6555852156670894,
        0.532715920154144,
        0.42664974924877697,
        0.3360255537218273,
        0.2593583624083433,
        0.19504503764850378,
        0.14147668336172592,
        0.09711816907264803,
        0.06052334083962618,
        0.030191275562659214,
        0.00012339457598623172,
    ],
);
