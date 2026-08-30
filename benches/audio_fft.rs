use std::f64::consts::PI;
use std::time::Duration;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

#[allow(dead_code)]
#[path = "../src/audio/fft.rs"]
mod fft;

fn deterministic_frame(len: usize) -> Vec<f64> {
    (0..len)
        .map(|index| {
            let t = index as f64 / len as f64;
            0.3 * (2.0 * PI * 3.0 * t).sin()
                + 0.2 * (2.0 * PI * 17.0 * t).cos()
                + 0.05 * (2.0 * PI * 61.0 * t).sin()
        })
        .collect()
}

fn dft_magnitude(input: &[f64], num_bins: usize) -> Vec<f64> {
    let n = input.len();
    let mut magnitudes = Vec::with_capacity(num_bins);
    for k in 0..num_bins {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (t, &sample) in input.iter().enumerate() {
            let angle = -2.0 * PI * k as f64 * t as f64 / n as f64;
            re += sample * angle.cos();
            im += sample * angle.sin();
        }
        magnitudes.push(re.hypot(im));
    }
    magnitudes
}

fn bench_audio_real_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_real_fft_magnitude");
    group.sample_size(10);
    for len in [400usize, 512, 1024] {
        let input = deterministic_frame(len);
        let bins = len / 2 + 1;
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(format!("dft_len_{len}"), |b| {
            b.iter(|| dft_magnitude(black_box(&input), black_box(bins)))
        });
        group.bench_function(format!("fft_len_{len}"), |b| {
            b.iter(|| fft::real_fft_magnitude(black_box(&input), black_box(bins)))
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(300));
    targets = bench_audio_real_fft
}
criterion_main!(benches);
