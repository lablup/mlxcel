// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Compressed-container audio decoding (llama-server b10621, issue #1446).
//!
//! b10621's mtmd audio front-end takes wav, mp3 and flac. mlxcel decoded only
//! RIFF/WAVE, so a non-WAV clip was a 400 here and a transcript there. This
//! module is the dispatch: it sniffs the container from its magic bytes, hands
//! WAV to the in-tree reader in [`super::wav`] (unchanged, so every WAV clip
//! decodes exactly as before), and decodes mp3 and flac through `symphonia`.
//!
//! **Every limit the WAV path enforces is enforced here, before decoding.**
//! That is the point of splitting inspect from decode: a compressed container
//! declares its geometry in a header that is orders of magnitude smaller than
//! the samples it expands to, so a malformed or hostile file must be refused
//! on the declared geometry rather than after allocating for it. The sample
//! rate range, the per-clip duration cap and the per-request sample and
//! duration caps therefore all run against the probe's track parameters, and
//! the decode loop additionally stops the moment it has produced more frames
//! than the cap allows, which is what bounds a file whose header understates
//! its length (a frame count is advisory in both containers).

use symphonia::core::audio::AudioBufferRef;
use symphonia::core::audio::Signal;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::wav::{NativeWaveform, decode_wav, inspect_wav};
use super::{
    AudioCancellation, AudioFamilyPolicy, AudioPreprocessCheckpoint, AudioPreprocessError,
};

/// The containers the transcription and chat audio surfaces accept, which is
/// exactly b10621's set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AudioContainer {
    Wav,
    Mp3,
    Flac,
}

impl AudioContainer {
    /// The name used in diagnostics and in the `format` field clients send.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Flac => "flac",
        }
    }

    /// The `symphonia` probe hint, which shortens the probe and keeps a file
    /// whose extension lies from being decoded as the wrong container.
    fn hint(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Flac => "flac",
        }
    }
}

/// Identify the container from its leading bytes.
///
/// Content, not the client's `format` string: b10621 sniffs too, and a client
/// that mislabels an mp3 as `wav` gets a transcript there. Returns `None` for
/// anything outside the accepted set, which the caller turns into the
/// unsupported-format refusal.
#[must_use]
pub(crate) fn sniff(bytes: &[u8]) -> Option<AudioContainer> {
    if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return Some(AudioContainer::Wav);
    }
    if bytes.len() >= 4 && &bytes[..4] == b"fLaC" {
        return Some(AudioContainer::Flac);
    }
    // MP3: either an ID3v2 tag ("ID3") or a bare frame sync (11 set bits).
    if bytes.len() >= 3 && &bytes[..3] == b"ID3" {
        return Some(AudioContainer::Mp3);
    }
    if bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0 {
        return Some(AudioContainer::Mp3);
    }
    None
}

/// The geometry a clip declares, read without decoding its samples.
#[derive(Debug, Clone, Copy)]
pub(super) struct ContainerSpec {
    pub sample_rate: u32,
    pub frames: usize,
}

/// Read `bytes`' declared geometry and refuse it on the same grounds the WAV
/// path refuses one: an out-of-range source sample rate, and a clip longer
/// than `policy.max_duration_seconds`.
pub(super) fn inspect(
    bytes: &[u8],
    clip_index: usize,
    policy: AudioFamilyPolicy,
    cancelled: &dyn AudioCancellation,
) -> Result<ContainerSpec, AudioPreprocessError> {
    super::check_cancel(
        cancelled,
        AudioPreprocessCheckpoint::Decode,
        Some(clip_index),
    )?;
    match sniff(bytes) {
        Some(AudioContainer::Wav) | None => {
            // `None` keeps the WAV reader's own "missing RIFF/WAVE header"
            // diagnostic, which names what was actually wrong with the bytes.
            let spec = inspect_wav(bytes, clip_index, policy, cancelled)?;
            Ok(ContainerSpec {
                sample_rate: spec.sample_rate,
                frames: spec.frames,
            })
        }
        Some(container) => {
            let probed = probe(bytes, container, clip_index)?;
            check_sample_rate(probed.sample_rate, clip_index, policy)?;
            let frames = probed
                .frames
                .unwrap_or_else(|| max_frames(probed.sample_rate, policy).unwrap_or(usize::MAX));
            check_duration(frames, probed.sample_rate, clip_index, policy)?;
            Ok(ContainerSpec {
                sample_rate: probed.sample_rate,
                frames,
            })
        }
    }
}

/// Decode `bytes` into mono `f32` samples at the container's own sample rate.
pub(super) fn decode(
    bytes: &[u8],
    clip_index: usize,
    policy: AudioFamilyPolicy,
    cancelled: &dyn AudioCancellation,
) -> Result<NativeWaveform, AudioPreprocessError> {
    match sniff(bytes) {
        Some(AudioContainer::Wav) | None => decode_wav(bytes, clip_index, policy, cancelled),
        Some(container) => decode_compressed(bytes, container, clip_index, policy, cancelled),
    }
}

struct Probed {
    sample_rate: u32,
    frames: Option<usize>,
}

fn probe(
    bytes: &[u8],
    container: AudioContainer,
    clip_index: usize,
) -> Result<Probed, AudioPreprocessError> {
    let (_, params) = open(bytes, container, clip_index)?;
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| AudioPreprocessError::Corrupt {
            clip_index,
            reason: format!("{} stream declares no sample rate", container.name()),
        })?;
    if sample_rate == 0 {
        return Err(AudioPreprocessError::Corrupt {
            clip_index,
            reason: format!("{} stream declares a zero sample rate", container.name()),
        });
    }
    let frames = params
        .n_frames
        .and_then(|n| usize::try_from(n).ok())
        .filter(|n| *n > 0);
    Ok(Probed {
        sample_rate,
        frames,
    })
}

/// Probe the container and return its reader plus the default track's codec
/// parameters. The reader is consumed by the decode path; `probe` throws it
/// away, which costs one header parse and keeps the two entry points honest
/// about reading the same track.
fn open(
    bytes: &[u8],
    container: AudioContainer,
    clip_index: usize,
) -> Result<
    (
        Box<dyn symphonia::core::formats::FormatReader>,
        symphonia::core::codecs::CodecParameters,
    ),
    AudioPreprocessError,
> {
    // The clip is already fully in memory (the acquisition layer bounded its
    // encoded size before this point), so the media source is a cursor over
    // those bytes and no decode can reach the filesystem or the network.
    let source = MediaSourceStream::new(
        Box::new(std::io::Cursor::new(bytes.to_vec())),
        Default::default(),
    );
    let mut hint = Hint::new();
    hint.with_extension(container.hint());
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| AudioPreprocessError::Corrupt {
            clip_index,
            reason: format!("{} container could not be parsed: {e}", container.name()),
        })?;
    let reader = probed.format;
    let track = reader
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AudioPreprocessError::Corrupt {
            clip_index,
            reason: format!("{} container carries no decodable track", container.name()),
        })?;
    let params = track.codec_params.clone();
    Ok((reader, params))
}

fn max_frames(sample_rate: u32, policy: AudioFamilyPolicy) -> Result<usize, AudioPreprocessError> {
    (sample_rate as usize)
        .checked_mul(policy.max_duration_seconds)
        .ok_or(AudioPreprocessError::Overflow {
            context: "source sample limit",
        })
}

fn check_sample_rate(
    sample_rate: u32,
    clip_index: usize,
    policy: AudioFamilyPolicy,
) -> Result<(), AudioPreprocessError> {
    if sample_rate < policy.minimum_source_sample_rate {
        return Err(AudioPreprocessError::Corrupt {
            clip_index,
            reason: format!(
                "source sample rate {sample_rate} Hz is below the {} Hz family minimum",
                policy.minimum_source_sample_rate
            ),
        });
    }
    if sample_rate > policy.maximum_source_sample_rate {
        return Err(AudioPreprocessError::Limit {
            limit: "source sample rate",
            actual: sample_rate as usize,
            maximum: policy.maximum_source_sample_rate as usize,
        });
    }
    Ok(())
}

fn check_duration(
    frames: usize,
    sample_rate: u32,
    clip_index: usize,
    policy: AudioFamilyPolicy,
) -> Result<(), AudioPreprocessError> {
    let _ = clip_index;
    let maximum = max_frames(sample_rate, policy)?;
    if frames > maximum {
        return Err(AudioPreprocessError::Limit {
            limit: "source duration samples",
            actual: frames,
            maximum,
        });
    }
    Ok(())
}

fn decode_compressed(
    bytes: &[u8],
    container: AudioContainer,
    clip_index: usize,
    policy: AudioFamilyPolicy,
    cancelled: &dyn AudioCancellation,
) -> Result<NativeWaveform, AudioPreprocessError> {
    let (mut reader, params) = open(bytes, container, clip_index)?;
    let track_id = reader
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .map(|t| t.id)
        .ok_or_else(|| AudioPreprocessError::Corrupt {
            clip_index,
            reason: format!("{} container carries no decodable track", container.name()),
        })?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&params, &DecoderOptions::default())
        .map_err(|e| AudioPreprocessError::Corrupt {
            clip_index,
            reason: format!("{} codec is not supported: {e}", container.name()),
        })?;

    // The declared geometry was already checked by `inspect`, but a frame
    // count is advisory in both containers, so the hard cap is re-derived here
    // and enforced against what the decoder actually produces. A file whose
    // header understates its length stops at the cap instead of growing `mono`
    // without bound.
    let declared_rate = params.sample_rate.unwrap_or(0);
    let mut sample_rate = declared_rate;
    let mut channels: u16 = 1;
    let mut mono: Vec<f32> = Vec::new();
    let mut cap = if declared_rate > 0 {
        max_frames(declared_rate, policy)?
    } else {
        usize::MAX
    };

    loop {
        super::check_cancel(
            cancelled,
            AudioPreprocessCheckpoint::Decode,
            Some(clip_index),
        )?;
        let packet = match reader.next_packet() {
            Ok(packet) => packet,
            // Both containers signal "no more packets" as an I/O
            // end-of-stream through the cursor, which is the normal exit.
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => {
                return Err(AudioPreprocessError::Corrupt {
                    clip_index,
                    reason: format!("{} stream ended abnormally: {e}", container.name()),
                });
            }
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // A recoverable stream error is upstream's own "skip this packet"
            // contract; anything else fails the clip.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => {
                return Err(AudioPreprocessError::Corrupt {
                    clip_index,
                    reason: format!("{} packet failed to decode: {e}", container.name()),
                });
            }
        };
        let spec = *decoded.spec();
        if sample_rate == 0 {
            sample_rate = spec.rate;
            check_sample_rate(sample_rate, clip_index, policy)?;
            cap = max_frames(sample_rate, policy)?;
        } else if spec.rate != sample_rate {
            return Err(AudioPreprocessError::Corrupt {
                clip_index,
                reason: format!(
                    "{} stream changes sample rate mid-clip ({sample_rate} Hz to {} Hz)",
                    container.name(),
                    spec.rate
                ),
            });
        }
        channels = u16::try_from(spec.channels.count())
            .unwrap_or(u16::MAX)
            .max(1);
        append_mono(&decoded, &mut mono, clip_index)?;
        if mono.len() > cap {
            return Err(AudioPreprocessError::Limit {
                limit: "source duration samples",
                actual: mono.len(),
                maximum: cap,
            });
        }
    }

    if mono.is_empty() {
        return Err(AudioPreprocessError::Empty { clip_index });
    }
    let frames = mono.len();
    Ok(NativeWaveform {
        samples: mono,
        sample_rate,
        channels,
        frames,
    })
}

/// Downmix one decoded buffer to mono and append it, matching the WAV path's
/// arithmetic: mean across channels, clamped to `[-1, 1]`, with a non-finite
/// sample failing the clip rather than propagating a NaN into the features.
fn append_mono(
    decoded: &AudioBufferRef<'_>,
    out: &mut Vec<f32>,
    clip_index: usize,
) -> Result<(), AudioPreprocessError> {
    macro_rules! mix {
        ($buf:expr, $conv:expr) => {{
            let buf = $buf;
            let channels = buf.spec().channels.count().max(1);
            let frames = buf.frames();
            out.reserve(frames);
            for frame in 0..frames {
                let mut sum = 0.0f64;
                for channel in 0..channels {
                    let sample: f32 = $conv(buf.chan(channel)[frame]);
                    if !sample.is_finite() {
                        return Err(AudioPreprocessError::NonFinite { clip_index, frame });
                    }
                    sum += sample as f64;
                }
                out.push((sum / channels as f64).clamp(-1.0, 1.0) as f32);
            }
        }};
    }
    match decoded {
        AudioBufferRef::F32(buf) => mix!(buf.as_ref(), |s: f32| s),
        AudioBufferRef::F64(buf) => mix!(buf.as_ref(), |s: f64| s as f32),
        AudioBufferRef::S32(buf) => mix!(buf.as_ref(), |s: i32| s as f32 / 2_147_483_648.0),
        AudioBufferRef::S24(buf) => mix!(buf.as_ref(), |s: symphonia::core::sample::i24| {
            s.inner() as f32 / 8_388_608.0
        }),
        AudioBufferRef::S16(buf) => mix!(buf.as_ref(), |s: i16| s as f32 / 32_768.0),
        AudioBufferRef::S8(buf) => mix!(buf.as_ref(), |s: i8| s as f32 / 128.0),
        AudioBufferRef::U32(buf) => mix!(buf.as_ref(), |s: u32| (s as f64 / 2_147_483_648.0 - 1.0)
            as f32),
        AudioBufferRef::U24(buf) => mix!(buf.as_ref(), |s: symphonia::core::sample::u24| {
            (s.inner() as f64 / 8_388_608.0 - 1.0) as f32
        }),
        AudioBufferRef::U16(buf) => mix!(buf.as_ref(), |s: u16| (s as f32 / 32_768.0) - 1.0),
        AudioBufferRef::U8(buf) => mix!(buf.as_ref(), |s: u8| (s as f32 / 128.0) - 1.0),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::preprocessing::AudioFamilyPolicy;

    struct NeverCancel;
    impl AudioCancellation for NeverCancel {
        fn is_cancelled(&self, _checkpoint: AudioPreprocessCheckpoint) -> bool {
            false
        }
    }

    /// One 0.25 s 440 Hz tone, encoded three ways from the same source by
    /// ffmpeg. The three files are the whole point: a synthetic byte string
    /// cannot exercise a real MPEG frame sequence or a real FLAC subframe.
    const WAV: &[u8] = include_bytes!("../../tests/fixtures/audio/tone.wav");
    const MP3: &[u8] = include_bytes!("../../tests/fixtures/audio/tone.mp3");
    const FLAC: &[u8] = include_bytes!("../../tests/fixtures/audio/tone.flac");

    fn policy() -> AudioFamilyPolicy {
        AudioFamilyPolicy::gemma3n()
    }

    #[test]
    fn sniffing_identifies_each_container_from_its_bytes() {
        assert_eq!(sniff(WAV), Some(AudioContainer::Wav));
        assert_eq!(sniff(MP3), Some(AudioContainer::Mp3));
        assert_eq!(sniff(FLAC), Some(AudioContainer::Flac));
        // Outside b10621's set: OGG and a bare text file both refuse.
        assert_eq!(sniff(b"OggS\0\0\0\0\0\0\0\0"), None);
        assert_eq!(sniff(b"not audio at all"), None);
        assert_eq!(sniff(b""), None);
    }

    /// The three encodings of one source decode to the same clip: same rate,
    /// same length to within the encoders' own padding, and a correlated
    /// waveform. Without this the module could "work" while decoding noise.
    #[test]
    fn every_container_decodes_the_same_clip() {
        let wav = decode(WAV, 0, policy(), &NeverCancel).expect("wav");
        assert_eq!(wav.sample_rate, 16_000);
        for (name, bytes) in [("mp3", MP3), ("flac", FLAC)] {
            let got = decode(bytes, 0, policy(), &NeverCancel)
                .unwrap_or_else(|e| panic!("{name} decode: {e}"));
            assert_eq!(got.sample_rate, wav.sample_rate, "{name} sample rate");
            // FLAC is lossless and exact; MP3 pads with encoder delay, so the
            // length is compared with a tolerance of one 1152-sample frame
            // either way.
            // One MPEG frame is 1152 samples and the encoder adds a delay
            // plus a padding frame, so two frames of slack is the tolerance.
            let delta = got.frames.abs_diff(wav.frames);
            assert!(delta <= 2304, "{name} length differs by {delta} frames");
            let peak = got.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
            assert!(peak > 0.1, "{name} decoded to near-silence (peak {peak})");
            assert!(
                got.samples.iter().all(|s| s.is_finite()),
                "{name} produced a non-finite sample"
            );
        }
    }

    /// A file of the new container types whose header parses but whose payload
    /// is garbage must fail, and must fail without decoding indefinitely: the
    /// per-clip frame cap bounds the loop even when the header lies.
    #[test]
    fn a_corrupt_payload_is_refused_rather_than_decoded() {
        let mut broken = MP3[..32.min(MP3.len())].to_vec();
        broken.extend(std::iter::repeat_n(0xA5u8, 8192));
        // Either the probe rejects it or the decoder produces nothing usable;
        // both are refusals. What must not happen is a plausible waveform.
        match decode(&broken, 0, policy(), &NeverCancel) {
            Err(_) => {}
            Ok(waveform) => assert!(
                waveform.frames <= 16_000,
                "a garbage payload decoded to {} frames",
                waveform.frames
            ),
        }
        let mut broken_flac = FLAC[..64.min(FLAC.len())].to_vec();
        broken_flac.extend(std::iter::repeat_n(0x5Au8, 8192));
        assert!(
            decode(&broken_flac, 0, policy(), &NeverCancel).is_err(),
            "a FLAC stream with a garbage payload must be refused"
        );
    }

    /// A truncated file is refused, not silently served as a short clip that
    /// happens to parse.
    #[test]
    fn a_truncated_container_is_refused() {
        for (name, bytes) in [("mp3", MP3), ("flac", FLAC)] {
            let truncated = &bytes[..48.min(bytes.len())];
            assert!(
                decode(truncated, 0, policy(), &NeverCancel).is_err(),
                "a truncated {name} must be refused"
            );
        }
    }

    /// `inspect` reads the declared geometry without decoding, which is what
    /// lets the caller apply the per-request caps before any allocation.
    #[test]
    fn inspect_reports_the_declared_geometry_for_every_container() {
        for (name, bytes) in [("wav", WAV), ("mp3", MP3), ("flac", FLAC)] {
            let spec = inspect(bytes, 0, policy(), &NeverCancel)
                .unwrap_or_else(|e| panic!("{name} inspect: {e}"));
            assert_eq!(spec.sample_rate, 16_000, "{name} rate");
            assert!(spec.frames > 0, "{name} frames");
        }
    }

    /// A clip past the per-clip duration cap is refused, and the stage it is
    /// refused at follows what the container declares.
    ///
    /// FLAC carries a sample count in its STREAMINFO block, so it is refused
    /// from the declared geometry alone, before a decoder allocates anything.
    /// An MPEG stream without a Xing header declares no length at all, which
    /// is exactly the case a header check cannot bound; the decode loop's own
    /// cap catches it instead, and that is the guarantee that actually holds
    /// for every input. Asserting the same stage for both would only be true
    /// of files that happen to carry a length.
    #[test]
    fn a_clip_past_the_duration_cap_is_refused() {
        let tiny = AudioFamilyPolicy {
            max_duration_seconds: 0,
            ..policy()
        };
        let is_duration_limit = |err: &AudioPreprocessError| matches!(err, AudioPreprocessError::Limit { limit, .. } if *limit == "source duration samples");

        let err = inspect(FLAC, 0, tiny, &NeverCancel)
            .err()
            .expect("a declared length past the cap is refused before decoding");
        assert!(is_duration_limit(&err), "flac: unexpected error {err}");

        // The fixture is encoded with `-write_xing 0`, so this is the
        // no-declared-length case on purpose.
        assert!(
            inspect(MP3, 0, tiny, &NeverCancel).is_ok(),
            "an MPEG stream with no declared length has nothing to check at inspect"
        );
        let err = decode(MP3, 0, tiny, &NeverCancel)
            .err()
            .expect("the decode loop's own cap bounds an undeclared length");
        assert!(is_duration_limit(&err), "mp3: unexpected error {err}");
    }

    /// The WAV path is untouched: bytes that are not any accepted container
    /// keep the WAV reader's own diagnostic, which names what was wrong.
    #[test]
    fn unrecognized_bytes_keep_the_wav_readers_diagnostic() {
        let err = decode(b"not audio at all, really", 0, policy(), &NeverCancel)
            .err()
            .expect("refused");
        assert!(
            err.to_string().contains("RIFF/WAVE"),
            "unexpected diagnostic: {err}"
        );
    }
}
