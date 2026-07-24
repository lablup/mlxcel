//! Owned Gemma3n audio inputs and prepared-prefill outputs.
//!
//! Canonical mel features and valid-frame masks enter `audio.main`; post-scale
//! merged embeddings and dense PLE leave it. Owned buffers make normal
//! completion, cancellation, and admission failure use the same drop path.

use std::fmt;

use crate::Gemma3nPreparedPrefill;
use crate::gemma3n_audio_config::{
    GEMMA3N_AUDIO_FRAME_BUCKETS, GEMMA3N_AUDIO_MAX_CLIPS, GEMMA3N_AUDIO_MAX_FRAMES,
    GEMMA3N_AUDIO_MEL_BINS, GEMMA3N_AUDIO_MODALITY_FAMILY, GEMMA3N_AUDIO_SOFT_TOKENS,
    validate_gemma3n_audio_frame_bucket,
};

pub fn select_gemma3n_audio_frame_bucket(frames: usize) -> Result<usize, Gemma3nAudioInputError> {
    if frames == 0 {
        return Err(Gemma3nAudioInputError::ZeroFrames);
    }
    GEMMA3N_AUDIO_FRAME_BUCKETS
        .iter()
        .copied()
        .find(|&bucket| frames <= bucket)
        .ok_or(Gemma3nAudioInputError::FrameLimit {
            frames,
            maximum: GEMMA3N_AUDIO_MAX_FRAMES,
        })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gemma3nAudioInputError {
    ZeroFrames,
    FrameLimit {
        frames: usize,
        maximum: usize,
    },
    ClipCount {
        clips: usize,
        maximum: usize,
    },
    LengthMismatch {
        kind: &'static str,
        actual: usize,
        expected: usize,
    },
    InvalidFrameLength {
        clip: usize,
        frames: usize,
        bucket: usize,
    },
    AllPadded {
        clip: usize,
    },
    NonFinite,
    NonZeroMaskedFeature,
}

impl fmt::Display for Gemma3nAudioInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroFrames => {
                f.write_str("Gemma3n audio input must contain at least one mel frame")
            }
            Self::FrameLimit { frames, maximum } => {
                write!(
                    f,
                    "Gemma3n audio input has {frames} frames; maximum is {maximum}"
                )
            }
            Self::ClipCount { clips, maximum } => {
                write!(
                    f,
                    "Gemma3n audio input has {clips} clips; maximum is {maximum}"
                )
            }
            Self::LengthMismatch {
                kind,
                actual,
                expected,
            } => {
                write!(
                    f,
                    "Gemma3n audio {kind} has {actual} elements; expected {expected}"
                )
            }
            Self::InvalidFrameLength {
                clip,
                frames,
                bucket,
            } => {
                write!(
                    f,
                    "Gemma3n audio clip {clip} has invalid frame length {frames} for bucket {bucket}"
                )
            }
            Self::AllPadded { clip } => {
                write!(f, "Gemma3n audio clip {clip} is entirely padded")
            }
            Self::NonFinite => f.write_str("Gemma3n audio mel input contains a non-finite value"),
            Self::NonZeroMaskedFeature => f.write_str(
                "Gemma3n audio masked mel rows must be explicit zeros at the IREE boundary",
            ),
        }
    }
}

impl std::error::Error for Gemma3nAudioInputError {}

/// Canonical, zero-padded input to one static `audio.main` frame bucket.
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma3nAudioInput {
    mel: Vec<f32>,
    valid_mask: Vec<u8>,
    frame_lengths: Vec<usize>,
    frame_bucket: usize,
}

impl Gemma3nAudioInput {
    pub fn new(
        mel: Vec<f32>,
        valid_mask: Vec<u8>,
        frame_lengths: Vec<usize>,
        frame_bucket: usize,
    ) -> Result<Self, Gemma3nAudioInputError> {
        let clips = frame_lengths.len();
        if !(1..=GEMMA3N_AUDIO_MAX_CLIPS).contains(&clips) {
            return Err(Gemma3nAudioInputError::ClipCount {
                clips,
                maximum: GEMMA3N_AUDIO_MAX_CLIPS,
            });
        }
        if !GEMMA3N_AUDIO_FRAME_BUCKETS.contains(&frame_bucket) {
            return Err(Gemma3nAudioInputError::FrameLimit {
                frames: frame_bucket,
                maximum: GEMMA3N_AUDIO_MAX_FRAMES,
            });
        }
        let mask_len =
            clips
                .checked_mul(frame_bucket)
                .ok_or(Gemma3nAudioInputError::LengthMismatch {
                    kind: "mask",
                    actual: valid_mask.len(),
                    expected: usize::MAX,
                })?;
        let mel_len = mask_len.checked_mul(GEMMA3N_AUDIO_MEL_BINS).ok_or(
            Gemma3nAudioInputError::LengthMismatch {
                kind: "mel tensor",
                actual: mel.len(),
                expected: usize::MAX,
            },
        )?;
        if valid_mask.len() != mask_len {
            return Err(Gemma3nAudioInputError::LengthMismatch {
                kind: "mask",
                actual: valid_mask.len(),
                expected: mask_len,
            });
        }
        if mel.len() != mel_len {
            return Err(Gemma3nAudioInputError::LengthMismatch {
                kind: "mel tensor",
                actual: mel.len(),
                expected: mel_len,
            });
        }
        if mel.iter().any(|value| !value.is_finite()) {
            return Err(Gemma3nAudioInputError::NonFinite);
        }
        for (clip, &frames) in frame_lengths.iter().enumerate() {
            if frames == 0 {
                return Err(Gemma3nAudioInputError::AllPadded { clip });
            }
            if frames > frame_bucket {
                return Err(Gemma3nAudioInputError::InvalidFrameLength {
                    clip,
                    frames,
                    bucket: frame_bucket,
                });
            }
            let mask = &valid_mask[clip * frame_bucket..(clip + 1) * frame_bucket];
            if mask[..frames].iter().any(|&value| value != 1)
                || mask[frames..].iter().any(|&value| value != 0)
            {
                return Err(Gemma3nAudioInputError::InvalidFrameLength {
                    clip,
                    frames,
                    bucket: frame_bucket,
                });
            }
        }
        if valid_mask.iter().enumerate().any(|(index, &valid)| {
            valid == 0
                && mel[index * GEMMA3N_AUDIO_MEL_BINS..(index + 1) * GEMMA3N_AUDIO_MEL_BINS]
                    .iter()
                    .any(|&value| value != 0.0)
        }) {
            return Err(Gemma3nAudioInputError::NonZeroMaskedFeature);
        }
        Ok(Self {
            mel,
            valid_mask,
            frame_lengths,
            frame_bucket,
        })
    }

    #[must_use]
    pub fn mel(&self) -> &[f32] {
        &self.mel
    }

    #[must_use]
    pub fn valid_mask(&self) -> &[u8] {
        &self.valid_mask
    }

    #[must_use]
    pub fn frame_lengths(&self) -> &[usize] {
        &self.frame_lengths
    }

    #[must_use]
    pub fn clips(&self) -> usize {
        self.frame_lengths.len()
    }

    #[must_use]
    pub fn frame_bucket(&self) -> usize {
        self.frame_bucket
    }
}

/// `audio.main` output ready for the existing Gemma3n dense-PLE prefill entry.
#[derive(Clone, Debug)]
pub struct Gemma3nAudioPreparedPrefill {
    request: Gemma3nPreparedPrefill,
    projected_lengths: Vec<usize>,
    placeholder_starts: Vec<usize>,
    frame_bucket: usize,
}

impl Gemma3nAudioPreparedPrefill {
    pub fn new(
        request: Gemma3nPreparedPrefill,
        audio_token_id: i32,
        projected_lengths: Vec<usize>,
        frame_bucket: usize,
    ) -> Result<Self, String> {
        validate_gemma3n_audio_frame_bucket(frame_bucket)?;
        if projected_lengths.is_empty()
            || projected_lengths.len() > GEMMA3N_AUDIO_MAX_CLIPS
            || projected_lengths
                .iter()
                .any(|&length| length == 0 || length > GEMMA3N_AUDIO_SOFT_TOKENS)
        {
            return Err("Gemma3n audio projected lengths violate the fixed clip contract".into());
        }
        let clips = projected_lengths.len();
        let expected_tokens = clips
            .checked_mul(GEMMA3N_AUDIO_SOFT_TOKENS)
            .ok_or_else(|| "Gemma3n audio placeholder count overflows".to_string())?;
        let prepared = request.prepared();
        let audio_modalities: Vec<_> = prepared
            .modalities
            .iter()
            .filter(|modality| modality.family == GEMMA3N_AUDIO_MODALITY_FAMILY)
            .collect();
        if audio_modalities.len() != 1
            || audio_modalities[0].item_count != clips
            || audio_modalities[0].token_count != expected_tokens
        {
            return Err(format!(
                "Gemma3n audio modality metadata must declare {clips} clips and \
                 {expected_tokens} placeholder tokens"
            ));
        }
        let mut placeholder_starts = Vec::with_capacity(clips);
        let mut index = 0;
        while index < prepared.token_ids.len() {
            if prepared.token_ids[index] != audio_token_id {
                index += 1;
                continue;
            }
            let start = index;
            while index < prepared.token_ids.len() && prepared.token_ids[index] == audio_token_id {
                index += 1;
            }
            let length = index - start;
            if length != GEMMA3N_AUDIO_SOFT_TOKENS {
                return Err(format!(
                    "Gemma3n audio placeholder run at token {start} has {length} rows; \
                     expected {GEMMA3N_AUDIO_SOFT_TOKENS}"
                ));
            }
            placeholder_starts.push(start);
        }
        if placeholder_starts.len() != clips {
            return Err(format!(
                "Gemma3n audio placeholder run count {} does not match {clips} clips",
                placeholder_starts.len()
            ));
        }
        Ok(Self {
            request,
            projected_lengths,
            placeholder_starts,
            frame_bucket,
        })
    }

    #[must_use]
    pub fn request(&self) -> &Gemma3nPreparedPrefill {
        &self.request
    }

    #[must_use]
    pub fn projected_lengths(&self) -> &[usize] {
        &self.projected_lengths
    }

    #[must_use]
    pub fn placeholder_starts(&self) -> &[usize] {
        &self.placeholder_starts
    }

    #[must_use]
    pub fn frame_bucket(&self) -> usize {
        self.frame_bucket
    }

    pub fn into_request(self) -> Gemma3nPreparedPrefill {
        self.request
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gemma3nDensePle;
    use mlxcel_core::session::{
        OwnedTensor, PreparedAttentionBias, PreparedModality, PreparedPositions, PreparedPrefill,
        PreparedTensorDType,
    };

    fn tensor(shape: Vec<usize>, values: Vec<f32>) -> OwnedTensor {
        OwnedTensor::new(
            values.into_iter().flat_map(f32::to_le_bytes).collect(),
            PreparedTensorDType::Float32,
            shape,
        )
        .unwrap()
    }

    fn prepared_audio_request(
        token_ids: Vec<i32>,
        clips: usize,
        audio_tokens: usize,
    ) -> Gemma3nPreparedPrefill {
        let sequence_len = token_ids.len();
        let prepared = PreparedPrefill::new(
            token_ids,
            tensor(vec![1, sequence_len, 2], vec![0.0; sequence_len * 2]),
            PreparedPositions::Sequential {
                start: 0,
                length: sequence_len,
            },
            PreparedAttentionBias {
                tensor: tensor(vec![1, 1, 1, sequence_len], vec![0.0; sequence_len]),
                causal: true,
            },
            vec![PreparedModality {
                family: GEMMA3N_AUDIO_MODALITY_FAMILY.into(),
                item_count: clips,
                token_count: audio_tokens,
            }],
        )
        .unwrap();
        let ple = Gemma3nDensePle::new(vec![0.0; sequence_len * 2], sequence_len, 1, 2).unwrap();
        Gemma3nPreparedPrefill::new(prepared, ple).unwrap()
    }

    #[test]
    fn frame_buckets_are_exact_and_fail_closed() {
        assert_eq!(select_gemma3n_audio_frame_bucket(1).unwrap(), 8);
        assert_eq!(select_gemma3n_audio_frame_bucket(8).unwrap(), 8);
        assert_eq!(select_gemma3n_audio_frame_bucket(9).unwrap(), 32);
        assert_eq!(
            select_gemma3n_audio_frame_bucket(GEMMA3N_AUDIO_MAX_FRAMES).unwrap(),
            GEMMA3N_AUDIO_MAX_FRAMES
        );
        assert!(matches!(
            select_gemma3n_audio_frame_bucket(GEMMA3N_AUDIO_MAX_FRAMES + 1),
            Err(Gemma3nAudioInputError::FrameLimit { .. })
        ));
    }

    #[test]
    fn input_rejects_all_padded_nonfinite_and_dirty_padding() {
        let bucket = 8;
        let valid = vec![1, 1, 0, 0, 0, 0, 0, 0];
        let input = Gemma3nAudioInput::new(
            vec![0.0; bucket * GEMMA3N_AUDIO_MEL_BINS],
            valid,
            vec![2],
            bucket,
        )
        .unwrap();
        assert_eq!(input.clips(), 1);
        assert!(matches!(
            Gemma3nAudioInput::new(
                vec![0.0; bucket * GEMMA3N_AUDIO_MEL_BINS],
                vec![1, 0, 1, 0, 0, 0, 0, 0],
                vec![2],
                bucket,
            ),
            Err(Gemma3nAudioInputError::InvalidFrameLength { .. })
        ));
        assert!(matches!(
            Gemma3nAudioInput::new(
                vec![0.0; bucket * GEMMA3N_AUDIO_MEL_BINS],
                vec![0; bucket],
                vec![0],
                bucket,
            ),
            Err(Gemma3nAudioInputError::AllPadded { .. })
        ));
        let mut nonfinite = vec![0.0; bucket * GEMMA3N_AUDIO_MEL_BINS];
        nonfinite[0] = f32::NAN;
        assert!(matches!(
            Gemma3nAudioInput::new(nonfinite, vec![1; bucket], vec![bucket], bucket),
            Err(Gemma3nAudioInputError::NonFinite)
        ));
        let mut dirty = vec![0.0; bucket * GEMMA3N_AUDIO_MEL_BINS];
        dirty[2 * GEMMA3N_AUDIO_MEL_BINS] = 1.0;
        assert!(matches!(
            Gemma3nAudioInput::new(dirty, vec![1, 1, 0, 0, 0, 0, 0, 0], vec![2], bucket,),
            Err(Gemma3nAudioInputError::NonZeroMaskedFeature)
        ));
    }

    #[test]
    fn prepared_audio_requires_exact_clip_spans_and_modality_metadata() {
        const AUDIO: i32 = 262_273;
        let mut tokens = vec![1];
        tokens.extend(std::iter::repeat_n(AUDIO, GEMMA3N_AUDIO_SOFT_TOKENS));
        tokens.push(2);
        tokens.extend(std::iter::repeat_n(AUDIO, GEMMA3N_AUDIO_SOFT_TOKENS));
        tokens.push(3);
        let request = prepared_audio_request(tokens.clone(), 2, 2 * GEMMA3N_AUDIO_SOFT_TOKENS);
        let prepared =
            Gemma3nAudioPreparedPrefill::new(request, AUDIO, vec![188, 17], 2_997).unwrap();
        assert_eq!(
            prepared.placeholder_starts(),
            &[1, GEMMA3N_AUDIO_SOFT_TOKENS + 2]
        );

        let mut malformed = tokens;
        malformed.remove(1);
        let request = prepared_audio_request(malformed, 2, 2 * GEMMA3N_AUDIO_SOFT_TOKENS);
        assert!(
            Gemma3nAudioPreparedPrefill::new(request, AUDIO, vec![188, 17], 2_997)
                .unwrap_err()
                .contains("placeholder run")
        );
    }
}
