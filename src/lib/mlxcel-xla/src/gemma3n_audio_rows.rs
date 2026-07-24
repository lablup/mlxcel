//! Fail-closed `audio.row_indices` validation at the native IREE boundary.

use std::fmt;

use crate::GEMMA3N_AUDIO_SOFT_TOKENS;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gemma3nAudioRowMapError {
    LengthMismatch {
        tokens: usize,
        rows: usize,
    },
    RowCountOverflow,
    MissingAudioRow {
        token: usize,
        expected: i32,
    },
    UnexpectedAudioRow {
        token: usize,
        row: i32,
    },
    OutOfRange {
        token: usize,
        row: i32,
        rows: usize,
    },
    NonCanonicalOrder {
        token: usize,
        row: i32,
        expected: i32,
    },
    PlaceholderCount {
        actual: usize,
        expected: usize,
    },
}

impl fmt::Display for Gemma3nAudioRowMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { tokens, rows } => write!(
                f,
                "Gemma3n audio row map has {rows} entries for {tokens} token rows"
            ),
            Self::RowCountOverflow => {
                f.write_str("Gemma3n audio row-map count exceeds the i32 graph ABI")
            }
            Self::MissingAudioRow { token, expected } => write!(
                f,
                "Gemma3n audio placeholder at token {token} is missing canonical row {expected}"
            ),
            Self::UnexpectedAudioRow { token, row } => write!(
                f,
                "Gemma3n non-audio token {token} carries unexpected audio row {row}"
            ),
            Self::OutOfRange { token, row, rows } => write!(
                f,
                "Gemma3n audio row {row} at token {token} is outside 0..{rows}"
            ),
            Self::NonCanonicalOrder {
                token,
                row,
                expected,
            } => write!(
                f,
                "Gemma3n audio row {row} at token {token} is duplicate or reordered; expected {expected}"
            ),
            Self::PlaceholderCount { actual, expected } => write!(
                f,
                "Gemma3n prompt has {actual} audio placeholders; expected {expected}"
            ),
        }
    }
}

impl std::error::Error for Gemma3nAudioRowMapError {}

/// Validate the complete static-capacity token and row-map buffers.
///
/// Every audio placeholder must map once, in prompt order, to the contiguous
/// flattened soft-token range. Every non-audio or padded token must carry `-1`.
pub fn validate_gemma3n_audio_row_indices(
    token_ids: &[i32],
    audio_token_id: i32,
    clips: usize,
    rows: &[i32],
) -> Result<(), Gemma3nAudioRowMapError> {
    if token_ids.len() != rows.len() {
        return Err(Gemma3nAudioRowMapError::LengthMismatch {
            tokens: token_ids.len(),
            rows: rows.len(),
        });
    }
    let expected_rows = clips
        .checked_mul(GEMMA3N_AUDIO_SOFT_TOKENS)
        .ok_or(Gemma3nAudioRowMapError::RowCountOverflow)?;
    let expected_rows_i32 =
        i32::try_from(expected_rows).map_err(|_| Gemma3nAudioRowMapError::RowCountOverflow)?;
    let mut next = 0_i32;
    for (token, (&token_id, &row)) in token_ids.iter().zip(rows).enumerate() {
        if token_id != audio_token_id {
            if row != -1 {
                return Err(Gemma3nAudioRowMapError::UnexpectedAudioRow { token, row });
            }
            continue;
        }
        if row < 0 {
            return Err(Gemma3nAudioRowMapError::MissingAudioRow {
                token,
                expected: next,
            });
        }
        if row >= expected_rows_i32 {
            return Err(Gemma3nAudioRowMapError::OutOfRange {
                token,
                row,
                rows: expected_rows,
            });
        }
        if row != next {
            return Err(Gemma3nAudioRowMapError::NonCanonicalOrder {
                token,
                row,
                expected: next,
            });
        }
        next += 1;
    }
    if next != expected_rows_i32 {
        return Err(Gemma3nAudioRowMapError::PlaceholderCount {
            actual: usize::try_from(next).unwrap_or_default(),
            expected: expected_rows,
        });
    }
    Ok(())
}

pub(crate) fn build_gemma3n_audio_row_indices(
    token_ids: &[i32],
    audio_token_id: i32,
    clips: usize,
) -> Result<Vec<i32>, Gemma3nAudioRowMapError> {
    let mut rows = Vec::with_capacity(token_ids.len());
    let mut next = 0_i32;
    for &token in token_ids {
        if token == audio_token_id {
            rows.push(next);
            next = next
                .checked_add(1)
                .ok_or(Gemma3nAudioRowMapError::RowCountOverflow)?;
        } else {
            rows.push(-1);
        }
    }
    validate_gemma3n_audio_row_indices(token_ids, audio_token_id, clips, &rows)?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    const AUDIO: i32 = 99;

    fn valid() -> (Vec<i32>, Vec<i32>) {
        let mut tokens = vec![1];
        tokens.extend(std::iter::repeat_n(AUDIO, GEMMA3N_AUDIO_SOFT_TOKENS));
        tokens.push(2);
        let rows = build_gemma3n_audio_row_indices(&tokens, AUDIO, 1).unwrap();
        (tokens, rows)
    }

    #[test]
    fn accepts_exact_placeholder_order_and_padding() {
        let (mut tokens, mut rows) = valid();
        tokens.resize(tokens.len() + 4, 0);
        rows.resize(rows.len() + 4, -1);
        validate_gemma3n_audio_row_indices(&tokens, AUDIO, 1, &rows).unwrap();
    }

    #[test]
    fn rejects_length_missing_and_non_audio_mappings() {
        let (tokens, rows) = valid();
        assert!(matches!(
            validate_gemma3n_audio_row_indices(&tokens, AUDIO, 1, &rows[..rows.len() - 1]),
            Err(Gemma3nAudioRowMapError::LengthMismatch { .. })
        ));
        let mut missing = rows.clone();
        missing[1] = -1;
        assert!(matches!(
            validate_gemma3n_audio_row_indices(&tokens, AUDIO, 1, &missing),
            Err(Gemma3nAudioRowMapError::MissingAudioRow { .. })
        ));
        let mut unexpected = rows;
        unexpected[0] = 0;
        assert!(matches!(
            validate_gemma3n_audio_row_indices(&tokens, AUDIO, 1, &unexpected),
            Err(Gemma3nAudioRowMapError::UnexpectedAudioRow { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_reordered_out_of_range_and_missing_count() {
        let (tokens, rows) = valid();
        let mut duplicate = rows.clone();
        duplicate[2] = 0;
        assert!(matches!(
            validate_gemma3n_audio_row_indices(&tokens, AUDIO, 1, &duplicate),
            Err(Gemma3nAudioRowMapError::NonCanonicalOrder { .. })
        ));
        let mut reordered = rows.clone();
        reordered.swap(1, 2);
        assert!(matches!(
            validate_gemma3n_audio_row_indices(&tokens, AUDIO, 1, &reordered),
            Err(Gemma3nAudioRowMapError::NonCanonicalOrder { .. })
        ));
        let mut out_of_range = rows;
        out_of_range[1] = GEMMA3N_AUDIO_SOFT_TOKENS as i32;
        assert!(matches!(
            validate_gemma3n_audio_row_indices(&tokens, AUDIO, 1, &out_of_range),
            Err(Gemma3nAudioRowMapError::OutOfRange { .. })
        ));
        let fewer_tokens = vec![AUDIO; GEMMA3N_AUDIO_SOFT_TOKENS - 1];
        let fewer_rows = (0..GEMMA3N_AUDIO_SOFT_TOKENS as i32 - 1).collect::<Vec<_>>();
        assert!(matches!(
            validate_gemma3n_audio_row_indices(&fewer_tokens, AUDIO, 1, &fewer_rows),
            Err(Gemma3nAudioRowMapError::PlaceholderCount { .. })
        ));
    }
}
