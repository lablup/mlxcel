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

//! Content parts accepted by the OpenAI Responses API request surface.

use serde::de::{DeserializeOwned, Error as _};
use serde::{Deserialize, Deserializer};

use super::request::{ContentPart, ImageUrl, InputAudio, VideoUrl};

pub const INPUT_IMAGE_FILE_ID_UNSUPPORTED: &str =
    "input_image.file_id is not supported by this server. Provide image_url instead.";
pub const INPUT_FILE_UNSUPPORTED: &str = "input_file is not supported by this server.";

/// Responses-native parts plus the chat-completions spellings accepted by
/// earlier mlxcel releases.
///
/// Unknown objects are retained so a function output can preserve them as
/// JSON text instead of silently discarding client data.
#[derive(Debug, Clone)]
pub enum ResponseInputPart {
    InputText {
        text: String,
    },
    InputImage {
        image_url: Option<String>,
        detail: Option<String>,
        file_id: Option<String>,
    },
    InputFile {
        raw: serde_json::Map<String, serde_json::Value>,
    },
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ImageUrl,
    },
    VideoUrl {
        raw: serde_json::Map<String, serde_json::Value>,
    },
    InputAudio {
        raw: serde_json::Map<String, serde_json::Value>,
    },
    Unknown {
        part_type: String,
        raw: serde_json::Map<String, serde_json::Value>,
    },
}

impl<'de> Deserialize<'de> for ResponseInputPart {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let serde_json::Value::Object(mut raw) = value else {
            return Err(D::Error::custom("response input part must be an object"));
        };
        let part_type = raw
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| D::Error::custom("response input part must include a string 'type'"))?
            .to_string();

        match part_type.as_str() {
            "input_text" => Ok(Self::InputText {
                text: required_field(&raw, &part_type, "text")?,
            }),
            "input_image" => Ok(Self::InputImage {
                image_url: optional_field(&raw, "image_url")?,
                detail: optional_field(&raw, "detail")?,
                file_id: optional_field(&raw, "file_id")?,
            }),
            "input_file" => {
                raw.remove("type");
                Ok(Self::InputFile { raw })
            }
            "text" => Ok(Self::Text {
                text: required_field(&raw, &part_type, "text")?,
            }),
            "image_url" => Ok(Self::ImageUrl {
                image_url: required_field(&raw, &part_type, "image_url")?,
            }),
            "video_url" => {
                let _: VideoUrl = required_field(&raw, &part_type, "video_url")?;
                raw.remove("type");
                Ok(Self::VideoUrl { raw })
            }
            "input_audio" => {
                let _: InputAudio = required_field(&raw, &part_type, "input_audio")?;
                raw.remove("type");
                Ok(Self::InputAudio { raw })
            }
            _ => {
                raw.remove("type");
                Ok(Self::Unknown { part_type, raw })
            }
        }
    }
}

fn required_field<T, E>(
    raw: &serde_json::Map<String, serde_json::Value>,
    part_type: &str,
    field: &str,
) -> Result<T, E>
where
    T: DeserializeOwned,
    E: serde::de::Error,
{
    raw.get(field)
        .cloned()
        .ok_or_else(|| E::custom(format!("{part_type}.{field} is required")))
        .and_then(|value| serde_json::from_value(value).map_err(E::custom))
}

fn optional_field<T, E>(
    raw: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<Option<T>, E>
where
    T: DeserializeOwned,
    E: serde::de::Error,
{
    serde_json::from_value(raw.get(field).cloned().unwrap_or(serde_json::Value::Null))
        .map_err(E::custom)
}

fn typed_field<T>(
    raw: &serde_json::Map<String, serde_json::Value>,
    part_type: &str,
    field: &str,
) -> Result<T, String>
where
    T: DeserializeOwned,
{
    let value = raw
        .get(field)
        .cloned()
        .ok_or_else(|| format!("{part_type}.{field} is required"))?;
    serde_json::from_value(value).map_err(|err| format!("{part_type}.{field} is invalid: {err}"))
}

impl ResponseInputPart {
    pub(crate) fn to_json_value(&self) -> serde_json::Value {
        match self {
            Self::InputText { text } => serde_json::json!({"type": "input_text", "text": text}),
            Self::InputImage {
                image_url,
                detail,
                file_id,
            } => {
                let mut raw = serde_json::Map::new();
                raw.insert("type".to_string(), serde_json::json!("input_image"));
                if let Some(image_url) = image_url {
                    raw.insert("image_url".to_string(), serde_json::json!(image_url));
                }
                if let Some(detail) = detail {
                    raw.insert("detail".to_string(), serde_json::json!(detail));
                }
                if let Some(file_id) = file_id {
                    raw.insert("file_id".to_string(), serde_json::json!(file_id));
                }
                serde_json::Value::Object(raw)
            }
            Self::InputFile { raw } => object_with_type("input_file", raw.clone()),
            Self::Text { text } => serde_json::json!({"type": "text", "text": text}),
            Self::ImageUrl { image_url } => {
                serde_json::json!({"type": "image_url", "image_url": image_url})
            }
            Self::VideoUrl { raw, .. } => object_with_type("video_url", raw.clone()),
            Self::InputAudio { raw, .. } => object_with_type("input_audio", raw.clone()),
            Self::Unknown { part_type, raw } => object_with_type(part_type, raw.clone()),
        }
    }
}

fn object_with_type(
    part_type: &str,
    mut raw: serde_json::Map<String, serde_json::Value>,
) -> serde_json::Value {
    raw.insert("type".to_string(), serde_json::json!(part_type));
    serde_json::Value::Object(raw)
}

impl TryFrom<&ResponseInputPart> for ContentPart {
    type Error = String;

    fn try_from(part: &ResponseInputPart) -> Result<Self, Self::Error> {
        match part {
            ResponseInputPart::InputText { text } | ResponseInputPart::Text { text } => {
                Ok(ContentPart::Text { text: text.clone() })
            }
            ResponseInputPart::InputImage {
                image_url,
                detail,
                file_id,
            } => match image_url {
                Some(url) => Ok(ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: url.clone(),
                        detail: detail.clone(),
                        max_soft_tokens: None,
                    },
                }),
                None if file_id.is_some() => Err(INPUT_IMAGE_FILE_ID_UNSUPPORTED.to_string()),
                None => Err("input_image.image_url is required by this server".to_string()),
            },
            ResponseInputPart::InputFile { .. } => Err(INPUT_FILE_UNSUPPORTED.to_string()),
            ResponseInputPart::ImageUrl { image_url } => Ok(ContentPart::ImageUrl {
                image_url: image_url.clone(),
            }),
            ResponseInputPart::VideoUrl { raw } => Ok(ContentPart::VideoUrl {
                video_url: typed_field(raw, "video_url", "video_url")?,
            }),
            ResponseInputPart::InputAudio { raw } => Ok(ContentPart::InputAudio {
                input_audio: typed_field(raw, "input_audio", "input_audio")?,
            }),
            ResponseInputPart::Unknown { part_type, .. } => Err(format!(
                "input part type '{part_type}' is not supported by this server"
            )),
        }
    }
}

impl TryFrom<ResponseInputPart> for ContentPart {
    type Error = String;

    fn try_from(part: ResponseInputPart) -> Result<Self, Self::Error> {
        ContentPart::try_from(&part)
    }
}

/// String or content-part-array form of a function call result.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ResponseToolOutput {
    Text(String),
    Parts(Vec<ResponseInputPart>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_image_maps_detail_to_chat_image() {
        let part: ResponseInputPart = serde_json::from_str(
            r#"{"type":"input_image","image_url":"data:image/png;base64,aA==","detail":"high"}"#,
        )
        .unwrap();
        let ContentPart::ImageUrl { image_url } = ContentPart::try_from(part).unwrap() else {
            panic!("expected image part");
        };
        assert_eq!(image_url.url, "data:image/png;base64,aA==");
        assert_eq!(image_url.detail.as_deref(), Some("high"));
        assert_eq!(image_url.max_soft_tokens, None);
    }

    #[test]
    fn file_id_only_image_has_named_error() {
        let part: ResponseInputPart =
            serde_json::from_str(r#"{"type":"input_image","file_id":"file_1"}"#).unwrap();
        assert_eq!(
            ContentPart::try_from(part).unwrap_err(),
            INPUT_IMAGE_FILE_ID_UNSUPPORTED
        );
    }

    #[test]
    fn unknown_part_retains_its_json_object() {
        let part: ResponseInputPart =
            serde_json::from_str(r#"{"type":"foo","answer":42}"#).unwrap();
        assert_eq!(
            part.to_json_value(),
            serde_json::json!({"type":"foo","answer":42})
        );
    }
}
