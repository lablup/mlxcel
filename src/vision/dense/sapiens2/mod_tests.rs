use super::*;

#[test]
fn pose_boxes_default_empty_clip_and_validate() {
    let opts = Map::new();
    assert_eq!(
        parse_pose_boxes(&opts, 1280, 720).unwrap(),
        [[0.0, 0.0, 1279.0, 719.0]]
    );

    let mut opts = Map::new();
    opts.insert("boxes".into(), serde_json::json!([]));
    assert!(parse_pose_boxes(&opts, 1280, 720).unwrap().is_empty());

    opts.insert("boxes".into(), serde_json::json!([[-4, -3, 1300, 800]]));
    assert_eq!(
        parse_pose_boxes(&opts, 1280, 720).unwrap(),
        [[0.0, 0.0, 1280.0, 720.0]]
    );

    opts.insert("boxes".into(), serde_json::json!([[0, 0, 3]]));
    assert!(parse_pose_boxes(&opts, 1280, 720).is_err());
    opts.insert("boxes".into(), serde_json::json!([[10, 10, 5, 20]]));
    assert!(parse_pose_boxes(&opts, 1280, 720).is_err());
}
