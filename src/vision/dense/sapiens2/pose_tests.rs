use super::*;

#[test]
fn sapiens2_bbox_padding_aspect_and_clip() {
    let (center, scale) = bbox_xyxy2cs([10., 20., 50., 100.], 1.25).unwrap();
    assert_eq!(center, [30., 60.]);
    assert_eq!(scale, [50., 100.]);
    assert_eq!(fix_aspect_ratio(scale, 0.75).unwrap(), [75., 100.]);
    assert_eq!(fix_aspect_ratio([100., 50.], 0.5).unwrap(), [100., 200.]);
    assert_eq!(
        clip_boxes(&[[-5., -8., 300., 200.]], 100, 80).unwrap(),
        vec![[0., 0., 100., 80.]]
    );
    assert!(clip_boxes(&[], 100, 80).unwrap().is_empty());
    assert!(clip_boxes(&[[200., 0., 300., 10.]], 100, 80).is_err());
    assert!(bbox_xyxy2cs([0., 0., f32::NAN, 5.], 1.25).is_err());
}

#[test]
fn sapiens2_dark_udp_recovers_fractional_gaussian_peak() {
    let mut heat = Vec::with_capacity(64 * 64);
    for y in 0..64 {
        for x in 0..64 {
            heat.push((-((x as f32 - 27.3).powi(2) + (y as f32 - 31.7).powi(2)) / 8.0).exp());
        }
    }
    let heat = mlxcel_core::from_slice_f32(&heat, &[1, 64, 64, 1]);
    let (keys, scores) = udp_decode_batch(&heat, (63, 63), 11).unwrap();
    let keys = mlxcel_core::utils::array_to_vec_f32(&keys);
    assert!((keys[0] - 27.3).abs() < 0.05, "{keys:?}");
    assert!((keys[1] - 31.7).abs() < 0.05, "{keys:?}");
    let scores = mlxcel_core::utils::array_to_vec_f32(&scores);
    assert!((scores[0] - 0.9777514).abs() < 1e-6);
}

#[test]
fn sapiens2_udp_empty_response_keeps_sentinel() {
    let heat = mlxcel_core::from_slice_f32(&[0.; 25], &[1, 5, 5, 1]);
    let (keys, scores) = udp_decode_batch(&heat, (4, 4), 3).unwrap();
    assert_eq!(mlxcel_core::utils::array_to_vec_f32(&keys), vec![-1., -1.]);
    assert_eq!(mlxcel_core::utils::array_to_vec_f32(&scores), vec![0.]);
}

#[test]
fn sapiens2_flip_permutation_is_involution() {
    let indices = flip_indices_from_pairs(7, &[[1, 2], [3, 5]]).unwrap();
    assert_eq!(indices, vec![0, 2, 1, 5, 4, 3, 6]);
    for i in 0..7 {
        assert_eq!(indices[indices[i] as usize], i as i32);
    }
    assert!(flip_indices_from_pairs(2, &[[0, 2]]).is_err());
    assert!(flip_indices_from_pairs(3, &[[0, 1], [1, 2]]).is_err());
}

#[test]
fn sapiens2_keypoints_map_to_source_coordinates() {
    let points = mlxcel_core::from_slice_f32(&[0., 0., 50., 100., 100., 200.], &[1, 3, 2]);
    let mapped = keypoints_to_image(&points, (100, 200), &[[60., 120.]], &[[80., 160.]]).unwrap();
    assert_eq!(
        mlxcel_core::utils::array_to_vec_f32(&mapped),
        vec![20., 40., 60., 120., 100., 200.]
    );
}
