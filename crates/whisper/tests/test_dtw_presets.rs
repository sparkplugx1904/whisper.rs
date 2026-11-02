use whisper_rs::{get_alignment_heads, AlignmentHeadsPreset};

#[test]
fn test_all_dtw_alignment_presets() {
    let presets = [
        AlignmentHeadsPreset::TinyEn,
        AlignmentHeadsPreset::Tiny,
        AlignmentHeadsPreset::BaseEn,
        AlignmentHeadsPreset::Base,
        AlignmentHeadsPreset::SmallEn,
        AlignmentHeadsPreset::Small,
        AlignmentHeadsPreset::MediumEn,
        AlignmentHeadsPreset::Medium,
        AlignmentHeadsPreset::LargeV1,
        AlignmentHeadsPreset::LargeV2,
        AlignmentHeadsPreset::LargeV3,
        AlignmentHeadsPreset::LargeV3Turbo,
    ];

    for preset in presets {
        let heads = get_alignment_heads(preset);
        assert!(
            !heads.is_empty(),
            "Preset {:?} must have pre-calibrated alignment heads",
            preset
        );
        for &(layer, head) in heads {
            // Verify reasonable layer and head numbers
            assert!(layer <= 32, "Layer index {} out of range", layer);
            assert!(head <= 32, "Head index {} out of range", head);
        }
    }

    assert_eq!(get_alignment_heads(AlignmentHeadsPreset::None), &[]);
    assert_eq!(get_alignment_heads(AlignmentHeadsPreset::Custom), &[]);
}
