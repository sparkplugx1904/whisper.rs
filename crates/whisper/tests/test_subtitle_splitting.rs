use whisper_rs::dtw::WordTiming;
use whisper_rs::Segment;

#[test]
fn test_to_karaoke_vtt_format() {
    let words = vec![
        WordTiming::new("Hello", 0.0, 0.5, 0.95),
        WordTiming::new("world!", 0.5, 1.2, 0.98),
    ];
    let seg = Segment::with_words(0.0, 1.2, "Hello world!".to_string(), vec![1, 2], words);
    let vtt = seg.to_karaoke_vtt();
    assert!(vtt.contains("00:00:00.000 --> 00:00:01.200"));
    assert!(vtt.contains("<00:00:00.000>Hello <00:00:00.500>world!"));
}

#[test]
fn test_to_karaoke_srt_format() {
    let words = vec![
        WordTiming::new("Hello", 0.0, 0.5, 0.95),
        WordTiming::new("world!", 0.5, 1.2, 0.98),
    ];
    let seg = Segment::with_words(0.0, 1.2, "Hello world!".to_string(), vec![1, 2], words);
    let srt = seg.to_karaoke_srt(1);
    assert!(srt.starts_with("1\n00:00:00,000 --> 00:00:01,200"));
    assert!(srt.contains("<00:00:00,000>Hello <00:00:00,500>world!"));
}
