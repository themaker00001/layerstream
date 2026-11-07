//! Compare the Rust tokenizer with the reference one on the strings in tests/golden.json.
//! Needs the model files in models/Qwen2.5-0.5B-Instruct; skips (loudly) if they are absent.

use layerstream::json::Json;
use layerstream::tokenizer::Tokenizer;
use std::path::PathBuf;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/Qwen2.5-0.5B-Instruct")
}

fn golden() -> Json {
    let text = std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden.json")).unwrap();
    Json::parse(&text).unwrap()
}

#[test]
fn encodes_exactly_like_the_reference_tokenizer() {
    if !model_dir().join("vocab.json").exists() {
        eprintln!("SKIPPED: model files not downloaded (see README)");
        return;
    }
    let tok = Tokenizer::load(&model_dir()).unwrap();
    let g = golden();
    let cases = g.get("tokenizer").unwrap().as_arr().unwrap();
    assert!(cases.len() >= 20);
    let mut failures = Vec::new();
    for c in cases {
        let text = c.get("text").unwrap().as_str().unwrap();
        let want: Vec<u32> = c.get("ids").unwrap().as_arr().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let got = tok.encode(text);
        if got != want {
            failures.push(format!("{text:?}\n   want {want:?}\n   got  {got:?}"));
        }
    }
    assert!(failures.is_empty(), "{} of {} cases differ:\n{}", failures.len(), cases.len(), failures.join("\n"));
}

#[test]
fn decode_inverts_encode_for_plain_text() {
    if !model_dir().join("vocab.json").exists() {
        eprintln!("SKIPPED: model files not downloaded (see README)");
        return;
    }
    let tok = Tokenizer::load(&model_dir()).unwrap();
    for text in ["Hello, world!", "  spaces  and\ttabs\n", "Café 日本語 😀", "def f(x):\n    return x"] {
        assert_eq!(tok.decode(&tok.encode(text)), text);
    }
    // Special tokens are encoded as single ids and hidden when decoding.
    let ids = tok.encode("<|im_start|>user\nHi<|im_end|>");
    assert_eq!(ids[0], 151644);
    assert_eq!(*ids.last().unwrap(), 151645);
    assert_eq!(tok.decode(&ids), "user\nHi");
}
