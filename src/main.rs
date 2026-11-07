use layerstream::model::{Model, Sampler};
use layerstream::provider::{Provider, ResidentProvider, StreamProvider};
use layerstream::seal::{self, OpenOptions, SealParams, SealedSource, SecretKey};
use layerstream::source::{FileSource, Source, Throttle};
use layerstream::sys;
use layerstream::tokenizer::{StreamDecoder, Tokenizer};
use layerstream::quant;
use layerstream::weights::{ModelFiles, QuantHeader, WeightFormat};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_SYSTEM: &str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
const DEFAULT_PROMPT: &str = "Explain in two sentences why the sky is blue.";
const EOS: [u32; 2] = [151645, 151643];
const BOOL_FLAGS: &[&str] = &["raw", "report", "quiet", "wipe", "help", "no-verify", "quick", "rtn"];

fn usage() -> ! {
    eprintln!(
        "layerstream: run a language model larger than your memory budget

USAGE
  layerstream generate --model DIR [options]       answer a prompt
  layerstream bench    --model DIR [options]       compare memory and speed of every mode
  layerstream gen-key  PATH                        write a new random key (mode 0600)
  layerstream seal     --model DIR --out FILE --key-file KEY     encrypt the model
                       (or --quantized FILE instead of --model, to encrypt a quantized model)
  layerstream quantize --model DIR --out FILE [--bits 8|4] [--group N] [--rtn]
                                                   store the weights as 8- or 4-bit integers
  layerstream eval     --model DIR --text FILE [options]   perplexity of the model on a text file
  layerstream tokenize --model DIR                 print token ids for each line on stdin

COMMON OPTIONS
  --model DIR          directory with the model files (tokenizer files are always read from here)
  --quantized FILE     read the weights from a file made by `quantize` instead of the bf16 ones
  --sealed FILE        read the model from an encrypted file made by `seal` (needs --key-file
                       or LAYERSTREAM_KEY); the plaintext weights are not touched
  --mode MODE          resident (load everything into memory) or stream (default)
  --slots N            stream: layer-sized buffers; 2 reads ahead while computing (default 2)
  --throttle-mbps N    cap storage reads at N MB/s, to emulate a device with slow flash
  --wipe               stream: overwrite each buffer with zeros as soon as it is used
  --threads N          compute threads (default: all cores)
  --max-ctx N          context limit in tokens (default 1024)

GENERATE OPTIONS
  --prompt TEXT        the user message           --system TEXT   system message
  --raw                send --prompt as plain text instead of a chat message
  --max-new N          tokens to generate (default 64)
  --temperature F      0 = greedy (default); also --top-k N, --top-p F, --seed N
  --report             print one machine-readable RESULT line instead of the text

QUANTIZE OPTIONS
  --bits N             8 (default) or 4          --group N   weights per shared scale (default 64
                       for 8 bits, 32 for 4 bits)   --rtn   plain rounding instead of clip search
  --embed-bits N --embed-group N   a different format for the embedding table / output layer
                       (default: same as the layers)

EVAL OPTIONS
  --text FILE          text to score, token by token   --max-tokens N   (default 512)
  --save-top1 FILE     write the model's top choice at every position
  --agree-with FILE    report how often this model's top choice matches a saved one

BENCH OPTIONS
  --sealed FILE --key-file KEY    also benchmark the encrypted file
  --quantized FILE                also benchmark a quantized file
  --quantized-sealed FILE         also benchmark an encrypted quantized file (needs --key-file)
  --max-new N                     tokens per run (default 32)   --quick   fewer, shorter runs"
    );
    std::process::exit(2);
}

struct Args {
    cmd: String,
    positional: Vec<String>,
    flags: HashMap<String, String>,
}

impl Args {
    fn parse() -> Args {
        let raw: Vec<String> = std::env::args().skip(1).collect();
        let Some(cmd) = raw.first().cloned() else { usage() };
        let mut flags = HashMap::new();
        let mut positional = Vec::new();
        let mut i = 1;
        while i < raw.len() {
            let Some(name) = raw[i].strip_prefix("--") else {
                positional.push(raw[i].clone());
                i += 1;
                continue;
            };
            if BOOL_FLAGS.contains(&name) {
                flags.insert(name.to_string(), "1".to_string());
                i += 1;
            } else {
                let Some(v) = raw.get(i + 1) else {
                    eprintln!("--{name} needs a value");
                    usage();
                };
                flags.insert(name.to_string(), v.clone());
                i += 2;
            }
        }
        Args { cmd, positional, flags }
    }

    fn has(&self, k: &str) -> bool {
        self.flags.contains_key(k)
    }

    fn str(&self, k: &str) -> Option<&str> {
        self.flags.get(k).map(String::as_str)
    }

    fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> T {
        match self.flags.get(k) {
            None => default,
            Some(v) => v.parse().unwrap_or_else(|_| {
                eprintln!("--{k}: cannot parse {v:?}");
                std::process::exit(2)
            }),
        }
    }

    fn model_dir(&self) -> PathBuf {
        PathBuf::from(self.str("model").unwrap_or_else(|| {
            eprintln!("--model DIR is required");
            std::process::exit(2)
        }))
    }
}

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1);
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

struct Built {
    model: Model,
    label: String,
    load_secs: f64,
}

fn build(args: &Args, default_mode: &str) -> Built {
    let dir = args.model_dir();
    let mode = args.str("mode").unwrap_or(default_mode).to_string();
    let slots: usize = args.num("slots", 2);
    let threads: usize = args.num("threads", default_threads());
    let max_ctx: usize = args.num("max-ctx", 1024);
    let wipe = args.has("wipe");
    let throttle = args.str("throttle-mbps").map(|v| {
        let mbps: f64 = v.parse().unwrap_or_else(|_| die("--throttle-mbps needs a number"));
        if mbps <= 0.0 {
            die("--throttle-mbps must be positive");
        }
        Throttle::new(mbps)
    });

    let t0 = Instant::now();
    let source: Arc<dyn Source> = match (args.str("sealed"), args.str("quantized")) {
        (Some(_), Some(_)) => die("use --sealed or --quantized, not both (encrypt a quantized file with `seal --quantized`)"),
        (Some(path), None) => {
            let key = SecretKey::from_cli(args.str("key-file")).unwrap_or_else(|e| die(e));
            let opts = OpenOptions { throttle, threads: None };
            Arc::new(SealedSource::open(Path::new(path), key, opts).unwrap_or_else(|e| die(e)))
        }
        (None, Some(path)) => {
            let files = ModelFiles::open_quantized(Path::new(path)).unwrap_or_else(|e| die(e));
            Arc::new(FileSource::open(files).unwrap_or_else(|e| die(e)).with_throttle(throttle))
        }
        (None, None) => {
            let files = ModelFiles::open(&dir).unwrap_or_else(|e| die(e));
            Arc::new(FileSource::open(files).unwrap_or_else(|e| die(e)).with_throttle(throttle))
        }
    };
    let provider: Box<dyn Provider> = match mode.as_str() {
        "resident" => Box::new(ResidentProvider::load(source).unwrap_or_else(|e| die(e))),
        "stream" => Box::new(StreamProvider::new(source, slots, wipe)),
        other => die(format!("unknown --mode {other:?} (use resident or stream)")),
    };
    let label = provider.report().name;
    let model = Model::new(provider, threads, max_ctx);
    Built { model, label, load_secs: t0.elapsed().as_secs_f64() }
}

fn chat_prompt(system: &str, user: &str) -> String {
    format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

fn cmd_generate(args: &Args) {
    let tok = Tokenizer::load(&args.model_dir()).unwrap_or_else(|e| die(e));
    let rss_tokenizer = sys::peak_rss_bytes();
    let user = args.str("prompt").unwrap_or(DEFAULT_PROMPT);
    let text = if args.has("raw") {
        user.to_string()
    } else {
        chat_prompt(args.str("system").unwrap_or(DEFAULT_SYSTEM), user)
    };
    let prompt_ids = tok.encode(&text);
    let max_new: usize = args.num("max-new", 64);
    let report = args.has("report");
    let quiet = args.has("quiet") || report;
    let mut sampler = Sampler::new(args.num("temperature", 0.0), args.num("top-k", 40), args.num("top-p", 0.95), args.num("seed", 1));

    let Built { mut model, label, load_secs } = build(args, "stream");
    let rss_after_load = sys::peak_rss_bytes();
    let bytes_at_load = model.report().io.bytes_read;
    if prompt_ids.len() + max_new > model.max_ctx() {
        die(format!("prompt ({}) + --max-new ({max_new}) exceeds --max-ctx ({})", prompt_ids.len(), model.max_ctx()));
    }

    let mut out_ids: Vec<u32> = Vec::new();
    let mut dec = StreamDecoder::default();
    let t_first = Instant::now();
    let mut logits = model.forward(&prompt_ids).unwrap_or_else(|e| die(e));
    let first_secs = t_first.elapsed().as_secs_f64();
    let mut decode_secs = 0.0;
    let mut decode_forwards = 0usize;
    for step in 0..max_new {
        let next = sampler.sample(&logits);
        out_ids.push(next);
        if !quiet {
            let s = dec.push(&tok.token_bytes(next, false));
            if !s.is_empty() {
                print!("{s}");
                let _ = std::io::stdout().flush();
            }
        }
        if EOS.contains(&next) || step + 1 == max_new {
            break;
        }
        let t = Instant::now();
        logits = model.forward(&[next]).unwrap_or_else(|e| die(e));
        decode_secs += t.elapsed().as_secs_f64();
        decode_forwards += 1;
    }
    if !quiet {
        println!("{}", dec.finish());
    }

    let rep = model.report();
    let peak = sys::peak_rss_bytes();
    let tok_s = if decode_secs > 0.0 { decode_forwards as f64 / decode_secs } else { 0.0 };

    if report {
        let ids: Vec<String> = out_ids.iter().map(u32::to_string).collect();
        println!(
            "RESULT peak_rss={} rss_tokenizer={} rss_after_load={} load_s={:.4} first_s={:.4} decode_s={:.4} prompt_tokens={} new_tokens={} decode_forwards={} bytes_at_load={} bytes_read={} read_s={:.4} decrypt_s={:.4} wait_s={:.4} resident_weights={} kv={} ids={}",
            peak,
            rss_tokenizer,
            rss_after_load,
            load_secs,
            first_secs,
            decode_secs,
            prompt_ids.len(),
            out_ids.len(),
            decode_forwards,
            bytes_at_load,
            rep.io.bytes_read,
            rep.io.read_ns as f64 / 1e9,
            rep.io.decrypt_ns as f64 / 1e9,
            rep.wait_ns as f64 / 1e9,
            rep.resident_weight_bytes,
            model.kv_cache_bytes(),
            ids.join(",")
        );
    } else {
        eprintln!("\n[layerstream] {label} | {} threads", model.threads());
        eprintln!("[layerstream] {} prompt tokens, {} generated", prompt_ids.len(), out_ids.len());
        eprintln!("[layerstream] load {load_secs:.2} s | first token {first_secs:.2} s | then {tok_s:.1} tokens/s");
        eprintln!(
            "[layerstream] peak memory {:.0} MB (weights held: {:.0} MB; tokenizer alone: {:.0} MB)",
            mb(peak),
            mb(rep.resident_weight_bytes),
            mb(rss_tokenizer)
        );
        if rep.io.bytes_read > 0 {
            eprintln!(
                "[layerstream] read {:.0} MB from storage in {:.2} s; decrypted in {:.2} s; engine waited {:.2} s for data",
                mb(rep.io.bytes_read),
                rep.io.read_ns as f64 / 1e9,
                rep.io.decrypt_ns as f64 / 1e9,
                rep.wait_ns as f64 / 1e9
            );
        }
    }
}

fn cmd_tokenize(args: &Args) {
    let tok = Tokenizer::load(&args.model_dir()).unwrap_or_else(|e| die(e));
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap_or_else(|e| die(e));
        let ids: Vec<String> = tok.encode(&unescape_line(&line)).iter().map(u32::to_string).collect();
        println!("{}", ids.join(" "));
    }
}

/// Input lines carry newlines, tabs, carriage returns and backslashes as the two-character
/// escapes `\n`, `\t`, `\r` and `\\`, so that one line can hold any text.
fn unescape_line(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn cmd_gen_key(args: &Args) {
    let Some(path) = args.positional.first() else { usage() };
    SecretKey::random().and_then(|k| k.write_to_file(path)).unwrap_or_else(|e| die(e));
    println!("[ok] wrote a new 256-bit key to {path} (64 hex characters, mode 0600).");
    println!("     Keep it out of version control and away from the encrypted file.");
}

fn cmd_seal(args: &Args) {
    let out = PathBuf::from(args.str("out").unwrap_or_else(|| die("--out FILE is required")));
    let key = SecretKey::from_cli(args.str("key-file")).unwrap_or_else(|e| die(e));
    let params = SealParams {
        layer_chunk: args.num("layer-chunk", SealParams::default().layer_chunk),
        embed_chunk_rows: args.num("embed-chunk-rows", SealParams::default().embed_chunk_rows),
    };
    let (config_text, files) = match args.str("quantized") {
        Some(q) => (
            QuantHeader::read(Path::new(q)).unwrap_or_else(|e| die(e)).config_text,
            ModelFiles::open_quantized(Path::new(q)).unwrap_or_else(|e| die(e)),
        ),
        None => {
            let dir = args.model_dir();
            (
                std::fs::read_to_string(dir.join("config.json")).unwrap_or_else(|e| die(format!("config.json: {e}"))),
                ModelFiles::open(&dir).unwrap_or_else(|e| die(e)),
            )
        }
    };
    let src = FileSource::open(files).unwrap_or_else(|e| die(e));
    let prefix = seal::random_nonce_prefix().unwrap_or_else(|e| die(e));
    let t = Instant::now();
    let s = seal::seal_model(&src, &config_text, &key, prefix, params, &out).unwrap_or_else(|e| die(e));
    println!(
        "[ok] sealed {:.0} MB into {:.0} MB ({} chunks) at {} in {:.1} s",
        mb(s.plain_bytes),
        mb(s.sealed_bytes),
        s.chunks,
        out.display(),
        t.elapsed().as_secs_f64()
    );
    if !args.has("no-verify") {
        let key2 = SecretKey::from_cli(args.str("key-file")).unwrap_or_else(|e| die(e));
        let sealed = SealedSource::open(&out, key2, OpenOptions::default()).unwrap_or_else(|e| die(e));
        seal::verify_same(&src, &sealed).unwrap_or_else(|e| die(format!("self-check failed: {e}")));
        println!("[ok] self-check: decrypting the new file gives exactly the original weights.");
    }
}

fn cmd_quantize(args: &Args) {
    let dir = args.model_dir();
    let out = PathBuf::from(args.str("out").unwrap_or_else(|| die("--out FILE is required")));
    let bits: u8 = args.num("bits", 8);
    let group: usize = args.num("group", if bits == 4 { 32 } else { 64 });
    let fmt = WeightFormat::Quant { bits, group };
    let embed_fmt = WeightFormat::Quant { bits: args.num("embed-bits", bits), group: args.num("embed-group", group) };
    let clip = !args.has("rtn");
    let config_text = std::fs::read_to_string(dir.join("config.json")).unwrap_or_else(|e| die(format!("config.json: {e}")));
    let files = ModelFiles::open(&dir).unwrap_or_else(|e| die(e));
    fmt.validate(&[files.cfg.hidden, files.cfg.inter]).unwrap_or_else(|e| die(e));
    embed_fmt.validate(&[files.cfg.hidden]).unwrap_or_else(|e| die(e));
    let src = FileSource::open(files).unwrap_or_else(|e| die(e));
    let t = Instant::now();
    let s = quant::quantize_model(&src, &config_text, fmt, embed_fmt, clip, args.num("threads", default_threads()), &out)
        .unwrap_or_else(|e| die(e));
    println!(
        "[ok] {}{} ({}): {:.0} MB -> {:.0} MB ({:.2}x smaller); weight error {:.2}% of the weights' RMS; {:.1} s",
        fmt.name(),
        if embed_fmt == fmt { String::new() } else { format!(", embedding {}", embed_fmt.name()) },
        if clip { "clip search" } else { "round to nearest" },
        mb(s.in_bytes),
        mb(s.out_bytes),
        s.in_bytes as f64 / s.out_bytes as f64,
        s.rel_rmse * 100.0,
        t.elapsed().as_secs_f64()
    );
    println!("     wrote {}", out.display());
}

/// Teacher-forced scoring: feed the text one token at a time and measure how much probability
/// the model gave to each next token. Lower perplexity = the model predicts the text better.
fn cmd_eval(args: &Args) {
    let tok = Tokenizer::load(&args.model_dir()).unwrap_or_else(|e| die(e));
    let path = args.str("text").unwrap_or_else(|| die("--text FILE is required"));
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| die(format!("{path}: {e}")));
    let max_tokens: usize = args.num("max-tokens", 512);
    let mut ids = tok.encode(&text);
    ids.truncate(max_tokens + 1);
    if ids.len() < 2 {
        die("the text is too short to score");
    }
    let Built { mut model, label, .. } = build(args, "resident");
    let n = ids.len() - 1;
    if n > model.max_ctx() {
        die(format!("{n} tokens exceed --max-ctx ({})", model.max_ctx()));
    }
    let (mut nll, mut hits) = (0.0f64, 0usize);
    let mut top1: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        let logits = model.forward(&ids[i..i + 1]).unwrap_or_else(|e| die(e));
        let (mut best, mut max) = (0usize, f32::NEG_INFINITY);
        for (j, &v) in logits.iter().enumerate() {
            if v > max {
                max = v;
                best = j;
            }
        }
        let lse = max as f64 + logits.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln();
        let target = ids[i + 1] as usize;
        nll += lse - logits[target] as f64;
        hits += usize::from(best == target);
        top1.push(best as u32);
    }
    let mean = nll / n as f64;
    // A fingerprint of the exact tokens scored, so two runs can be checked to have used the
    // same text before their numbers are compared.
    let text_hash = ids.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &t| (h ^ t as u64).wrapping_mul(0x0000_0100_0000_01b3));
    let mut agree = String::from("-");
    if let Some(p) = args.str("agree-with") {
        let other = std::fs::read_to_string(p).unwrap_or_else(|e| die(format!("{p}: {e}")));
        let other: Vec<u32> = other.trim().split(',').filter_map(|s| s.parse().ok()).collect();
        if other.len() != top1.len() {
            die(format!("{p} holds {} positions but this run scored {}", other.len(), top1.len()));
        }
        let same = top1.iter().zip(&other).filter(|(a, b)| a == b).count();
        agree = format!("{:.4}", same as f64 / n as f64);
    }
    if let Some(p) = args.str("save-top1") {
        let s: Vec<String> = top1.iter().map(u32::to_string).collect();
        std::fs::write(p, s.join(",")).unwrap_or_else(|e| die(format!("{p}: {e}")));
    }
    println!(
        "EVAL tokens={n} text={text_hash:016x} mean_nll={mean:.4} ppl={:.3} top1_acc={:.4} top1_agree={agree}   [{label}]",
        mean.exp(),
        hits as f64 / n as f64
    );
}

// ---------------------------------------------------------------------------
// bench: every configuration in its own process, so peak memory is per configuration
// ---------------------------------------------------------------------------

struct Run {
    peak: f64,
    tokenizer: f64,
    load_s: f64,
    first_s: f64,
    tok_s: f64,
    per_token_mb: f64,
    wait_s: f64,
    decrypt_s: f64,
    new_tokens: usize,
    ids: Vec<u32>,
}

fn parse_result(stdout: &str) -> Option<Run> {
    let line = stdout.lines().find(|l| l.starts_with("RESULT "))?;
    let get = |k: &str| -> Option<&str> { line.split_whitespace().find_map(|kv| kv.strip_prefix(k)?.strip_prefix('=')) };
    let f = |k: &str| get(k)?.parse::<f64>().ok();
    let decode_forwards = f("decode_forwards")?;
    let decode_s = f("decode_s")?;
    // Weights read after loading: zero for resident modes, which read everything up front.
    let bytes = f("bytes_read")? - f("bytes_at_load")?;
    let forwards = decode_forwards + 1.0; // the prompt pass reads every layer once too
    Some(Run {
        peak: f("peak_rss")?,
        tokenizer: f("rss_tokenizer")?,
        load_s: f("load_s")?,
        first_s: f("first_s")?,
        tok_s: if decode_s > 0.0 { decode_forwards / decode_s } else { 0.0 },
        per_token_mb: bytes / forwards / (1024.0 * 1024.0),
        wait_s: f("wait_s")?,
        decrypt_s: f("decrypt_s")?,
        new_tokens: f("new_tokens")? as usize,
        ids: get("ids")?.split(',').filter_map(|s| s.parse().ok()).collect(),
    })
}

fn cmd_bench(args: &Args) {
    let exe = std::env::current_exe().unwrap_or_else(|e| die(e));
    let model = args.model_dir().to_string_lossy().to_string();
    let quick = args.has("quick");
    let max_new: usize = args.num("max-new", if quick { 16 } else { 32 });
    let prompt = args.str("prompt").unwrap_or(DEFAULT_PROMPT).to_string();
    let sealed = args.str("sealed").map(str::to_string);
    let key_file = args.str("key-file").map(str::to_string);
    let quantized = args.str("quantized").map(str::to_string);
    let q_sealed = args.str("quantized-sealed").map(str::to_string);
    if (sealed.is_some() || q_sealed.is_some()) && key_file.is_none() && std::env::var("LAYERSTREAM_KEY").is_err() {
        die("--sealed and --quantized-sealed need --key-file (or LAYERSTREAM_KEY)");
    }

    struct Cfg {
        label: String,
        extra: Vec<String>,
        max_new: usize,
    }
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let mut cfgs = vec![
        Cfg { label: "resident (everything in memory)".into(), extra: s(&["--mode", "resident"]), max_new },
        Cfg { label: "stream, 1 buffer (no read-ahead)".into(), extra: s(&["--mode", "stream", "--slots", "1"]), max_new },
        Cfg { label: "stream, 2 buffers (read-ahead)".into(), extra: s(&["--mode", "stream", "--slots", "2"]), max_new },
    ];
    let capped_new = if quick { 3 } else { 5 };
    for mbps in [2000.0, 500.0] {
        cfgs.push(Cfg {
            label: format!("stream, storage capped at {mbps:.0} MB/s"),
            extra: vec!["--mode".into(), "stream".into(), "--slots".into(), "2".into(), "--throttle-mbps".into(), format!("{mbps}")],
            max_new: capped_new,
        });
    }
    if let Some(path) = &sealed {
        let mut base = vec!["--sealed".to_string(), path.clone()];
        if let Some(k) = &key_file {
            base.extend(["--key-file".to_string(), k.clone()]);
        }
        let with = |more: &[&str]| {
            let mut v = base.clone();
            v.extend(more.iter().map(|x| x.to_string()));
            v
        };
        cfgs.push(Cfg { label: "encrypted: resident (decrypt once)".into(), extra: with(&["--mode", "resident"]), max_new });
        cfgs.push(Cfg { label: "encrypted: stream, 2 buffers".into(), extra: with(&["--mode", "stream", "--slots", "2", "--wipe"]), max_new });
        cfgs.push(Cfg {
            label: "encrypted: stream, capped at 500 MB/s".into(),
            extra: with(&["--mode", "stream", "--slots", "2", "--wipe", "--throttle-mbps", "500"]),
            max_new: capped_new,
        });
    }

    // Quantized models are different models from the bf16 one: their rows are labelled
    // "quantized" so they are compared with each other, not with the bf16 runs.
    if let Some(path) = &quantized {
        let with = |more: &[&str]| {
            let mut v = vec!["--quantized".to_string(), path.clone()];
            v.extend(more.iter().map(|x| x.to_string()));
            v
        };
        cfgs.push(Cfg { label: "quantized: resident".into(), extra: with(&["--mode", "resident"]), max_new });
        cfgs.push(Cfg { label: "quantized: stream, 1 buffer".into(), extra: with(&["--mode", "stream", "--slots", "1"]), max_new });
        cfgs.push(Cfg { label: "quantized: stream, 2 buffers".into(), extra: with(&["--mode", "stream", "--slots", "2"]), max_new });
        for mbps in [2000.0, 500.0] {
            cfgs.push(Cfg {
                label: format!("quantized: stream, capped at {mbps:.0} MB/s"),
                extra: with(&["--mode", "stream", "--slots", "2", "--throttle-mbps", &format!("{mbps}")]),
                max_new: capped_new,
            });
        }
    }
    if let Some(path) = &q_sealed {
        let mut base = vec!["--sealed".to_string(), path.clone()];
        if let Some(k) = &key_file {
            base.extend(["--key-file".to_string(), k.clone()]);
        }
        let with = |more: &[&str]| {
            let mut v = base.clone();
            v.extend(more.iter().map(|x| x.to_string()));
            v
        };
        cfgs.push(Cfg { label: "encrypted quantized: stream, 2 buffers".into(), extra: with(&["--mode", "stream", "--slots", "2", "--wipe"]), max_new });
        cfgs.push(Cfg {
            label: "encrypted quantized: stream, capped at 500 MB/s".into(),
            extra: with(&["--mode", "stream", "--slots", "2", "--wipe", "--throttle-mbps", "500"]),
            max_new: capped_new,
        });
    }

    println!("prompt: {prompt:?}  (greedy; every run in a fresh process)\n");
    let mut results: Vec<(String, Option<Run>)> = Vec::new();
    for cfg in &cfgs {
        let mut cmd = std::process::Command::new(&exe);
        cmd.args(["generate", "--report", "--prompt", &prompt, "--max-new", &cfg.max_new.to_string()]);
        cmd.args(["--model", &model]);
        cmd.args(&cfg.extra);
        eprint!("  running: {} ... ", cfg.label);
        let out = cmd.output().unwrap_or_else(|e| die(e));
        let run = if out.status.success() { parse_result(&String::from_utf8_lossy(&out.stdout)) } else { None };
        match &run {
            Some(r) => eprintln!("{:.1} tokens/s", r.tok_s),
            None => eprintln!("FAILED\n{}", String::from_utf8_lossy(&out.stderr)),
        }
        results.push((cfg.label.clone(), run));
    }

    let reference_for = |quantized: bool| {
        results.iter().find_map(|(l, r)| if l.contains("quantized") == quantized { r.as_ref().map(|r| r.ids.clone()) } else { None })
    };
    let (ref_plain, ref_quant) = (reference_for(false), reference_for(true));
    println!(
        "\n{:<48} {:>9} {:>9} {:>9} {:>9} {:>10} {:>9}  identical",
        "configuration", "peak MB", "model MB", "load s", "tokens/s", "read MB/tok", "wait s"
    );
    println!("{}", "-".repeat(122));
    for (label, run) in &results {
        let Some(r) = run else {
            println!("{label:<48} (failed)");
            continue;
        };
        let reference = if label.contains("quantized") { &ref_quant } else { &ref_plain };
        let n = r.ids.len().min(reference.as_ref().map_or(0, Vec::len));
        let same = reference.as_ref().is_some_and(|ref_ids| ref_ids[..n] == r.ids[..n]);
        let read = if r.per_token_mb > 0.0 { format!("{:.0}", r.per_token_mb) } else { "-".into() };
        println!(
            "{label:<48} {:>9.0} {:>9.0} {:>9.2} {:>9.1} {:>10} {:>9.2}  {}",
            mb(r.peak as u64),
            mb((r.peak - r.tokenizer).max(0.0) as u64),
            r.load_s,
            r.tok_s,
            read,
            r.wait_s,
            if same { format!("yes ({n} tokens)") } else { "NO".to_string() }
        );
        let _ = (r.first_s, r.decrypt_s, r.new_tokens);
    }
    println!("\nidentical: same tokens as the first run of its kind. Quantized models are different models from the bf16 one,");
    println!("so they are compared with each other, not with the bf16 runs.");
    println!("peak MB: whole process. model MB: peak minus the tokenizer's own footprint.");
    println!("tokens/s: generation after the first token. wait s: time the engine stalled waiting for weights.");
}

fn main() {
    let args = Args::parse();
    if args.has("help") {
        usage();
    }
    match args.cmd.as_str() {
        "generate" => cmd_generate(&args),
        "tokenize" => cmd_tokenize(&args),
        "gen-key" => cmd_gen_key(&args),
        "seal" => cmd_seal(&args),
        "quantize" => cmd_quantize(&args),
        "eval" => cmd_eval(&args),
        "bench" => cmd_bench(&args),
        _ => usage(),
    }
}

#[cfg(test)]
mod tests {
    use super::unescape_line;

    #[test]
    fn unescape_handles_every_escape_and_leaves_unknown_ones_alone() {
        assert_eq!(unescape_line(r"a\nb\tc\rd\\e"), "a\nb\tc\rd\\e");
        assert_eq!(unescape_line(r"keep \x and trailing \"), "keep \\x and trailing \\");
        assert_eq!(unescape_line(r"\\n"), "\\n"); // an escaped backslash then a plain n
        assert_eq!(unescape_line(""), "");
    }
}
