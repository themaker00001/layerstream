//! Where model bytes come from. A [`Source`] can fetch any layer block or any range of
//! embedding rows on demand; it knows nothing about memory budgets. What to keep in memory
//! and what to stream is decided one level up, by a [`crate::provider::Provider`].

use crate::sys;
use crate::weights::{Config, LayerLayout, ModelFiles};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct IoStats {
    /// Bytes read from storage.
    pub bytes_read: u64,
    /// Nanoseconds spent in reads.
    pub read_ns: u64,
    /// Nanoseconds spent decrypting (zero for an unencrypted source).
    pub decrypt_ns: u64,
}

pub trait Source: Send + Sync {
    fn cfg(&self) -> &Config;
    fn layout(&self) -> &LayerLayout;
    fn final_norm(&self) -> &[f32];
    /// Fill `buf` (exactly `layout().total` bytes) with layer `l`'s block.
    fn read_layer(&self, l: usize, buf: &mut [u8]) -> Result<(), String>;
    /// Fill `buf` (exactly `rows * row_bytes`) with embedding rows `first..first + rows`.
    fn read_embed_rows(&self, first: usize, rows: usize, buf: &mut [u8]) -> Result<(), String>;
    fn stats(&self) -> IoStats;
    fn describe(&self) -> String;
}

/// Caps how fast a source may read, to emulate a device with slow storage (an SD card, a
/// phone's flash) on a machine with a fast SSD. Reads are real; they are padded with sleeps so
/// that the long-run rate never exceeds the cap. Clones share one budget, so concurrent reads
/// share the cap instead of each getting its own.
#[derive(Clone)]
pub struct Throttle {
    bytes_per_s: f64,
    next_free: Arc<Mutex<Instant>>,
}

impl Throttle {
    pub fn new(mb_per_s: f64) -> Throttle {
        assert!(mb_per_s > 0.0);
        Throttle { bytes_per_s: mb_per_s * 1e6, next_free: Arc::new(Mutex::new(Instant::now())) }
    }

    /// Reserve time for reading `bytes`; the caller sleeps until the returned deadline.
    pub fn reserve(&self, bytes: usize) -> Instant {
        let dur = Duration::from_secs_f64(bytes as f64 / self.bytes_per_s);
        let mut next = self.next_free.lock().unwrap_or_else(|p| p.into_inner());
        // A reader that wakes a little late may start slightly in the past, so that sleep
        // overshoot is made up instead of lowering the long-run rate below the cap. The
        // allowance is small, so it cannot turn into a burst.
        let now = Instant::now();
        let floor = now.checked_sub(Duration::from_millis(5)).unwrap_or(now);
        let start = (*next).max(floor);
        *next = start + dur;
        *next
    }

    pub fn mb_per_s(&self) -> f64 {
        self.bytes_per_s / 1e6
    }

    pub fn wait_until(deadline: Instant) {
        let now = Instant::now();
        if deadline > now {
            std::thread::sleep(deadline - now);
        }
    }
}

/// One embedding row as f32.
pub fn embed_row_f32(src: &dyn Source, token: usize, out: &mut [f32]) -> Result<(), String> {
    let cfg = src.cfg();
    if token >= cfg.vocab {
        return Err(format!("token id {token} is outside the vocabulary ({})", cfg.vocab));
    }
    let mut raw = vec![0u8; cfg.row_bytes()];
    src.read_embed_rows(token, 1, &mut raw)?;
    crate::kernels::bf16_slice_to_f32(&raw, out);
    sys::wipe(&mut raw);
    Ok(())
}

/// Reads straight from the `.safetensors` file with positioned reads.
pub struct FileSource {
    files: ModelFiles,
    file: File,
    throttle: Option<Throttle>,
    bytes: AtomicU64,
    ns: AtomicU64,
}

impl FileSource {
    pub fn open(files: ModelFiles) -> Result<FileSource, String> {
        let file = File::open(&files.path).map_err(|e| format!("{}: {e}", files.path.display()))?;
        Ok(FileSource { files, file, throttle: None, bytes: AtomicU64::new(0), ns: AtomicU64::new(0) })
    }

    /// Cap read speed (see [`Throttle`]).
    pub fn with_throttle(mut self, throttle: Option<Throttle>) -> FileSource {
        self.throttle = throttle;
        self
    }

    fn read_at(&self, buf: &mut [u8], offset: u64) -> Result<(), String> {
        let deadline = self.throttle.as_ref().map(|t| t.reserve(buf.len()));
        let t = Instant::now();
        self.file.read_exact_at(buf, offset).map_err(|e| format!("read at {offset}: {e}"))?;
        if let Some(d) = deadline {
            Throttle::wait_until(d);
        }
        self.bytes.fetch_add(buf.len() as u64, Ordering::Relaxed);
        self.ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    }

    pub fn files(&self) -> &ModelFiles {
        &self.files
    }

    /// The final norm's raw bf16 bytes (used when sealing).
    pub fn read_final_norm_raw(&self, buf: &mut [u8]) -> Result<(), String> {
        if buf.len() != self.files.norm_len {
            return Err("final norm buffer has the wrong size".into());
        }
        self.read_at(buf, self.files.norm_start)
    }
}

impl Source for FileSource {
    fn cfg(&self) -> &Config {
        &self.files.cfg
    }

    fn layout(&self) -> &LayerLayout {
        &self.files.layout
    }

    fn final_norm(&self) -> &[f32] {
        &self.files.final_norm
    }

    fn read_layer(&self, l: usize, buf: &mut [u8]) -> Result<(), String> {
        if l >= self.files.cfg.layers {
            return Err(format!("layer {l} out of range"));
        }
        if buf.len() != self.files.layout.total {
            return Err("layer buffer has the wrong size".into());
        }
        self.read_at(buf, self.files.layer_start[l])
    }

    fn read_embed_rows(&self, first: usize, rows: usize, buf: &mut [u8]) -> Result<(), String> {
        let cfg = &self.files.cfg;
        if first + rows > cfg.vocab || buf.len() != rows * cfg.row_bytes() {
            return Err("embedding range out of bounds".into());
        }
        self.read_at(buf, self.files.embed_start + (first * cfg.row_bytes()) as u64)
    }

    fn stats(&self) -> IoStats {
        IoStats {
            bytes_read: self.bytes.load(Ordering::Relaxed),
            read_ns: self.ns.load(Ordering::Relaxed),
            decrypt_ns: 0,
        }
    }

    fn describe(&self) -> String {
        let mut s = String::from("plain file");
        if let Some(t) = &self.throttle {
            s.push_str(&format!(", storage capped at {:.0} MB/s", t.mb_per_s()));
        }
        s
    }
}
