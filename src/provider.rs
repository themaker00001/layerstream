//! Providers decide where the weights live while the model runs.
//!
//! * [`ResidentProvider`] loads everything into memory once (the usual approach).
//! * [`StreamProvider`] keeps only `slots` layer-sized buffers in memory. A background
//!   thread fills the next buffer from the [`Source`] while the engine computes on the
//!   current one, so with two slots reading overlaps with computing.
//!
//! The engine asks for items in a fixed order each forward pass (layer 0 .. n-1, then the
//! output-head chunks), which is what makes read-ahead possible.

use crate::source::{embed_row_f32, IoStats, Source};
use crate::sys;
use crate::weights::{Config, LayerLayout};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

/// Embedding rows per output-head chunk. The tied embedding table doubles as the output
/// layer, so computing the logits means streaming the whole table once per token.
pub const HEAD_ROWS: usize = 4096;

pub struct HeadChunk<'a> {
    pub first: usize,
    pub rows: usize,
    pub data: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct ProviderReport {
    pub name: String,
    /// Weight bytes held in memory for the whole run.
    pub resident_weight_bytes: u64,
    pub io: IoStats,
    /// Time the engine spent blocked waiting for data (zero when everything is resident).
    pub wait_ns: u64,
}

pub trait Provider {
    fn cfg(&self) -> &Config;
    fn layout(&self) -> &LayerLayout;
    fn final_norm(&self) -> &[f32];
    fn embed_row(&mut self, token: usize, out: &mut [f32]) -> Result<(), String>;
    fn begin_forward(&mut self, with_head: bool) -> Result<(), String>;
    /// Layer `l`'s block, valid until the next call on this provider.
    fn layer(&mut self, l: usize) -> Result<&[u8], String>;
    /// Output-head chunk `c`, valid until the next call on this provider.
    fn head_chunk(&mut self, c: usize) -> Result<HeadChunk<'_>, String>;
    fn end_forward(&mut self) -> Result<(), String>;
    fn report(&self) -> ProviderReport;

    fn n_head_chunks(&self) -> usize {
        self.cfg().vocab.div_ceil(HEAD_ROWS)
    }
}

// ---------------------------------------------------------------------------
// Resident
// ---------------------------------------------------------------------------

pub struct ResidentProvider {
    source: Arc<dyn Source>,
    layers: Vec<Vec<u8>>,
    embed: Vec<u8>,
}

impl ResidentProvider {
    /// Read the whole model into memory.
    pub fn load(source: Arc<dyn Source>) -> Result<ResidentProvider, String> {
        let cfg = source.cfg().clone();
        let total = source.layout().total;
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let mut b = vec![0u8; total];
            source.read_layer(l, &mut b)?;
            layers.push(b);
        }
        let mut embed = vec![0u8; cfg.vocab * cfg.row_bytes()];
        source.read_embed_rows(0, cfg.vocab, &mut embed)?;
        Ok(ResidentProvider { source, layers, embed })
    }
}

impl Provider for ResidentProvider {
    fn cfg(&self) -> &Config {
        self.source.cfg()
    }

    fn layout(&self) -> &LayerLayout {
        self.source.layout()
    }

    fn final_norm(&self) -> &[f32] {
        self.source.final_norm()
    }

    fn embed_row(&mut self, token: usize, out: &mut [f32]) -> Result<(), String> {
        let rb = self.source.cfg().row_bytes();
        if token >= self.source.cfg().vocab {
            return Err(format!("token id {token} is outside the vocabulary"));
        }
        crate::kernels::bf16_slice_to_f32(&self.embed[token * rb..(token + 1) * rb], out);
        Ok(())
    }

    fn begin_forward(&mut self, _with_head: bool) -> Result<(), String> {
        Ok(())
    }

    fn layer(&mut self, l: usize) -> Result<&[u8], String> {
        self.layers.get(l).map(|v| v.as_slice()).ok_or_else(|| format!("layer {l} out of range"))
    }

    fn head_chunk(&mut self, c: usize) -> Result<HeadChunk<'_>, String> {
        let cfg = self.source.cfg();
        let first = c * HEAD_ROWS;
        if first >= cfg.vocab {
            return Err("head chunk out of range".into());
        }
        let rows = HEAD_ROWS.min(cfg.vocab - first);
        let rb = cfg.row_bytes();
        Ok(HeadChunk { first, rows, data: &self.embed[first * rb..(first + rows) * rb] })
    }

    fn end_forward(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn report(&self) -> ProviderReport {
        ProviderReport {
            name: format!("resident ({})", self.source.describe()),
            resident_weight_bytes: (self.layers.iter().map(Vec::len).sum::<usize>() + self.embed.len()) as u64,
            io: self.source.stats(),
            wait_ns: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Item {
    Layer(usize),
    Head(usize),
}

struct Job {
    item: Item,
    buf: Vec<u8>,
}

struct Done {
    item: Item,
    buf: Vec<u8>,
    len: usize,
    res: Result<(), String>,
}

pub struct StreamProvider {
    source: Arc<dyn Source>,
    job_tx: Option<Sender<Job>>,
    done_rx: Receiver<Done>,
    worker: Option<JoinHandle<()>>,
    free: Vec<Vec<u8>>,
    current: Option<(Vec<u8>, usize)>,
    schedule: Vec<Item>,
    next_issue: usize,
    next_take: usize,
    in_flight: usize,
    slots: usize,
    buf_size: usize,
    wipe_used: bool,
    wait_ns: u64,
}

impl StreamProvider {
    /// `slots` is the number of layer-sized buffers: 1 reads and computes in turn, 2 reads
    /// the next layer while the current one is being computed. With `wipe_used` every buffer
    /// is overwritten with zeros as soon as the engine is done with it.
    pub fn new(source: Arc<dyn Source>, slots: usize, wipe_used: bool) -> StreamProvider {
        let slots = slots.max(1);
        let cfg = source.cfg().clone();
        let layer_bytes = source.layout().total;
        let buf_size = layer_bytes.max(HEAD_ROWS.min(cfg.vocab) * cfg.row_bytes());
        let (job_tx, job_rx) = channel::<Job>();
        let (done_tx, done_rx) = channel::<Done>();
        let src = Arc::clone(&source);
        let worker = std::thread::spawn(move || {
            for job in job_rx {
                let mut buf = job.buf;
                let (len, res) = match job.item {
                    Item::Layer(l) => (layer_bytes, src.read_layer(l, &mut buf[..layer_bytes])),
                    Item::Head(c) => {
                        let first = c * HEAD_ROWS;
                        let rows = HEAD_ROWS.min(cfg.vocab - first);
                        let len = rows * cfg.row_bytes();
                        (len, src.read_embed_rows(first, rows, &mut buf[..len]))
                    }
                };
                if done_tx.send(Done { item: job.item, buf, len, res }).is_err() {
                    break;
                }
            }
        });
        StreamProvider {
            source,
            job_tx: Some(job_tx),
            done_rx,
            worker: Some(worker),
            free: (0..slots).map(|_| vec![0u8; buf_size]).collect(),
            current: None,
            schedule: Vec::new(),
            next_issue: 0,
            next_take: 0,
            in_flight: 0,
            slots,
            buf_size,
            wipe_used,
            wait_ns: 0,
        }
    }

    fn recycle_current(&mut self) {
        if let Some((mut buf, len)) = self.current.take() {
            if self.wipe_used {
                sys::wipe(&mut buf[..len]);
            }
            self.free.push(buf);
        }
    }

    fn issue_more(&mut self) {
        while self.next_issue < self.schedule.len() {
            let Some(buf) = self.free.pop() else { break };
            let job = Job { item: self.schedule[self.next_issue], buf };
            if self.job_tx.as_ref().expect("worker alive").send(job).is_err() {
                break;
            }
            self.next_issue += 1;
            self.in_flight += 1;
        }
    }

    fn take(&mut self, want: Item) -> Result<(), String> {
        if self.schedule.get(self.next_take) != Some(&want) {
            return Err(format!("provider asked for {want:?} out of schedule order"));
        }
        self.recycle_current();
        self.issue_more();
        let t = Instant::now();
        let done = self.done_rx.recv().map_err(|_| "the read-ahead thread stopped".to_string())?;
        self.wait_ns += t.elapsed().as_nanos() as u64;
        self.in_flight -= 1;
        let Done { item, buf, len, res } = done;
        if let Err(e) = res {
            self.free.push(buf);
            return Err(e);
        }
        if item != want {
            self.free.push(buf);
            return Err(format!("read-ahead delivered {item:?} but {want:?} was expected"));
        }
        self.current = Some((buf, len));
        self.next_take += 1;
        Ok(())
    }
}

impl Provider for StreamProvider {
    fn cfg(&self) -> &Config {
        self.source.cfg()
    }

    fn layout(&self) -> &LayerLayout {
        self.source.layout()
    }

    fn final_norm(&self) -> &[f32] {
        self.source.final_norm()
    }

    fn embed_row(&mut self, token: usize, out: &mut [f32]) -> Result<(), String> {
        embed_row_f32(self.source.as_ref(), token, out)
    }

    fn begin_forward(&mut self, with_head: bool) -> Result<(), String> {
        // Recover from an earlier pass that stopped early.
        self.recycle_current();
        while self.in_flight > 0 {
            let done = self.done_rx.recv().map_err(|_| "the read-ahead thread stopped".to_string())?;
            self.in_flight -= 1;
            self.free.push(done.buf);
        }
        let cfg = self.source.cfg();
        self.schedule = (0..cfg.layers).map(Item::Layer).collect();
        if with_head {
            self.schedule.extend((0..cfg.vocab.div_ceil(HEAD_ROWS)).map(Item::Head));
        }
        self.next_issue = 0;
        self.next_take = 0;
        self.issue_more();
        Ok(())
    }

    fn layer(&mut self, l: usize) -> Result<&[u8], String> {
        self.take(Item::Layer(l))?;
        let (buf, len) = self.current.as_ref().unwrap();
        Ok(&buf[..*len])
    }

    fn head_chunk(&mut self, c: usize) -> Result<HeadChunk<'_>, String> {
        self.take(Item::Head(c))?;
        let (buf, len) = self.current.as_ref().unwrap();
        let first = c * HEAD_ROWS;
        let rows = len / self.source.cfg().row_bytes();
        Ok(HeadChunk { first, rows, data: &buf[..*len] })
    }

    fn end_forward(&mut self) -> Result<(), String> {
        self.recycle_current();
        if self.next_take != self.schedule.len() || self.in_flight != 0 {
            return Err("forward pass ended before every scheduled item was consumed".into());
        }
        Ok(())
    }

    fn report(&self) -> ProviderReport {
        ProviderReport {
            name: format!("stream, {} slot{} ({})", self.slots, if self.slots == 1 { "" } else { "s" }, self.source.describe()),
            resident_weight_bytes: (self.slots * self.buf_size) as u64,
            io: self.source.stats(),
            wait_ns: self.wait_ns,
        }
    }
}

impl Drop for StreamProvider {
    fn drop(&mut self) {
        self.job_tx.take(); // closes the channel, which ends the worker's loop
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
        // Do not leave weights lying in freed memory.
        if self.wipe_used {
            for b in self.free.iter_mut() {
                sys::wipe(b);
            }
        }
    }
}
