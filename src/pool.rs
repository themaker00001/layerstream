//! A tiny persistent thread pool: `run(n_tasks, f)` calls `f(0..n_tasks)` across the
//! threads and the caller, handing out task indices from a shared atomic counter so that
//! fast cores take more tasks (this matters on chips with mixed performance and efficiency
//! cores).
//!
//! Spawning threads for every matrix multiply would cost more than it saves, hence the
//! persistent workers.

use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

type TaskFn = dyn Fn(usize) + Sync;

#[derive(Clone, Copy)]
struct Job {
    f: *const TaskFn,
    n_tasks: usize,
    next: *const AtomicUsize,
}

// SAFETY: the pointers are only dereferenced while `Pool::run` is blocked waiting for every
// worker to finish, so they stay valid; `TaskFn` is `Sync`, and `AtomicUsize` is `Sync`.
unsafe impl Send for Job {}

struct State {
    generation: u64,
    job: Option<Job>,
    pending: usize,
    shutdown: bool,
    panicked: bool,
}

struct Shared {
    state: Mutex<State>,
    start: Condvar,
    done: Condvar,
}

pub struct Pool {
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
    workers: usize,
    /// Only one job runs at a time.
    run_lock: Mutex<()>,
}

/// Turn a borrowed task function into a raw pointer with the borrow's lifetime erased.
///
/// The caller must keep the referent alive for as long as the pointer is dereferenced.
fn erase<'a>(f: &'a (dyn Fn(usize) + Sync + 'a)) -> *const TaskFn {
    // SAFETY: only the lifetime bound of the trait object changes; layout is identical.
    unsafe {
        std::mem::transmute::<*const (dyn Fn(usize) + Sync + 'a), *const TaskFn>(f as *const (dyn Fn(usize) + Sync + 'a))
    }
}

/// Run tasks until the shared counter is exhausted.
///
/// # Safety
/// `job.f` and `job.next` must be valid for the whole call.
unsafe fn run_tasks(job: Job) {
    let f = &*job.f;
    let next = &*job.next;
    loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= job.n_tasks {
            break;
        }
        f(i);
    }
}

impl Pool {
    /// A pool that uses `threads` threads in total, counting the caller.
    pub fn new(threads: usize) -> Pool {
        let workers = threads.max(1) - 1;
        let shared = Arc::new(Shared {
            state: Mutex::new(State { generation: 0, job: None, pending: 0, shutdown: false, panicked: false }),
            start: Condvar::new(),
            done: Condvar::new(),
        });
        let handles = (0..workers)
            .map(|_| {
                let shared = Arc::clone(&shared);
                std::thread::spawn(move || worker(shared))
            })
            .collect();
        Pool { shared, handles, workers, run_lock: Mutex::new(()) }
    }

    pub fn threads(&self) -> usize {
        self.workers + 1
    }

    /// Call `f(i)` for every `i` in `0..n_tasks`, spread across the pool. Blocks until all
    /// tasks are done. Panics if a task panicked.
    pub fn run(&self, n_tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        if n_tasks == 0 {
            return;
        }
        if self.workers == 0 || n_tasks == 1 {
            (0..n_tasks).for_each(f);
            return;
        }
        let one_at_a_time = self.run_lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let next = AtomicUsize::new(0);
        // `run` does not return (even by panic) until every worker has acknowledged the job
        // and stopped touching `f` and `next`, so erasing the borrow's lifetime is sound.
        let job = Job { f: erase(f), n_tasks, next: &next as *const AtomicUsize };
        {
            let mut st = self.shared.state.lock().unwrap();
            st.job = Some(job);
            st.pending = self.workers;
            st.panicked = false;
            st.generation += 1;
            self.shared.start.notify_all();
        }
        // The caller works too. Catch a panic so we can still wait for the workers.
        let mine = catch_unwind(AssertUnwindSafe(|| unsafe { run_tasks(job) }));
        let mut st = self.shared.state.lock().unwrap();
        while st.pending > 0 {
            st = self.shared.done.wait(st).unwrap();
        }
        st.job = None;
        let worker_panicked = st.panicked;
        drop(st);
        // Release before any panic below, so a failed job does not poison later ones.
        drop(one_at_a_time);
        if let Err(p) = mine {
            resume_unwind(p);
        }
        if worker_panicked {
            panic!("a pool task panicked");
        }
    }
}

fn worker(shared: Arc<Shared>) {
    let mut seen = 0u64;
    loop {
        let job = {
            let mut st = shared.state.lock().unwrap();
            while st.generation == seen && !st.shutdown {
                st = shared.start.wait(st).unwrap();
            }
            if st.shutdown {
                return;
            }
            seen = st.generation;
            st.job.expect("a job accompanies every generation")
        };
        let result = catch_unwind(AssertUnwindSafe(|| unsafe { run_tasks(job) }));
        let mut st = shared.state.lock().unwrap();
        if result.is_err() {
            st.panicked = true;
        }
        st.pending -= 1;
        if st.pending == 0 {
            shared.done.notify_all();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut st = self.shared.state.lock().unwrap();
            st.shutdown = true;
            self.shared.start.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// A raw pointer that may be shared across pool tasks that write to disjoint ranges.
pub struct SyncPtr<T>(*mut T);

// SAFETY: users promise that concurrent tasks touch disjoint elements.
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    pub fn new(slice: &mut [T]) -> Self {
        SyncPtr(slice.as_mut_ptr())
    }

    /// Method (not field) access so closures capture the whole `Sync` wrapper.
    pub fn get(&self) -> *mut T {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_runs_exactly_once() {
        let pool = Pool::new(6);
        for n in [0usize, 1, 2, 5, 97, 1000] {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            pool.run(n, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "n = {n}");
        }
    }

    #[test]
    fn disjoint_writes_through_syncptr() {
        let pool = Pool::new(4);
        let mut out = vec![0u32; 10_000];
        let p = SyncPtr::new(&mut out);
        pool.run(100, &|t| {
            let chunk = unsafe { std::slice::from_raw_parts_mut(p.get().add(t * 100), 100) };
            for (k, v) in chunk.iter_mut().enumerate() {
                *v = (t * 100 + k) as u32;
            }
        });
        assert!(out.iter().enumerate().all(|(i, v)| *v == i as u32));
    }

    #[test]
    fn many_back_to_back_runs() {
        let pool = Pool::new(8);
        let total = AtomicUsize::new(0);
        for _ in 0..2000 {
            pool.run(16, &|_| {
                total.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(total.load(Ordering::Relaxed), 2000 * 16);
    }

    #[test]
    fn a_panicking_task_is_reported_and_the_pool_survives() {
        let pool = Pool::new(4);
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.run(8, &|i| {
                if i == 3 {
                    panic!("boom");
                }
            })
        }));
        assert!(r.is_err());
        let count = AtomicUsize::new(0);
        pool.run(8, &|_| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn single_thread_pool_runs_inline() {
        let pool = Pool::new(1);
        let count = AtomicUsize::new(0);
        pool.run(10, &|_| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), 10);
    }
}
