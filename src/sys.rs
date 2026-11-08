//! Small operating-system helpers: peak memory use, how much of a file the operating system
//! is holding in its cache, system memory pressure, and wiping.

use std::ffi::c_void;
use std::os::fd::AsRawFd;
use std::path::Path;

#[repr(C)]
struct Rusage {
    utime: [i64; 2],
    stime: [i64; 2],
    longs: [i64; 14], // ru_maxrss is the first
}

extern "C" {
    fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn mincore(addr: *mut c_void, len: usize, vec: *mut u8) -> i32;
    fn getpagesize() -> i32;
    #[cfg(target_os = "macos")]
    fn sysctlbyname(name: *const std::ffi::c_char, oldp: *mut c_void, oldlenp: *mut usize, newp: *mut c_void, newlen: usize) -> i32;
}

/// How much of the file at `path` the operating system currently holds in its file cache:
/// `(cached bytes, file bytes)`. That memory is real RAM, but it belongs to the system, not to
/// this process, so it is not part of [`peak_rss_bytes`]; the system may drop it at any time.
pub fn file_cache_bytes(path: &Path) -> Result<(u64, u64), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let len = file.metadata().map_err(|e| e.to_string())?.len() as usize;
    if len == 0 {
        return Ok((0, 0));
    }
    // SAFETY: plain libc call with no arguments.
    let page = unsafe { getpagesize() }.max(1) as usize;
    // SAFETY: a read-only shared mapping of a regular file. Nothing is ever read through it, so
    // no page is faulted in; it only lets `mincore` say which of its pages are already resident.
    let addr = unsafe { mmap(std::ptr::null_mut(), len, 1 /* PROT_READ */, 1 /* MAP_SHARED */, file.as_raw_fd(), 0) };
    if addr as isize == -1 {
        return Err(format!("{}: could not map the file to inspect its cache", path.display()));
    }
    let mut resident = vec![0u8; len.div_ceil(page)];
    // SAFETY: `addr`/`len` describe the mapping made above, and `resident` has one byte per page.
    let rc = unsafe { mincore(addr, len, resident.as_mut_ptr()) };
    // SAFETY: unmapping the mapping made above, once.
    unsafe { munmap(addr, len) };
    if rc != 0 {
        return Err(format!("{}: mincore failed", path.display()));
    }
    let pages = resident.iter().filter(|&&b| b & 1 != 0).count();
    Ok(((pages * page).min(len) as u64, len as u64))
}

/// System-wide memory state as macOS reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryState {
    /// The system's own "memory free percentage" (it counts memory it can reclaim, such as
    /// file cache, as free).
    pub free_percent: u32,
    /// 1 = normal, 2 = warning, 4 = critical.
    pub pressure: u32,
}

impl MemoryState {
    pub fn pressure_name(&self) -> &'static str {
        match self.pressure {
            1 => "normal",
            2 => "warning",
            4 => "critical",
            _ => "unknown",
        }
    }
}

/// Current system memory state; `None` where the operating system does not offer it (only
/// macOS is supported).
pub fn memory_state() -> Option<MemoryState> {
    #[cfg(target_os = "macos")]
    {
        let get = |name: &str| -> Option<i32> {
            let c = std::ffi::CString::new(name).ok()?;
            let mut v: i32 = 0;
            let mut len = std::mem::size_of::<i32>();
            // SAFETY: `v` and `len` describe a 4-byte output buffer; no new value is passed.
            let rc = unsafe { sysctlbyname(c.as_ptr(), &mut v as *mut i32 as *mut c_void, &mut len, std::ptr::null_mut(), 0) };
            (rc == 0 && len == 4).then_some(v)
        };
        let free = get("kern.memorystatus_level")?;
        let pressure = get("kern.memorystatus_vm_pressure_level")?;
        Some(MemoryState { free_percent: free.clamp(0, 100) as u32, pressure: pressure.max(0) as u32 })
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Peak resident set size of this process so far, in bytes (what `/usr/bin/time -l` calls
/// "maximum resident set size").
pub fn peak_rss_bytes() -> u64 {
    let mut ru = Rusage { utime: [0; 2], stime: [0; 2], longs: [0; 14] };
    // SAFETY: `ru` has the layout of `struct rusage` (144 bytes) on 64-bit macOS and Linux.
    let rc = unsafe { getrusage(0, &mut ru) };
    if rc != 0 {
        return 0;
    }
    let v = ru.longs[0].max(0) as u64;
    if cfg!(target_os = "macos") {
        v // bytes
    } else {
        v * 1024 // kilobytes on Linux
    }
}

/// Overwrite `buf` with zeros using volatile writes that the compiler may not remove.
/// Works a machine word at a time, so wiping tens of megabytes stays cheap.
pub fn wipe(buf: &mut [u8]) {
    // SAFETY: every bit pattern is a valid u64; `align_to_mut` only reinterprets the aligned middle.
    let (head, mid, tail) = unsafe { buf.align_to_mut::<u64>() };
    for b in head.iter_mut() {
        // SAFETY: `b` is a valid, exclusive reference.
        unsafe { std::ptr::write_volatile(b, 0) };
    }
    for w in mid.iter_mut() {
        // SAFETY: as above.
        unsafe { std::ptr::write_volatile(w, 0) };
    }
    for b in tail.iter_mut() {
        // SAFETY: as above.
        unsafe { std::ptr::write_volatile(b, 0) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_rss_is_plausible_and_grows_with_allocation() {
        let before = peak_rss_bytes();
        assert!(before > 1 << 20 && before < 1 << 36, "implausible peak rss {before}");
        let mut big = vec![0u8; 64 << 20];
        for i in (0..big.len()).step_by(4096) {
            big[i] = 1; // touch every page
        }
        std::hint::black_box(&big);
        assert!(peak_rss_bytes() >= before + (32 << 20));
    }

    #[test]
    fn a_file_just_written_and_read_is_reported_as_cached() {
        let path = std::env::temp_dir().join(format!("layerstream-cache-test-{}", std::process::id()));
        // Not a multiple of the page size, so the last page is partial.
        let data = vec![7u8; (5 << 20) + 123];
        std::fs::write(&path, &data).unwrap();
        let read_back = std::fs::read(&path).unwrap(); // pulls every page into the cache
        assert_eq!(read_back.len(), data.len());
        let (cached, total) = file_cache_bytes(&path).unwrap();
        assert_eq!(total, data.len() as u64);
        assert!(cached <= total, "cannot cache more than the file ({cached} of {total})");
        assert!(cached >= total * 9 / 10, "a file that was just read should be nearly all cached ({cached} of {total})");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn cache_probe_handles_empty_and_missing_files() {
        let path = std::env::temp_dir().join(format!("layerstream-cache-empty-{}", std::process::id()));
        std::fs::write(&path, b"").unwrap();
        assert_eq!(file_cache_bytes(&path).unwrap(), (0, 0));
        std::fs::remove_file(&path).unwrap();
        assert!(file_cache_bytes(&path).is_err());
    }

    #[test]
    fn memory_state_is_sane_where_available() {
        let state = memory_state();
        if cfg!(target_os = "macos") {
            let m = state.expect("macOS should report its memory state");
            assert!(m.free_percent <= 100);
            assert!([1, 2, 4].contains(&m.pressure), "unexpected pressure level {}", m.pressure);
            assert_ne!(m.pressure_name(), "unknown");
        } else {
            assert!(state.is_none());
        }
    }

    #[test]
    fn wipe_clears_every_byte_at_any_alignment() {
        for offset in 0..9 {
            for len in [0usize, 1, 7, 8, 9, 100, 1000] {
                let mut v = vec![0xABu8; len + 16];
                wipe(&mut v[offset.min(8)..offset.min(8) + len]);
                assert!(v[offset.min(8)..offset.min(8) + len].iter().all(|&b| b == 0));
            }
        }
    }
}
