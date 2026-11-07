//! Small operating-system helpers: peak memory use and wiping.

#[repr(C)]
struct Rusage {
    utime: [i64; 2],
    stime: [i64; 2],
    longs: [i64; 14], // ru_maxrss is the first
}

extern "C" {
    fn getrusage(who: i32, usage: *mut Rusage) -> i32;
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
