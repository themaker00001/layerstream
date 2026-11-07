//! Reader for the header of a `.safetensors` file.
//!
//! Layout: an 8-byte little-endian header length `N`, then `N` bytes of JSON mapping each
//! tensor name to `{dtype, shape, data_offsets}`, then the raw tensor data. Offsets in the
//! JSON are relative to the start of the data; this reader turns them into absolute file
//! offsets so tensors can be fetched later with a positioned read.

use crate::json::Json;
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute offset of the first byte in the file.
    pub start: u64,
    /// Absolute offset one past the last byte.
    pub end: u64,
}

impl TensorInfo {
    pub fn len(&self) -> usize {
        (self.end - self.start) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Debug, Clone)]
pub struct Header {
    pub tensors: HashMap<String, TensorInfo>,
    pub data_start: u64,
    pub file_len: u64,
}

fn dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "BF16" | "F16" => Some(2),
        "F32" => Some(4),
        _ => None,
    }
}

impl Header {
    pub fn read(path: &Path) -> Result<Header, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let file_len = file.metadata().map_err(|e| e.to_string())?.len();
        if file_len < 8 {
            return Err("file too short for a safetensors header".into());
        }
        let mut len_bytes = [0u8; 8];
        file.read_exact_at(&mut len_bytes, 0).map_err(|e| e.to_string())?;
        let n = u64::from_le_bytes(len_bytes);
        if n == 0 || n > 100 * 1024 * 1024 || 8 + n > file_len {
            return Err(format!("implausible safetensors header length {n}"));
        }
        let mut raw = vec![0u8; n as usize];
        file.read_exact_at(&mut raw, 8).map_err(|e| e.to_string())?;
        let text = std::str::from_utf8(&raw).map_err(|_| "header is not UTF-8".to_string())?;
        let json = Json::parse(text)?;
        let data_start = 8 + n;

        let mut tensors = HashMap::new();
        for (name, v) in json.as_obj().ok_or("header is not a JSON object")? {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v.get("dtype").and_then(Json::as_str).ok_or("tensor without dtype")?.to_string();
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(Json::as_arr)
                .ok_or("tensor without shape")?
                .iter()
                .map(|d| d.as_u64().map(|x| x as usize).ok_or("bad shape entry"))
                .collect::<Result<_, _>>()?;
            let offs = v.get("data_offsets").and_then(Json::as_arr).ok_or("tensor without data_offsets")?;
            if offs.len() != 2 {
                return Err(format!("{name}: data_offsets must have two entries"));
            }
            let a = offs[0].as_u64().ok_or("bad offset")?;
            let b = offs[1].as_u64().ok_or("bad offset")?;
            let (start, end) = (data_start + a, data_start + b);
            if end < start || end > file_len {
                return Err(format!("{name}: data range {start}..{end} is outside the file ({file_len} bytes)"));
            }
            let size = dtype_size(&dtype).ok_or_else(|| format!("{name}: unsupported dtype {dtype}"))?;
            let numel: usize = shape.iter().product();
            if (end - start) as usize != numel * size {
                return Err(format!("{name}: {} bytes does not match shape {shape:?} of {dtype}", end - start));
            }
            tensors.insert(name.clone(), TensorInfo { dtype, shape, start, end });
        }
        Ok(Header { tensors, data_start, file_len })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors.get(name).ok_or_else(|| format!("tensor {name} not found"))
    }

    /// Like [`tensor`](Self::tensor) but also checks dtype (BF16) and shape.
    pub fn bf16_tensor(&self, name: &str, shape: &[usize]) -> Result<&TensorInfo, String> {
        let t = self.tensor(name)?;
        if t.dtype != "BF16" {
            return Err(format!("{name}: expected BF16, found {}", t.dtype));
        }
        if t.shape != shape {
            return Err(format!("{name}: expected shape {shape:?}, found {:?}", t.shape));
        }
        Ok(t)
    }
}
