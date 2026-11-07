//! A small JSON parser, enough for model configs, safetensors headers and tokenizer files.
//!
//! No dependencies are used on purpose. Objects keep their key order (as a vector of
//! pairs), which also keeps parsing a 150,000-entry vocabulary cheap.

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(text: &str) -> Result<Json, String> {
        let mut p = Parser { s: text.as_bytes(), i: 0 };
        p.skip_ws();
        let v = p.value(0)?;
        p.skip_ws();
        if p.i != p.s.len() {
            return Err(format!("trailing characters at byte {}", p.i));
        }
        Ok(v)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(items) => items.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// The number as an exact non-negative integer, or `None`.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 && *n < 9.0e15 => Some(*n as u64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(o) => Some(o),
            _ => None,
        }
    }

    /// Serialise back to JSON text (compact). Used by tests and tools that write small files.
    pub fn to_text(&self) -> String {
        match self {
            Json::Null => "null".into(),
            Json::Bool(b) => b.to_string(),
            Json::Num(n) => n.to_string(),
            Json::Str(s) => {
                let mut out = String::with_capacity(s.len() + 2);
                out.push('"');
                for c in s.chars() {
                    match c {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                        c => out.push(c),
                    }
                }
                out.push('"');
                out
            }
            Json::Arr(a) => format!("[{}]", a.iter().map(Json::to_text).collect::<Vec<_>>().join(",")),
            Json::Obj(o) => format!(
                "{{{}}}",
                o.iter().map(|(k, v)| format!("{}:{}", Json::Str(k.clone()).to_text(), v.to_text())).collect::<Vec<_>>().join(",")
            ),
        }
    }
}

/// Stream the entries of a flat `{"string": integer, ...}` object (such as a tokenizer
/// vocabulary) to `f` without building a tree, so a 150,000-entry file costs no memory beyond
/// what the caller keeps.
pub fn for_each_int_entry(text: &str, f: &mut dyn FnMut(String, u64) -> Result<(), String>) -> Result<(), String> {
    let mut p = Parser { s: text.as_bytes(), i: 0 };
    p.skip_ws();
    if p.s.get(p.i) != Some(&b'{') {
        return p.err("expected an object");
    }
    p.i += 1;
    p.skip_ws();
    if p.s.get(p.i) == Some(&b'}') {
        return Ok(());
    }
    loop {
        p.skip_ws();
        if p.s.get(p.i) != Some(&b'"') {
            return p.err("expected a string key");
        }
        let key = p.string()?;
        p.skip_ws();
        if p.s.get(p.i) != Some(&b':') {
            return p.err("expected ':'");
        }
        p.i += 1;
        p.skip_ws();
        let value = match p.number()? {
            Json::Num(n) if n >= 0.0 && n.fract() == 0.0 => n as u64,
            _ => return p.err("expected a non-negative integer"),
        };
        f(key, value)?;
        p.skip_ws();
        match p.s.get(p.i) {
            Some(b',') => p.i += 1,
            Some(b'}') => return Ok(()),
            _ => return p.err("expected ',' or '}'"),
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

const MAX_DEPTH: usize = 64;

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn err<T>(&self, msg: &str) -> Result<T, String> {
        Err(format!("JSON error at byte {}: {msg}", self.i))
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.s[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_DEPTH {
            return self.err("nesting too deep");
        }
        match self.s.get(self.i) {
            None => self.err("unexpected end"),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') if self.eat("true") => Ok(Json::Bool(true)),
            Some(b'f') if self.eat("false") => Ok(Json::Bool(false)),
            Some(b'n') if self.eat("null") => Ok(Json::Null),
            Some(c) if *c == b'-' || c.is_ascii_digit() => self.number(),
            Some(_) => self.err("unexpected character"),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, String> {
        self.i += 1; // {
        let mut items = Vec::new();
        self.skip_ws();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Obj(items));
        }
        loop {
            self.skip_ws();
            if self.s.get(self.i) != Some(&b'"') {
                return self.err("expected a string key");
            }
            let key = self.string()?;
            self.skip_ws();
            if self.s.get(self.i) != Some(&b':') {
                return self.err("expected ':'");
            }
            self.i += 1;
            self.skip_ws();
            let v = self.value(depth + 1)?;
            items.push((key, v));
            self.skip_ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(items));
                }
                _ => return self.err("expected ',' or '}'"),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, String> {
        self.i += 1; // [
        let mut items = Vec::new();
        self.skip_ws();
        if self.s.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return self.err("expected ',' or ']'"),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.s.len() {
            return self.err("short \\u escape");
        }
        let mut v = 0u32;
        for k in 0..4 {
            let d = (self.s[self.i + k] as char).to_digit(16);
            match d {
                Some(d) => v = v * 16 + d,
                None => return self.err("bad \\u escape"),
            }
        }
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(&c) = self.s.get(self.i) else {
                return self.err("unterminated string");
            };
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let Some(&e) = self.s.get(self.i) else {
                        return self.err("unterminated escape");
                    };
                    self.i += 1;
                    match e {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if !self.eat("\\u") {
                                    return self.err("lone high surrogate");
                                }
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return self.err("bad low surrogate");
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return self.err("lone low surrogate");
                            } else {
                                hi
                            };
                            let ch = char::from_u32(cp).ok_or_else(|| "bad code point".to_string())?;
                            let mut b = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut b).as_bytes());
                        }
                        _ => return self.err("unknown escape"),
                    }
                }
                _ => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| "invalid UTF-8 in string".to_string())
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        while self.i < self.s.len()
            && matches!(self.s[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
        {
            self.i += 1;
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).unwrap();
        text.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| format!("JSON error at byte {start}: bad number {text:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_values() {
        let j = Json::parse(r#" {"a": [1, 2.5, -3e2, true, null], "b": {"c": "x\ny"}, "d": ""} "#).unwrap();
        let a = j.get("a").unwrap().as_arr().unwrap();
        assert_eq!(a.len(), 5);
        assert_eq!(a[0].as_u64(), Some(1));
        assert_eq!(a[2].as_f64(), Some(-300.0));
        assert_eq!(a[3].as_bool(), Some(true));
        assert_eq!(j.get("b").unwrap().get("c").unwrap().as_str(), Some("x\ny"));
        assert_eq!(j.get("d").unwrap().as_str(), Some(""));
        assert_eq!(j.get("zzz"), None);
    }

    #[test]
    fn handles_unicode_escapes_and_raw_utf8() {
        let j = Json::parse(r#"["Ġa", "😀", "Ġ😀é"]"#).unwrap();
        let a = j.as_arr().unwrap();
        assert_eq!(a[0].as_str(), Some("\u{120}a"));
        assert_eq!(a[1].as_str(), Some("😀"));
        assert_eq!(a[2].as_str(), Some("Ġ😀é"));
    }

    #[test]
    fn rejects_malformed_input() {
        for bad in ["", "{", "[1,]", r#"{"a" 1}"#, r#"{"a":1,}"#, "tru", r#""\x""#, r#""\ud800""#, "1 2", "[1] x"] {
            assert!(Json::parse(bad).is_err(), "accepted {bad:?}");
        }
        assert!(Json::parse(&"[".repeat(200)).is_err(), "depth limit");
    }

    #[test]
    fn streams_flat_integer_objects() {
        let mut seen = Vec::new();
        for_each_int_entry(r#" { "a": 1, "b\n": 22 , "Ġc": 0 } "#, &mut |k, v| {
            seen.push((k, v));
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, vec![("a".to_string(), 1), ("b\n".to_string(), 22), ("\u{120}c".to_string(), 0)]);
        for bad in ["[]", r#"{"a": "x"}"#, r#"{"a": -1}"#, r#"{"a": 1.5}"#, r#"{"a" 1}"#, r#"{"a": 1,}"#] {
            assert!(for_each_int_entry(bad, &mut |_, _| Ok(())).is_err(), "accepted {bad:?}");
        }
        assert!(for_each_int_entry("{}", &mut |_, _| Ok(())).is_ok());
    }

    #[test]
    fn integer_accessor_is_strict() {
        assert_eq!(Json::Num(5.0).as_u64(), Some(5));
        assert_eq!(Json::Num(5.5).as_u64(), None);
        assert_eq!(Json::Num(-1.0).as_u64(), None);
    }
}
