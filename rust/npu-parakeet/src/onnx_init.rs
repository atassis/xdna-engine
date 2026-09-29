//! Minimal ONNX-initializer reader: extracts named weight tensors straight out of an .onnx
//! file's protobuf, with no onnxruntime and no general ONNX-execution crate (per the port's
//! whole point -- deleting that dependency class, not swapping it for a bigger one). Reads only
//! what `decoder.rs` needs: `ModelProto.graph` (field 7), `GraphProto.node`/`initializer`
//! (fields 1/5), `NodeProto.input`/`op_type` (fields 1/4), `TensorProto.dims`/`data_type`/
//! `name`/`raw_data` (fields 1/2/8/9) -- field numbers from onnx/onnx.proto (opset-independent).
//! Every other field in these messages is skipped unread.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub type Result<T> = std::result::Result<T, String>;

/// A `TensorProto` initializer, decoded to native f32 (`data_type` 1 = FLOAT; nothing here uses
/// another dtype).
pub struct Tensor {
    pub dims: Vec<i64>,
    pub data: Vec<f32>,
}

pub struct OnnxGraph {
    pub initializers: HashMap<String, Tensor>,
    /// (op_type, input names), in file order -- enough to find e.g. the two "LSTM" nodes and
    /// resolve their W/R/B initializer names (LSTM weights get compiler-generated names, so
    /// there is no fixed string to look them up by; the node's input list is the only way).
    pub nodes: Vec<(String, Vec<String>)>,
}

// --- protobuf wire format: tag = (field_num << 3) | wire_type; wire_type 0 = varint, 2 = length-delimited. ---

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    fn eof(&self) -> bool {
        self.pos >= self.buf.len()
    }
    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let b = *self.buf.get(self.pos).ok_or("varint: truncated")?;
            self.pos += 1;
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift >= 64 {
                return Err("varint: too long".into());
            }
        }
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or("bytes: overflow")?;
        let s = self.buf.get(self.pos..end).ok_or("bytes: truncated")?;
        self.pos = end;
        Ok(s)
    }
    fn len_delim(&mut self) -> Result<&'a [u8]> {
        let n = self.varint()? as usize;
        self.bytes(n)
    }
    /// (field_num, wire_type)
    fn tag(&mut self) -> Result<(u32, u32)> {
        let t = self.varint()?;
        Ok(((t >> 3) as u32, (t & 7) as u32))
    }
    fn skip(&mut self, wire_type: u32) -> Result<()> {
        match wire_type {
            0 => {
                self.varint()?;
            }
            1 => {
                self.bytes(8)?;
            }
            2 => {
                self.len_delim()?;
            }
            5 => {
                self.bytes(4)?;
            }
            _ => return Err(format!("skip: unsupported wire_type {wire_type}")),
        }
        Ok(())
    }
}

fn parse_tensor(buf: &[u8]) -> Result<(String, Tensor)> {
    let mut r = Reader::new(buf);
    let mut dims = Vec::new();
    let mut data_type = 0i32;
    let mut name = String::new();
    let mut raw_data: Option<&[u8]> = None;
    let mut float_data: Vec<f32> = Vec::new();
    while !r.eof() {
        let (field, wt) = r.tag()?;
        match field {
            1 => {
                // repeated int64 dims = 1. onnx's own protobuf writer emits this UNPACKED (one
                // varint-wire tag per dim, not the proto3 packed-by-default LEN encoding) --
                // confirmed by hex-dumping a real initializer; handle both to not re-break on a
                // differently-produced .onnx file.
                if wt == 0 {
                    dims.push(r.varint()? as i64);
                } else {
                    let d = r.len_delim()?;
                    let mut dr = Reader::new(d);
                    while !dr.eof() {
                        dims.push(dr.varint()? as i64);
                    }
                }
            }
            2 => data_type = r.varint()? as i32,
            4 => {
                // repeated float float_data = 4 (packed fixed32, LEN)
                let d = r.len_delim()?;
                for chunk in d.chunks_exact(4) {
                    float_data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
            }
            8 => name = String::from_utf8_lossy(r.len_delim()?).into_owned(),
            9 => raw_data = Some(r.len_delim()?),
            _ => r.skip(wt)?,
        }
    }
    if data_type != 1 {
        return Err(format!("tensor {name}: data_type {data_type} != FLOAT(1)"));
    }
    let data = if let Some(raw) = raw_data {
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    } else {
        float_data
    };
    let want: i64 = dims.iter().product::<i64>().max(0);
    if data.len() as i64 != want {
        return Err(format!(
            "tensor {name}: {} elements, dims {:?} want {want}",
            data.len(),
            dims
        ));
    }
    Ok((name, Tensor { dims, data }))
}

fn parse_node(buf: &[u8]) -> Result<(String, Vec<String>)> {
    let mut r = Reader::new(buf);
    let mut op_type = String::new();
    let mut inputs = Vec::new();
    while !r.eof() {
        let (field, wt) = r.tag()?;
        match field {
            1 => inputs.push(String::from_utf8_lossy(r.len_delim()?).into_owned()),
            4 => op_type = String::from_utf8_lossy(r.len_delim()?).into_owned(),
            _ => r.skip(wt)?,
        }
    }
    Ok((op_type, inputs))
}

fn parse_graph(buf: &[u8]) -> Result<OnnxGraph> {
    let mut r = Reader::new(buf);
    let mut initializers = HashMap::new();
    let mut nodes = Vec::new();
    while !r.eof() {
        let (field, wt) = r.tag()?;
        match field {
            1 => nodes.push(parse_node(r.len_delim()?)?),
            5 => {
                let (name, t) = parse_tensor(r.len_delim()?)?;
                initializers.insert(name, t);
            }
            _ => r.skip(wt)?,
        }
    }
    Ok(OnnxGraph { initializers, nodes })
}

/// Parse `path` (a `ModelProto`) and return its graph's initializers + node list.
pub fn load(path: &Path) -> Result<OnnxGraph> {
    let buf = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut r = Reader::new(&buf);
    while !r.eof() {
        let (field, wt) = r.tag()?;
        if field == 7 {
            return parse_graph(r.len_delim()?);
        }
        r.skip(wt)?;
    }
    Err(format!("{}: no graph (field 7) found", path.display()))
}
