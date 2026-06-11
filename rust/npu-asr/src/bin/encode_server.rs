//! Persistent NPU encode coprocess for the ASR service (Task 2). Loads the 16-block encoder +
//! weights ONCE, holds the (single-tenant) NPU, and serves encode requests over a framed
//! stdin/stdout protocol — so a Python front-end (onnx-asr preproc+decode) can use our fast Rust
//! NPU encoder without reloading weights per request.
//!
//! NPU is single-tenant — flm-asr.service/voxd.service must be stopped while this runs.
//!
//! Protocol (little-endian, on stdin/stdout):
//!   request : u32 T            (mel frames, 1..=any; truncated to 1600 = 16 s window)
//!             f32 * (64*T)     mel features, channel-major [64, T]
//!   response: u32 valid_len    (= (min(T,1600)-1)/4 + 1)
//!             f32 * (768*400)  encoded, row-major [768, 400]
//! A request with T==0 is a clean shutdown signal.

use std::io::{stdin, stdout, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::rc::Rc;

use ndarray::prelude::*;
use npu_asr::encoder::{subsample, Encoder};
use npu_asr::weights::WeightStore;
use npu_xrt::Device;

const MEL: usize = 64;
const WIN: usize = 1600;
const T_OUT: usize = 400;
const D: usize = 768;

fn read_u32<R: Read>(r: &mut R) -> Option<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).ok().map(|_| u32::from_le_bytes(b))
}

fn main() {
    let root = Path::new(".");
    let ws = WeightStore::load(&root.join("artifacts/encoder")).expect("load encoder weights");
    let dev = Rc::new(Device::open(0).expect("open NPU (stop flm-asr/voxd first)"));
    let enc = Encoder::new(dev, root, &ws, 16);
    eprintln!("[encode_server] ready — 16-block encoder loaded, NPU open");

    let mut inp = BufReader::new(stdin());
    let mut out = BufWriter::new(stdout());
    loop {
        let t = match read_u32(&mut inp) {
            Some(0) | None => break,
            Some(t) => t as usize,
        };
        let mut buf = vec![0u8; t * MEL * 4];
        if inp.read_exact(&mut buf).is_err() {
            break;
        }
        let teff = t.min(WIN);
        // mel [64,T] channel-major -> padded [64,1600]
        let mut audio = Array2::<f32>::zeros((MEL, WIN));
        for c in 0..MEL {
            let cbase = c * t;
            for ti in 0..teff {
                let idx = (cbase + ti) * 4;
                audio[[c, ti]] = f32::from_le_bytes([buf[idx], buf[idx + 1], buf[idx + 2], buf[idx + 3]]);
            }
        }
        let valid = (teff.max(1) - 1) / 4 + 1;
        let x0 = subsample(&ws, &audio);
        let outs = enc.forward_blocks(&x0, valid); // mask padded frames past valid_len
        let encoded = outs[outs.len() - 1].t().to_owned(); // [768,400]

        out.write_all(&(valid as u32).to_le_bytes()).unwrap();
        let mut resp = Vec::with_capacity(D * T_OUT * 4);
        for v in encoded.iter() {
            resp.extend_from_slice(&v.to_le_bytes());
        }
        out.write_all(&resp).unwrap();
        out.flush().unwrap();
    }
    eprintln!("[encode_server] shutdown");
}
