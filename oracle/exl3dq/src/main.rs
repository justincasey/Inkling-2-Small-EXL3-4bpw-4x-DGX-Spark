//! exl3dq: dumps REXL3's CPU EXL3 reference (gb-exl3) as .npy files, so the oracle's numpy port (`oracle/exl3.py`) can be
//! checked against it bit for bit.
//!
//!   exl3dq table   <3inst|mcg|mul1> <out.npy>            codebook, uint16 [65536] fp16 bits
//!   exl3dq groups  <dir>                                  one JSON line per EXL3 group gb-exl3's scan accepts (and refuses)
//!   exl3dq wq      <dir> <prefix> <out.npy>               W_q, uint16 [K, N] fp16 bits (gb_exl3::unpack)
//!   exl3dq dense   <dir> <prefix> <out.npy>               effective weight, float64 [K, N] (gb_exl3::decode_weight)
//!   exl3dq forward <dir> <prefix> <x.npy> <out.npy>       layer output, float64 [rows, N] (gb_exl3::forward), x float64 [rows, K]
//!
//! `<dir>` is a folder of .safetensors shards, scanned with `gb_exl3::checkpoint::scan(dir, read_markers = true)`, so the
//! group must pass the engine's own checks (shapes, scales, codebook marker value).
use gb_exl3::checkpoint::{scan, Checkpoint, LoadedGroup};
use gb_exl3::{codebook, decode_weight, forward, unpack, Codebook};
use std::io::{Read, Write};
use std::path::Path;
use std::process::ExitCode;

fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let shape_s = if shape.len() == 1 { format!("({},)", dims[0]) } else { format!("({})", dims.join(", ")) };
    let mut dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_s}, }}");
    // magic (6) + version (2) + length (2) + dict + '\n' is a multiple of 64
    while (10 + dict.len() + 1) % 64 != 0 {
        dict.push(' ');
    }
    dict.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(dict.len() as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out
}

fn write_u16(path: &str, shape: &[usize], data: &[u16]) -> std::io::Result<()> {
    let mut b = npy_header("<u2", shape);
    b.reserve(data.len() * 2);
    for v in data {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::File::create(path)?.write_all(&b)
}

fn write_f64(path: &str, shape: &[usize], data: &[f64]) -> std::io::Result<()> {
    let mut b = npy_header("<f8", shape);
    b.reserve(data.len() * 8);
    for v in data {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::File::create(path)?.write_all(&b)
}

/// A C-order little-endian float64 .npy of any rank (version 1 or 2 header).
fn read_f64(path: &str) -> Result<(Vec<usize>, Vec<f64>), String> {
    let mut b = Vec::new();
    std::fs::File::open(path).and_then(|mut f| f.read_to_end(&mut b)).map_err(|e| format!("{path}: {e}"))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(format!("{path}: not a .npy file"));
    }
    let (hlen, start) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
        v => return Err(format!("{path}: .npy version {v}")),
    };
    let header = std::str::from_utf8(&b[start..start + hlen]).map_err(|e| e.to_string())?;
    if !header.contains("'descr': '<f8'") || header.contains("'fortran_order': True") {
        return Err(format!("{path}: need C-order '<f8', header {header}"));
    }
    let s = header.split("'shape':").nth(1).ok_or("no shape")?;
    let s = &s[s.find('(').ok_or("no shape tuple")? + 1..s.find(')').ok_or("no shape tuple")?];
    let shape: Vec<usize> =
        s.split(',').map(str::trim).filter(|t| !t.is_empty()).map(|t| t.parse::<usize>().map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
    let data: Vec<f64> = b[start + hlen..].chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    if data.len() != shape.iter().product::<usize>() {
        return Err(format!("{path}: {} values for shape {shape:?}", data.len()));
    }
    Ok((shape, data))
}

fn open(dir: &str) -> Result<Checkpoint, String> {
    scan(Path::new(dir), true).map_err(|e| format!("{dir}: {e}"))
}

fn load(dir: &str, prefix: &str) -> Result<LoadedGroup, String> {
    let ck = open(dir)?;
    if let Some(why) = ck.bad.get(prefix) {
        return Err(format!("{prefix}: refused by gb-exl3: {why}"));
    }
    ck.load_group(prefix).map_err(|e| e.to_string())
}

fn run(args: &[String]) -> Result<(), String> {
    let a: Vec<&str> = args.iter().map(String::as_str).collect();
    match a.as_slice() {
        ["table", cb, out] => {
            let cb = Codebook::from_name(cb).ok_or_else(|| format!("unknown codebook {cb}"))?;
            write_u16(out, &[65536], &codebook(cb)).map_err(|e| e.to_string())
        }
        ["groups", dir] => {
            let ck = open(dir)?;
            for (prefix, g) in &ck.groups {
                let line = serde_json::json!({
                    "prefix": prefix, "bits": g.bits.as_f64(), "codebook": g.codebook.name(), "k": g.k, "n": g.n,
                    "in_scales": g.in_scales, "out_scales": g.out_scales, "bias": g.bias,
                });
                println!("{line}");
            }
            for (prefix, why) in &ck.bad {
                println!("{}", serde_json::json!({ "prefix": prefix, "refused": why }));
            }
            Ok(())
        }
        ["wq", dir, prefix, out] => {
            let g = load(dir, prefix)?;
            let w = unpack(&g.trellis, g.kt, g.nt, g.group.bits, &codebook(g.group.codebook));
            write_u16(out, &[g.group.k, g.group.n], &w).map_err(|e| e.to_string())
        }
        ["dense", dir, prefix, out] => {
            let g = load(dir, prefix)?;
            let w = decode_weight(&g.trellis, g.kt, g.nt, g.group.bits, &codebook(g.group.codebook), &g.suh, &g.svh);
            write_f64(out, &[g.group.k, g.group.n], &w).map_err(|e| e.to_string())
        }
        ["forward", dir, prefix, x, out] => {
            let g = load(dir, prefix)?;
            let (shape, xs) = read_f64(x)?;
            if shape.len() != 2 || shape[1] != g.group.k {
                return Err(format!("x must be [rows, {}], got {shape:?}", g.group.k));
            }
            let table = codebook(g.group.codebook);
            let y = forward(&xs, shape[0], &g.trellis, g.kt, g.nt, g.group.bits, &table, &g.suh, &g.svh, g.bias.as_deref());
            write_f64(out, &[shape[0], g.group.n], &y).map_err(|e| e.to_string())
        }
        _ => Err("usage: exl3dq table <cb> <out.npy> | groups <dir> | wq|dense <dir> <prefix> <out.npy> | forward <dir> <prefix> <x.npy> <out.npy>".into()),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("exl3dq: {e}");
            ExitCode::FAILURE
        }
    }
}
