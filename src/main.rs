use std::path::Path;
use std::env;

use audio::fileio::*;
use audio::read_transform_write;
use audio::transforms;
use audio::buffers::BUFFER_CAP;

const HELP: &str = "usage: audio [input wav]";

fn main() {
    // handle input args
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("{}", HELP);
        return
    }
    let wav_in = Path::new(args.get(1).unwrap());
    let wav_out = Path::new(args.get(2).unwrap());
    check_wav(wav_in).expect("input WAV is invalid");

    // build Transform
    let mut tf = transforms::Phaser::new(8, 0.5, 0.5);

    // WAV I/O
    let mut reader: WavReaderAdapter<_,i32,f32> = WavReaderAdapter::from_path(wav_in)
        .expect("couldn't read input wav");
    let wavspec = reader.reader.spec();
    let mut writer: WavWriterAdapter<_,i32,f32> = WavWriterAdapter::from_path(wav_out, wavspec)
        .expect("couldn't open output wav");

    // jump to lib.rs
    read_transform_write(reader.iter_chunk(BUFFER_CAP), &mut writer, &mut tf)
        .expect("wav transformation failed");
}
