use std::path::Path;
use std::env;

use audio::fileio::*;
use audio::*;
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
    let wav_in = Path::new(args.get(1).expect("input WAV is missing"));
    let wav_out = Path::new(args.get(2).expect("output WAV is missing"));
    check_wav(wav_in).expect("input WAV is invalid");

    // build Transform
    let mut tf = transforms::Phaser::new(8, 0.5, 0.5);

    // WAV I/O
    let mut reader: WavReaderAdapter<_,i32> = WavReaderAdapter::from_path(wav_in)
        .expect("couldn't read input wav");
    eprintln!("Loaded {}", wav_in.display());
    let wavspec = reader.reader.spec();
    let mut writer: WavWriterAdapter<_,i32> = WavWriterAdapter::from_path(wav_out, wavspec)
        .expect("couldn't open output wav");

    let sampler = reader.iter_chunk(BUFFER_CAP);
    // let sampler = reader.iter_overlap(BUFFER_CAP, BUFFER_CAP/2);

    // jump to lib.rs
    read_transform_write(sampler, &mut writer, &mut tf)
        .expect("wav transformation failed");
    // let title = wav_in.file_stem().unwrap().to_str().unwrap().into();
    // let outpath = wav_in.with_extension("png");
    // eprint!("Writing {} ...", outpath.display());
    // spectrogram(sampler, outpath, title)
    //     .expect("spectrogram creation failed");
    eprintln!("Done.");
}
