use std::path::Path;
use std::env;

use audio::fileio::*;
use audio::*;
use audio::buffers::BUFFER_CAP;
    
const HELP: &str = "usage: spectrogram [input wav] [output image]";

fn main() {
    // handle input args
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("{}", HELP);
        return
    }
    let wav_in = Path::new(args.get(1).expect("input WAV is missing"));
    check_wav(wav_in).expect("input WAV is invalid");

    // WAV I/O
    let mut reader: WavReaderAdapter<_,i32> = WavReaderAdapter::from_path(wav_in)
        .expect("couldn't read input wav");
    eprintln!("Loaded {}", wav_in.display());
    // let wavspec = reader.reader.spec();

    let sampler = reader.iter_chunk(BUFFER_CAP);

    // jump to lib.rs
    let title = wav_in.file_stem().unwrap().to_str().unwrap().into();
    let outpath = Path::new(args.get(2).expect("output image file is missing"));
    spectrogram(sampler, outpath, title)
        .expect("spectrogram creation failed");
    eprintln!("Wrote spectrogram to {} ...", outpath.display());
}
