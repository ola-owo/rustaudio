use std::path::Path;
use std::env;
use std::fs::File;

use hound::{WavReader,WavWriter,read_wave_header};

use audio::*;

// use crate::buffers::*;
// use crate::utils::{Int, Float};
// use crate::transforms::*;

const BUFFER_CAP: usize = 4096;
const HELP: &str = "usage: audio [input wav]";

type Int = audio::utils::Int;
type Float = audio::utils::Float;

fn main() {
    // handle input args
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("{}", HELP);
        return
    }
    let wav_in = Path::new(args.get(1).unwrap());
    let wav_out = Path::new(args.get(2).unwrap());

    // check whether input file is really a wav
    check_wav(wav_in).expect("input file is not a WAV");

    // initialize reader
    let mut wav_reader =  WavReader::open(wav_in)
        .expect("couldn't open file");
    let wavspec_in = wav_reader.spec();
    dbg!(wavspec_in);
    let wav = buffers::WavIO::<Int,Float>::new();
    let sampler = wav.iter(&mut wav_reader, BUFFER_CAP);
    // let sampler = wav.iter_overlap(&mut wav_reader, BUFFER_CAP, BUFFER_CAP/2);

    // initialize wav writer
    let mut wavspec_out = wavspec_in.clone();
    wavspec_out.sample_rate *= 2;
    dbg!(wavspec_out);
    let writer = WavWriter::create(wav_out, wavspec_out)
        .expect("couldn't create WAV writer");

    // let mut tf = Resampler::new(24000);
    let tf = transforms::Decimator::new(2);
    
    // jump to lib.rs
    read_tf_write(wav, sampler, tf, writer);
}

fn check_wav<P: AsRef<Path>>(path: P) -> Result<(),hound::Error> {
    let mut f = File::open(path)?;
    read_wave_header(&mut f)?;
    Ok(())
}
