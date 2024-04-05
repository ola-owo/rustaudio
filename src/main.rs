mod transforms;
mod buffers;

use std::path::Path;
use std::env;
use std::fs::File;
use hound::{WavReader,WavWriter,read_wave_header};
use buffers::{ChunkedSampler,write_buffer};

use transforms::*;

const BUFFER_CAP: usize = 2048;

const HELP: &str = "usage: audio [input wav] [output wav]";

fn main() {
    // handle input args
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("{}", HELP);
        return
    }
    let wav_file = args.get(1).unwrap();
    let wav_outfile = args.get(2).unwrap();

    // check whether input file is really a wav
    let mut f = File::open(wav_file)
        .expect("couldn't open WAV file");
    read_wave_header(&mut f)
        .expect("Input is not a WAV file");

    // initialize reader
    let path = Path::new(wav_file);
    let mut wav_reader =  WavReader::open(path)
        .expect("couldn't open file");
    let wavspec = wav_reader.spec();
    dbg!(&wavspec);
    let mut sample_iter = wav_reader.samples::<i16>();
    let mut sample_buffer = ChunkedSampler::new(
        &mut sample_iter,
        BUFFER_CAP,
        wavspec.channels,
        wavspec.sample_rate
    );

    // initialize wav writer
    let outpath = Path::new(wav_outfile);
    let mut writer = WavWriter::create(outpath, wavspec)
        .expect("couldn't create WAV writer");

    // read & write 1st buffer
    let buf = sample_buffer.next().expect("couldn't load buffer");
    let chunk1 = buf.data();
    let rms = chunk1.iter()
        .fold(0, |acc, &x| acc + (x as i64).pow(2) as u64);
    let rms = (rms as f64 / chunk1.len() as f64).sqrt();
    println!("BUFFER INFO:");
    println!("> buffer size: {}", chunk1.len());
    println!("> rms = {}", rms);
    println!("> min = {}", chunk1.iter().min().unwrap());
    println!("> max = {}", chunk1.iter().max().unwrap());

    write_buffer(&mut writer, chunk1);

    // let fs = wavspec.sample_rate as f64;
    let mut tf = Phaser::new(2, 0.0, 0.9, 0.5);
    for mut buf in sample_buffer {
        tf.transform(&mut buf);
        write_buffer(&mut writer, buf.data());
    }

    writer.finalize().expect("Couldn't finalize output file");
}
