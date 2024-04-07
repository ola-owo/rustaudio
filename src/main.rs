mod transforms;
mod buffers;

use std::path::Path;
use std::env;
use std::fs::File;
use hound::{WavReader,WavWriter,read_wave_header};

use buffers::ChunkedSampler;
use transforms::*;

use crate::buffers::SampleConverter;

const BUFFER_CAP: usize = 2048;
const HELP: &str = "usage: audio [input wav] [output wav]";

type Float: = f32; // type used for internal processing
type Int = i16; // type of samples in wav file

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
    let mut wav_samples = wav_reader.samples::<Int>();
    let samp_converter: SampleConverter<_, Float> = SampleConverter::new(&wav_samples);
    let mut sample_buffer = ChunkedSampler::new(
        &mut wav_samples,
        BUFFER_CAP,
        &wavspec
    );

    // initialize wav writer
    let outpath = Path::new(wav_outfile);
    let mut writer = WavWriter::create(outpath, wavspec)
        .expect("couldn't create WAV writer");

    // read & write 1st buffer
    let buf = sample_buffer.next().expect("couldn't load buffer");
    let chunk1 = buf.data();
    let rms = chunk1.iter()
        .fold(0.0_f64, |acc, &x| acc + (x*x) as f64);
    let rms = (rms as f64 / chunk1.len() as f64).sqrt();
    println!("BUFFER INFO:");
    println!("> buffer size: {}", chunk1.len());
    println!("> rms = {}", rms);
    // println!("> min = {}", chunk1.iter().min().unwrap());
    // println!("> max = {}", chunk1.iter().max().unwrap());
    println!("> min = {}", chunk1.iter().fold(Float::INFINITY, |a, &b| a.min(b)));
    println!("> max = {}", chunk1.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b)));

    // write_buffer(&mut writer, &buf);
    // sample_buffer.write_buffer(&mut writer, &buf);
    samp_converter.write_buffer(&mut writer, buf.data());

    // let fs = wavspec.sample_rate as f64;
    let mut tf = Phaser::new(8, 0.0, 0.9, 0.9);
    for mut buf in sample_buffer {
        tf.transform(&mut buf);
        samp_converter.write_buffer(&mut writer, buf.data());
    }

    writer.finalize().expect("Couldn't finalize output file");
}
