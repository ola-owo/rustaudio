mod transforms;
mod buffers;

use std::path::Path;
use std::f64::consts::TAU;
use hound::{WavReader,WavWriter};
use buffers::{ChunkedSampler,write_buffer};
use transforms::Transform;

// use crate::transforms::Transform;

const WAVFILE: &str = "./data/maggi.wav";
const WAV_OUTPUT: &str = "./data/maggi_new.wav";
const BUFFER_CAP: usize = 512;

fn main() {
    // initialize reader
    let path = Path::new(WAVFILE);
    let mut wav_reader =  WavReader::open(path)
        .expect("couldn't open file");
    let wavspec = wav_reader.spec();
    let sample_format = match wavspec.sample_format {
        hound::SampleFormat::Float => "float",
        hound::SampleFormat::Int => "int"
    };
    println!("WAV spec:\n> channels: {}\n> fs: {}\n> bit depth: {}\n> dtype: {}", 
        wavspec.channels, wavspec.sample_rate, wavspec.bits_per_sample, sample_format);
    let mut sample_iter = wav_reader.samples::<i16>();
    let mut sample_buffer = ChunkedSampler::new(
        &mut sample_iter,
        BUFFER_CAP,
        wavspec.channels,
        wavspec.sample_rate
    );

    // initialize wav writer
    let outpath = Path::new(&WAV_OUTPUT);
    let mut writer = WavWriter::create(outpath, wavspec)
        .expect("couldn't create WAV writer");

    // read & write 1st buffer
    let buf = sample_buffer.next().expect("couldn't load buffer");
    let data = buf.data();
    let rms = data.iter()
        .fold(0.0, |acc,x| acc + (*x as f64).powi(2))
        .sqrt();
    let bufmin = data.iter().min().unwrap();
    let bufmax = data.iter().max().unwrap();
    println!("{} samples loaded", data.len());
    println!("mean = {}, min = {}, max = {}", rms, bufmin, bufmax);

    write_buffer(&mut writer, data);

    // read & write remaining buffers
    // let amp_quieter = transforms::Amp::from_db(-10.0);
    // let trifilt = transforms::Conv1d::new(Vec::from(TRIANGLE_19));
    let fc = 5000.0; // cutoff freq in hz
    let wc = TAU * fc / (wavspec.sample_rate as f64);
    let mut lowpass = transforms::Conv1d::sinc(60, wc);
    for mut buf in sample_buffer {
        // amp_quieter.transform(&mut buf);
        lowpass.transform(&mut buf);
        write_buffer(&mut writer, buf.data());
    }
    lowpass.clear_memory();
}