mod transforms;
mod buffers;

use std::path::Path;
use hound::{WavReader,WavWriter};
use buffers::{ChunkedSampler,write_buffer};
use transforms::Transform;

const WAVFILE: &str = "./data/maggi.wav";
const WAV_OUTPUT: &str = "./data/maggi-new.wav";
const BUFFER_CAP: usize = 1024;

fn main() {
    // initialize reader
    let path = Path::new(WAVFILE);
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
    let outpath = Path::new(&WAV_OUTPUT);
    let mut writer = WavWriter::create(outpath, wavspec)
        .expect("couldn't create WAV writer");

    // read & write 1st buffer
    let buf = sample_buffer.next().expect("couldn't load buffer");
    let chunk1 = buf.data();
    let rms = chunk1.iter()
        .fold(0, |acc, &x| acc + x.pow(2) as u64);
    let rms = (rms as f64 / chunk1.len() as f64).sqrt();
    println!("BUFFER INFO:");
    println!("> buffer size: {}", chunk1.len());
    println!("> rms = {}", rms);
    println!("> min = {}", chunk1.iter().min().unwrap());
    println!("> max = {}", chunk1.iter().max().unwrap());

    write_buffer(&mut writer, chunk1);

    // read & write remaining buffers
    let fs = wavspec.sample_rate as f64;
    let fc: f64 = 50.0; // cutoff freq in hz
    let mut tf = transforms::Conv1d::butterworth(256, fc, 3, fs);
    // let mut tf = transforms::Conv1d::sinc(60, wc);
    // let mut tf = transforms::Chain::new(tf)
    //     .push(transforms::Amp::from_db(3.0));
    for mut buf in sample_buffer {
        tf.transform(&mut buf);
        write_buffer(&mut writer, buf.data());
    }
    // lowpass.clear_memory();
}