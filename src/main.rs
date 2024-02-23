mod transforms;
mod buffers;

use std::path::Path;
use hound::{WavReader,WavWriter};
use buffers::{ChunkedSampler,write_buffer};

use transforms::Transform;

const WAVFILE: &str = "./data/bubbly2.wav";
const WAV_OUTPUT: &str = "./data/bubbly2-new.wav";
const BUFFER_CAP: usize = 2048;

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
        .fold(0, |acc, &x| acc + (x as i64).pow(2) as u64);
    let rms = (rms as f64 / chunk1.len() as f64).sqrt();
    println!("BUFFER INFO:");
    println!("> buffer size: {}", chunk1.len());
    println!("> rms = {}", rms);
    println!("> min = {}", chunk1.iter().min().unwrap());
    println!("> max = {}", chunk1.iter().max().unwrap());

    write_buffer(&mut writer, chunk1);

    // basic passthrough transform (y[n] = x[n])
    // let mut tf = transforms::DiffEq::new(vec![1.0], vec![1.0], buf.channels());
    let fs = wavspec.sample_rate as f64;
    let mut tf = transforms::Chain::new(transforms::Amp::db(10.0))
        .push(transforms::DiffEq::butterworth2(500.0, fs, wavspec.channels))
        .push(transforms::Chain::new(transforms::Amp::db(20.0)));
    // read & write remaining buffers
    for mut buf in sample_buffer {
        tf.transform(&mut buf);
        write_buffer(&mut writer, buf.data());
    }
}