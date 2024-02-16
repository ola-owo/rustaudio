mod transforms;
mod buffers;

use std::path::Path;
use hound::{WavReader,WavWriter};
use buffers::{ChunkedSampler,write_buffer};
use transforms::Transform;

const WAVFILE: &str = "./data/maggi.wav";
const WAV_OUTPUT: &str = "./data/maggi-new.wav";
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
    let data = buf.data();
    let rms = data.iter()
        .fold(0, |acc, &x| acc + x.pow(2) as u32);
    let rms = (rms as f64 / data.len() as f64).sqrt();
    println!("BUFFER INFO:");
    println!("> buffer size: {}", data.len());
    println!("> rms = {}", rms);
    println!("> min = {}", data.iter().min().unwrap());
    println!("> max = {}", data.iter().max().unwrap());

    write_buffer(&mut writer, data);

    // read & write remaining buffers
    let fs = wavspec.sample_rate as f64;
    let fc: f64 = 1000.0; // cutoff freq in hz
    let lowpass = transforms::Conv1d::butterworth(200, fc, 1, fs);
    // let lowpass = transforms::Conv1d::sinc(60, wc);
    let amp = transforms::Amp::from_db(3.0);
    let mut chain = transforms::Chain::new(lowpass)
        .push(amp);
    for mut buf in sample_buffer {
        chain.transform(&mut buf);
        write_buffer(&mut writer, buf.data());
    }
    // lowpass.clear_memory();
}