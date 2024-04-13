mod transforms;
mod buffers;
mod spectral;
mod plot;

use std::path::Path;
use std::env;
use std::fs::File;

use hound::{WavReader,read_wave_header};
use ndarray::{s, Axis};
use rustfft::num_complex::Complex;

use buffers::{OverlapSampler, SampleBuffer, SampleConverter};
use spectral::STFT;

const BUFFER_CAP: usize = 2048;
const HELP: &str = "usage: audio [input wav]";

type Float: = f32; // type used for internal processing
type Int = i16; // type of samples in wav file
type CFloat = Complex<Float>;

fn main() {
    // handle input args
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("{}", HELP);
        return
    }
    let wav_file = args.get(1).unwrap();
    // let wav_outfile = args.get(2).unwrap();

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
    // take only 10s of audio
    let n_chunks = (30.0 * wavspec.sample_rate as f64 / BUFFER_CAP as f64) as usize;
    let sample_buffer: OverlapSampler<'_, _, _, Float> = OverlapSampler::new(
        &mut wav_samples,
        BUFFER_CAP,
        BUFFER_CAP / 2,
        &wavspec
    );

    // initialize wav writer
    // let outpath = Path::new(wav_outfile);
    // let mut writer = WavWriter::create(outpath, wavspec)
    //     .expect("couldn't create WAV writer");

    // build stft array
    let mut stft = STFT::new(&sample_buffer);
    let stft_data_raw = stft.build(sample_buffer);

    /* fix stft array:
     *   remove negative freq components
     *   get abs of each fft value
     *   normalize each channel so that max=1
     */
    let npt = stft_data_raw.shape()[2];
    let mut stft_data = stft_data_raw.slice_move(s![.., .., ..npt/2])
        .mapv(|z| z.norm());
    for mut stft_data_ch in stft_data.axis_iter_mut(Axis(0)) {
        let max = stft_data_ch.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b)).max(1.0);
        stft_data_ch.map_inplace(|x| {*x = *x / max});
    }

    // plot spectrogram
    plot::spectrogram2d(
        &Path::new("data").join("stft.png"),
        &stft_data.slice_move(s![0,..,..]),
        wavspec.sample_rate
    ).unwrap();
}

// print summary stats
fn chunk_summary(buf: &SampleBuffer<Float>) {
    let chunk1 = buf.data();
    let rms = chunk1.iter()
        .fold(0.0_f64, |acc, &x| acc + (x*x) as f64);
    let rms = (rms as f64 / chunk1.len() as f64).sqrt();
    println!("BUFFER INFO:");
    println!("> buffer size: {}", chunk1.len());
    println!("> rms = {}", rms);
    println!("> min = {}", chunk1.iter().fold(Float::INFINITY, |a, &b| a.min(b)));
    println!("> max = {}", chunk1.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b)));
}
