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

use buffers::{BufferSampler, OverlapSampler, SampleBuffer, SampleConverter};
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
    let wav_in_path = Path::new(wav_file);
    let mut wav_reader =  WavReader::open(wav_in_path)
        .expect("couldn't open file");
    let wavspec = wav_reader.spec();
    dbg!(&wavspec);
    let mut wav_samples = wav_reader.samples::<Int>();
    let samp_converter: SampleConverter<_, Float> = SampleConverter::new(&wav_samples);
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
    // also save time and freq step-sizes for later
    let fs = wavspec.sample_rate as Float;
    let tstep = sample_buffer.step_size() as Float / fs;
    let fstep = fs as Float / sample_buffer.buffer_size() as Float;
    let mut stft = STFT::new(&sample_buffer);
    let stft_data_raw = stft.build(sample_buffer);

    /* fix stft array:
    *   remove negative freq components
    *   get abs of each fft value
    *   normalize each channel so that max=1
    */
    let npt = stft_data_raw.shape()[2];
    let mut stft_data = stft_data_raw.slice_move(s![0, .., ..npt/2])
    .mapv(|z| z.norm().log10());
    for mut stft_data_ch in stft_data.axis_iter_mut(Axis(0)) {
        let max = stft_data_ch.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
        if max > 0.0 { // avoid dividing by 0
            stft_data_ch.map_inplace(|x| {*x /= max});
        }
    }

    // get time and freq arrays
    let (nts, npts) = stft_data.dim();
    let tvals: Vec<_> = (0..nts)
        .map(|x| x as Float * tstep)
        .collect();
    let fvals = (0..npts).map(|x| x as Float * fstep).collect();

    // plot spectrogram
    let out_path = wav_in_path.with_extension("png");
    plot::spectrogram_log(
        &out_path,
        &stft_data,
        fvals,
        tvals,
        format!("Spectrogram of {}", wav_in_path.file_name().unwrap().to_str().unwrap())
    ).unwrap();
    println!("{} saved.", out_path.display());
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
