use std::path::Path;
use fundsp::hacker32::U1;
use fundsp::wave::Wave32;
use fundsp::filter::*;

const WAVFILE: &str = "./data/maggi.wav";

#[allow(dead_code)]
fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn main() {
    const HIGHCUT: f32 = 1000.0;
    const LOWCUT: f32 = 1000.0;
    let path = Path::new(&WAVFILE);
    let wave = Wave32::load(path).expect("couldn't load file");
    let fs = wave.sample_rate();
    let nch = wave.channels();
    println!("{} channels @ {} Hz", nch, fs);

    let highpass: Highpole<f32,f32,U1> = Highpole::new(fs, HIGHCUT);
    let bq_coef: BiquadCoefs<f32> = BiquadCoefs::butter_lowpass(fs as f32, LOWCUT);
    let bq: Biquad<f32,f32> = Biquad::with_coefs(bq_coef);
    println!("{:?}", bq.coefs());
    // println!("Node IDs: {} | {}", highpass.ID, bq.ID);
}
