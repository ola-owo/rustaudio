// std lib imports
use std::f64::consts::*;
use std::ops::Rem;
// external crates
use rustfft::{FftPlanner, num_complex::Complex};
use num_traits::{Num, AsPrimitive};
use num_traits::identities::zero;
// local crates
use crate::buffers::SampleBuffer;

type Float = f64;
type Int = i16; // default sample data type
type CFloat = Complex<Float>;

/*
Transform an audio buffer
filter, pan, gain, whatever
*/
pub trait Transform {
    fn transform(&mut self, buf: &mut SampleBuffer<Int>);
}

// Dummy transform that does nothing
pub struct PassThrough;

impl Transform for PassThrough {
    fn transform(&mut self, _buf: &mut SampleBuffer<Int>) {}
}

// Chain multiple transforms together
pub struct Chain {
    chain: Vec<Box<dyn Transform>>,
    length: usize
}

impl Chain {
    pub fn new(tf: impl Transform + 'static) -> Self {
        Self {chain: vec![Box::new(tf)], length: 1}
    }

    // 'static bound requires input [tf] to be an owned type,
    // which all Transform implementations are
    pub fn push(mut self, tf: impl Transform + 'static) -> Self {
        self.chain.push(Box::new(tf));
        self.length += 1;
        self
    }

    pub fn len(&self) -> usize {
        self.length
    }
}

impl Transform for Chain {
    fn transform(&mut self, buf: &mut SampleBuffer<Int>) {
        for tf_box in self.chain.iter_mut() {
            tf_box.transform(buf);
        }
    }
}

/*
Amp: scale signal up or down.
*/
pub struct Amp {
    // here, gain is a multiplier
    // dB change = 20 * log10(gain)
    gain: Float
}

#[allow(dead_code)]
impl Amp {
    pub fn new(gain: Float) -> Self {
        Self {gain}
    }
    pub fn from_db(db: Float) -> Self {
        let gain = Self::db2gain(db);
        Self {gain}
    }
    pub fn setgain(&mut self, gain: Float) {
        self.gain = gain;
    }
    pub fn setdb(&mut self, db: Float) {
        let gain = Self::db2gain(db);
        self.gain = gain;
    }
    fn db2gain(db: Float) -> Float{
        10.0_f64.powf(-db / 20.0)
    }
}

impl Transform for Amp {
    fn transform(&mut self, buf: &mut SampleBuffer<Int>) {
        let data = buf.data_mut();
        *data = data.iter_mut()
            .map(|x| ((*x as Float) * self.gain) as Int)
            .collect::<Vec<Int>>();
    }
}

/*
Conv1d: Convolve signal with a kernel.
Can be used for filtering, reverb, whatever.
*/
pub struct Conv1d {
    kernel: Vec<Float>,
    lastchunk: Vec<Float>
}

#[allow(dead_code)]
impl Conv1d {
    // create a filter using a custom convolution kernel
    pub fn new(kernel: Vec<Float>) -> Self {
        Self {kernel, lastchunk: vec![]}
    }

    // create a triangular window filter
    pub fn triangle_win(n: usize) -> Self {
        let mut kernel = Vec::<Float>::with_capacity(n);
        let nhalf = n / 2;
        for i in 1..=nhalf {
            kernel.push(i as Float);
        }
        if n % 2 == 0 {
            // 1 up to n/2, n/2 down to 1
            for i in (1..=nhalf).rev() {
                kernel.push(i as Float);
            }
        } else {
            // 1 up to n/2, 1+n/2, down to 1
            for i in (1..=1+nhalf).rev() {
                kernel.push(i as Float);
            }
        }
        Self {kernel, lastchunk: vec![]}
    }

    // create a rectangular window filter
    pub fn rect_win(n: usize) -> Self {
        let rectheight = 1.0 / (n as Float);
        let kernel = vec![rectheight; n];
        Self {kernel, lastchunk: vec![]}
    }

    // create a sinc-window low pass filter
    pub fn sinc(n: usize, wc: Float) -> Self {
        let sinc = |i: i32| match i {
            0 => wc * FRAC_1_PI,
            _ => (wc * (i as Float)).sin() * FRAC_1_PI / (i as Float)
        };
        let ib = (n as i32)/2 + 1;
        let ia = ib - (n as i32);
        let kernel = (ia..ib)
            .map(sinc)
            .collect::<Vec<Float>>();
        Self {kernel, lastchunk: vec![]}
    }

    /*
    Create digital butterworth filter,
    using a bilinear transform,
        W_cont = 2 * fs * tan(w_disc / 2)
    then sampling the freq response,
    then an IFFT.

    n   digital filter kernel length
    fc  cutoff freq (hz)
    ord analog filter order
    fs  sample rate (hz)
    */
    pub fn butterworth(n: usize, fc: Float, ord: u32, fs: Float) -> Self {
        const OVERSAMPLE_FACTOR: usize = 15;
        let wc = TAU * fc; // cutoff freq in rad/s
        let ts = fs.recip(); // sample period in s

        // Compute butter poles
        // pole magnitude is actually wc, but we're normalizing to unit length
        let ordf = ord as Float;
        let odd_ord = ord % 2 == 1;
        let poles = match odd_ord {
            // poles are ordered ccw from j to -j axis
            // odd order: include -1
            true => {
                let i0 = (ord as i32 - 1) / 2;
                (-i0 ..= i0)
                .map(|i| Complex::from_polar(
                    1.0,
                    PI * ((ord as i32 + i) as Float / ordf)
                ))
                .collect::<Vec<CFloat>>()
            },
            false => {
                let i0 = (ord as i32) / 2;
                (-i0 ..= i0-1)
                .map(|i| Complex::from_polar(
                    1.0,
                    PI * (1.0 + (ordf.recip() * (0.5 + i as Float)))
                ))
                .collect::<Vec<CFloat>>()
            }
        };
        for (i, pole) in poles.iter().enumerate() {
            println!("POLE {}: {:?}", i, pole);
        }

        // bilinear transform poles from s to z space
        // let kwarp = wc / (0.5 * wc / fs).tan(); // freq warp centered at wc
        let kwarp = (wc * ts * 0.5).tan().recip();
        let poles_bilin = poles.iter()
            .map(|&p| (kwarp + p) / (kwarp - p))
            .collect::<Vec<CFloat>>();
        for (i, pole) in poles_bilin.iter().enumerate() {
            println!("POLE {} (prewarped): {:?}", i, pole);
        }

        // get poles in upper left quadrant (Re<0, Im>0) -- EXCLUDING p=-1
        // let poles_upper = Vec::from(&poles[..(ord as usize)/2]);
        // println!("UPPER POLES: {:?}", &poles_upper);

        // freq response function
        let freqfn = |wnorm: Float| {
            // get freq resp component from each pole
            // let f = poles_upper.iter()
            //     .map(|p| Complex::new(1.0 - wnorm.powi(2), -2.0 * wnorm * p.re))
            //     .product::<CFloat>()
            //     .finv();
            // if odd order, include p=-1 component
            // match odd_ord {
            //         false => f,
            //         true => f * Complex::new(1.0, wnorm).finv()
            //     }
            // };

            // don't combine conj poles
            // just compute 1 / prod[ jw - p_i ]
            poles_bilin.iter()
                .map(|p| Complex::new(-p.re, -p.im + wnorm))
                .product::<CFloat>()
                .finv()
        };

        // sample the freq response
        let nsamp = OVERSAMPLE_FACTOR * n * 2;
        let nsamp_step = (nsamp as Float).recip() * TAU;
        let i2w = |x: usize| {
            /*
            original range: [0, nsamp)
            a = x * stepsz: [0, 2PI)
            b = a + PI:     [PI, 3PI)
            c = b % 2PI:    [PI, 2PI) + [0, PI)
            d = c - PI:     [0, PI) + [-PI, 0)

            e = d * fs      [0, fs/2) + [-fs/2, 0)
            */
            ((x as Float * nsamp_step) + PI).rem(TAU) - PI
            // (((x as Float * nsamp_step) + PI).rem(TAU) - PI) * fs
        };

        // wvals: frequencies to sample: [-pi, pi] rad/samp
        let wvals: Vec<Float> = (0..nsamp)
            .map(i2w)
            .collect();

        // sample freqfn
        let mut hvals: Vec<CFloat> = wvals.into_iter()
            .map(freqfn)
            .collect();
        let _hvals_abs: Vec<Float> = hvals.iter().map(|&x| x.norm()).collect();
        let _hvals_arg: Vec<Float> = hvals.iter().map(|&x| x.arg()).collect();

        // inverse fft
        let mut fft_plan = FftPlanner::new();
        let ifft = fft_plan.plan_fft_inverse(nsamp as usize);
        ifft.process(&mut hvals);

        // resample ifft to n points and discard nonreal parts
        // also normalize ifft output: scale by 1/len().sqrt()
        // also ignore latter half bc ifft result is symmetric
        let fft_scalar = (hvals.len() as Float).sqrt().recip();
        let kernel: Vec<Float> = hvals[..n*OVERSAMPLE_FACTOR]
            .iter()
            .step_by(OVERSAMPLE_FACTOR)
            .map(|&x| x.re * fft_scalar)
            .collect();

        Self {kernel, lastchunk: vec![]}
    }

    // clear cached last chunk
    pub fn clear_memory(&mut self) {
        self.lastchunk = vec![];
    }

}

impl Transform for Conv1d {
    // apply (causal) filter to buffer,
    // this will delay signal by k-1 samples
    fn transform(&mut self, buf: &mut SampleBuffer<Int>) {
        // note: multi-channel vectors are interleaved:
        // [left, right, left, right, ...]
        let numch = buf.channels();
        let data = buf.data_mut();
        let k = self.kernel.len();
        let n = data.len();
        let npad = numch as usize * (k-1); // amount of padding - LEFT SIDE ONLY

        // make padded array (as float)
        let mut buf_padded = Vec::<Float>::with_capacity(npad + data.len());
        if self.lastchunk.len() == npad {
            buf_padded.append(&mut self.lastchunk);
        } else {
            buf_padded.append(&mut vec![0.0; npad]);
        }
        buf_padded.extend(&mut data.iter().map(|x| *x as Float));

        // make output array
        let mut output = Vec::<Int>::with_capacity(data.len());

        // convolve buf_padded with kernel
        // output[i] = conv(kernel, bufpart)
        //  where bufpart = buf_padded[i : i+k*c] -> step_by(2)
        for ia in 0..n {
            let ib = ia + (k-1) * (numch as usize) + 1;
            let bufpart = buf_padded[ia..ib]
                .iter()
                .step_by(numch as usize);
            let res: Float = self.kernel
                .iter().rev()
                .zip(bufpart)
                .map(|(k,x)| k*x)
                .sum();
            output.push(res as Int);
        }

        // save last chunk (n*(k-1)) samples
        self.lastchunk = data[..npad].iter().map(|&x| x as Float).collect();

        // move buffer pointer to output vector
        *data = output;
    }
}


/*
ToMono: average all signals together
*/
pub struct ToMono;

impl Transform for ToMono {
    fn transform(&mut self, buf: &mut SampleBuffer<Int>) {
        let numch = buf.channels();
        let data = buf.data_mut();
        *data = data.chunks(numch as usize)
            .map(|x| x.iter().sum())
            .collect();
    }
}

#[allow(dead_code)]
fn energy<T:Num+Copy>(vec: &Vec<T>) -> T {
    vec
        .iter()
        .fold(zero::<T>(), |acc, &x| acc + x*x)
}

#[allow(dead_code)]
fn rms<T:Num+AsPrimitive<R>, R:'static+num_traits::Float>(vec: &Vec<T>) -> R {
    let e = energy(vec).as_();
    e.sqrt()
}