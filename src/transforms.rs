// std lib imports
use std::{f64::consts::*};
use std::ops::Rem;
// external crates
use rustfft::{FftPlanner, num_complex::Complex};
// local crates
use crate::buffers::SampleBuffer;

type Float = f64;
type Int = i16; // default sample data type

// Transform an audio buffer
// filter, pan, gain, whatever
pub trait Transform {
    fn transform(&mut self, buf: &mut SampleBuffer<Int>);
}

/*
Amp: Increase or decrease gain, that's it.
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

    // create a triangular widow filter
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

    // clear cached last chunk
    pub fn clear_memory(&mut self) {
        self.lastchunk = vec![];
    }

}

impl Transform for Conv1d {
    // apply (causal) filter to buffer,
    // this will delay signal by k-1 samples
    fn transform(&mut self, buf: &mut SampleBuffer<Int>) {
        /* note: multi-channel vectors are interleaved:
         * e.g. [left, right, left, right, ...]
         */
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
