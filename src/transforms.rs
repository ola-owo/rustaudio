//use itertools::{Interleave,interleave};
use std::f64::consts::*;
use crate::buffers::SampleBuffer;

type Float = f64;
type Int = i16; // default sample data type

// Transform an audio buffer
// filter, pan, gain, whatever
pub trait Transform {
    fn transform(&self, buf: &mut SampleBuffer<Int>);
}

pub struct Amp {
    // here, gain is a multiplier
    // dB change = 20 * log10(gain)
    gain: Float
}

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
    fn transform(&self, buf: &mut SampleBuffer<Int>) {
        let data = buf.data_mut();
        *data = data.iter_mut()
            .map(|x| ((*x as Float) * self.gain) as Int)
            .collect::<Vec<Int>>();
    }
}

pub struct Conv1d {
    kernel: Vec<Float>
}

impl Conv1d {
    pub fn new(kernel: Vec<Float>) -> Self {
        Self {kernel}
    }

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
        Self {kernel}
    }

    pub fn rect_win(n: usize) -> Self {
        let rectheight = 1.0 / (n as Float);
        let kernel = vec![rectheight; n];
        Self {kernel}
    }

    pub fn sinc_lowpass(n: usize, wc: Float) -> Self {
        let sinc = |i: i32| match i {
            0 => wc * FRAC_1_PI,
            _ => (wc * (i as Float)).sin() * FRAC_1_PI / (i as Float)
        };
        let ib = (n as i32)/2 + 1;
        let ia = ib - (n as i32);
        let kernel = (ia..ib)
            .map(sinc)
            .collect::<Vec<Float>>();
        Self {kernel}
    }

}

impl Transform for Conv1d {
    fn transform(&self, buf: &mut SampleBuffer<Int>) {
        /* note: vector is interleaved:
         * odd samples are left, even samples are right
         */
        let numch = buf.channels();
        let data = buf.data_mut();
        let k = self.kernel.len();
        let n = data.len();
        let npad = k / 2 * numch as usize; // amt of zero-padding on either end
        // let mut kernel_padded: Vec<_> = self.kernel
        //     .iter()
        //     .flat_map(|x| vec![*x,0.0])
        //     .collect();
        // kernel_padded.pop(); // remove last 0
        
        // make padded array (as float)
        let mut buf_padded = Vec::<Float>::with_capacity(2*npad + data.len());
        for _ in 0..npad {
            buf_padded.push(0.0);
        }
        buf_padded.extend(&mut data.iter_mut().map(|x| *x as Float));
        for _ in 0..npad {
            buf_padded.push(0.0);
        }

        // make output array
        let mut output = Vec::<Int>::with_capacity(data.len());

        // convolve buf_padded with kernel
        // output[i] = conv(kernel, bufpart)
        //  where bufpart = buf_padded[i : i+k*c] -> step_by(2)
        for ia in 0..n {
            let ib = ia + k * (numch as usize) - 1;
            let bufpart = buf_padded[ia..ib].iter().step_by(numch as usize);
            let res: Float = self.kernel.iter().zip(bufpart).map(|(k,x)| k*x).sum();
            output.push(res as Int);
        }

        // move buffer pointer to output vector
        *data = output;
    }
}
