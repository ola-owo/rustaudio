// std lib imports
use std::f64::consts::*;
use std::collections::VecDeque;
use std::iter::zip;
// external crates
use rustfft::{FftPlanner, num_complex::Complex};
use num_traits::{Num, AsPrimitive};
use num_traits::identities::zero;
// local crates
use crate::buffers::{SampleBuffer, ChannelCount};

type Float = f64;
type Int = i16; // default sample data type
type CFloat = Complex<Float>;

/*
Transform an audio buffer
filter, pan, gain, whatever
*/
pub trait Transform {
    // fn transform(&mut self, buf: &mut SampleBuffer<Int>);
    fn transform(&mut self, buf: &mut SampleBuffer<Int>);
    fn reset(&mut self);
}

// Dummy transform that does nothing
pub struct PassThrough;

impl Transform for PassThrough {
    fn transform(&mut self, _buf: &mut SampleBuffer<Int>) {}
    fn reset(&mut self) {}
}

// Chain multiple transforms together
pub struct Chain {
    chain: Vec<Box<dyn Transform>>,
    length: usize
}

impl Chain {
    pub fn new() -> Self {
        Self {chain: vec![], length: 0}
    }

    pub fn from(tf: impl Transform + 'static) -> Self {
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

#[macro_export]
macro_rules! chain {
    ( $tf0:expr, $( $tf:expr ),* ) => {
        transforms::Chain::from($tf0)
        $(
            .push($tf)
        )*
    };
}

impl Transform for Chain {
    fn transform(&mut self, buf: &mut SampleBuffer<Int>) {
        for tf_box in self.chain.iter_mut() {
            tf_box.transform(buf);
        }
    }

    fn reset(&mut self) {
        for tf_box in self.chain.iter_mut() {
            tf_box.reset();
        }
    }
}

/*
 * Chain transforms together in parallel
 * 
 * all transforms are weighted equally (for now)
 */
pub struct ParallelChain {
    chain: Vec<Box<dyn Transform>>,
    length: usize
}

impl ParallelChain {
    pub fn new() -> Self {
        Self {chain: vec![], length: 0}
    }

    pub fn from(tf: impl Transform + 'static) -> Self {
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

#[macro_export]
macro_rules! parallel_chain {
    ( $tf0:expr, $( $tf:expr ),* ) => {
        transforms::ParallelChain::from($tf0)
        $(
            .push($tf)
        )*
    };
}

impl Transform for ParallelChain {
    /*
    newbuf = (all zeros)
    for each tf:
        copy buf (from original)
        transform buf
        scale buf
        add to newbuf
    reassign buf to newbuf
    */
    fn transform(&mut self, buf: &mut SampleBuffer<Int>) {
        // data_final = final data vector (as float)
        let weight = (self.length as Float).recip();
        let mut data_final = vec![0.0; buf.len()];

        // run each transform, scale result, and add to data_final
        let mut buf_copy: SampleBuffer<Int>;
        for tf_box in self.chain.iter_mut() {
            buf_copy = buf.clone();
            tf_box.transform(&mut buf_copy);
            let buf_final_iter = data_final.iter_mut();
            let buf_part_iter = buf_copy.data().iter();
            for (s_final, s_part) in std::iter::zip(buf_final_iter, buf_part_iter) {
                *s_final = *s_final + *s_part as Float * weight;
            }
        }

        // convert output data to Int
        let mut data_final_int = Vec::with_capacity(buf.len());
        for s in data_final.iter() {
            data_final_int.push(*s as Int);
        }

        // assign new data to buf
        *buf.data_mut() = data_final_int;
    }

    fn reset(&mut self) {
        for tf_box in self.chain.iter_mut() {
            tf_box.reset();
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
    pub fn db(db: Float) -> Self {
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
        10.0_f64.powf(db / 20.0)
    }
}

impl Transform for Amp {
    fn transform(&mut self, buf: &mut SampleBuffer<Int>) {
        let data = buf.data_mut();
        *data = data.iter_mut()
            .map(|x| ((*x as Float) * self.gain) as Int)
            .collect::<Vec<Int>>();
    }

    fn reset(&mut self) {}
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
        // poles are ordered CCW and are symmetric across negative real axis
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
                    PI * ((i + ord as i32) as Float / ordf)
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

        // get upper left quadrant (not including p=-1)
        let (poles_upper, _poles_lower) = poles.split_at(ord as usize / 2);

        // bilinear transform poles from s to z space
        let wwarp = (wc * ts * 0.5).tan();

        // freq response function
        // input (1/z) = exp(-j*w)
        // output = H(1/z)
        let freqfn = |z: Complex<Float>| {
            let one_plus_z = 1.0 + z;
            let one_minus_z = 1.0 - z;
            // polefn(p) combines p and p* components of H(z)
            let polefn = |p: &Complex<Float>| {
                let p_sum = 2.0 * p.re * p.arg().cos(); // p + p* = 2 r cos(w)
                let p_prod = p.re.powi(2); // p* p = r^2
                let numer = wwarp.powi(2) * (1.0 + 2.0*z + z.powu(2));
                let denom = one_minus_z.powi(2)
                                          - wwarp * p_sum * one_plus_z * one_minus_z
                                          + (wwarp * one_plus_z).powi(2) * p_prod;
                numer / denom
            };
            let h = poles_upper.iter()
                .map(|p| polefn(p))
                .product();
            if odd_ord {
                // pole0 = H(z) component at p = exp(-pi)
                let pole0 = (wwarp * one_plus_z) / (1.0 + wwarp + (wwarp - 1.0) * z);
                h * pole0
            } else {
                h
            }
        };

        // sample the freq response
        let nsamp = OVERSAMPLE_FACTOR * n;
        let nsamp_step = (nsamp as Float).recip() * PI;

        // convert sample indices 0..nsamp to 1/z values,
        // representing frequencies 0 to 2PI
        let i2z = |x: usize| {
            /*
                original range: [0, nsamp)
                a = x * stepsz: [0, 2PI)
                b = exp(-a j) :  [1 .. j .. -1 .. -j .. 0)
            */
            let a = x as Float * nsamp_step;
            Complex::from_polar(1.0, -a)
        };
        let mut zvals: Vec<Complex<Float>> = Vec::with_capacity(2 * nsamp);
        // zvals_1 = H(1/z) from 0 to PI (inclusive, len=nsamp+1)
        let zvals_1: Vec<Complex<Float>> = (0..nsamp+1).map(i2z).collect();
        zvals.extend_from_slice(&zvals_1[..]);
        // zvals_2 = H(1/z) from PI to 2PI (exclusive, len=nsamp-1)
        let zvals_2 = zvals_1[1..nsamp].iter().rev();
        zvals.extend(zvals_2);

        // sample freqfn
        let mut hvals: Vec<CFloat> = zvals.into_iter()
            .map(freqfn)
            .collect();
        let _hvals_abs: Vec<Float> = hvals.iter().map(|&x| x.norm()).collect();
        let _hvals_arg: Vec<Float> = hvals.iter().map(|&x| x.arg()).collect();

        // inverse fft
        let mut fft_plan = FftPlanner::new();
        let ifft = fft_plan.plan_fft_inverse(nsamp as usize);
        ifft.process(&mut hvals);
        let _ihvals_abs: Vec<Float> = hvals.iter().map(|&x| x.norm()).collect();
        let _ihvals_arg: Vec<Float> = hvals.iter().map(|&x| x.arg()).collect();
        let _ihvals_re: Vec<Float> = hvals.iter().map(|&x| x.re).collect();

        // truncate ifft to n points and discard nonreal parts
        // also normalize ifft output: scale by 1/sqrt(len())
        let fft_scalar = (hvals.len() as Float).sqrt().recip();
         let kernel: Vec<Float> = hvals.iter()
            .take(n)
            .map(|&x| x.re * fft_scalar)
            .collect();

        Self {kernel, lastchunk: vec![]}
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
            output.push(res.round() as Int);
        }

        // save last chunk (n*(k-1)) samples
        self.lastchunk = data[data.len()-npad..].iter().map(|&x| x as Float).collect();

        // move buffer pointer to output vector
        *data = output;
    }

    // delete cached data chunk
    fn reset(&mut self) {
        self.lastchunk = vec![];
    }
}

/*
 * (EXPERIMENTAL) Represent systems as difference equations
 * This might be more efficient than Conv1d
 * 
 * BORING MATH:
 *        [a0 + a1 z^{-1} + ... + ak z^{-k}]     Y(z)
 * H(z) = __________________________________  =  ____
 *        [b0 + b1 z^{-1} + ... + bm z^{-m}]     X(z)
 * 
 * <--> a0 x[n] + ... + ak x[n-k] = b0 y[n] + ... + bm y[n-m]
 * <--> y[n] = [(a0 x[n] + ... + ak x[n-k]) - (b1 y[n-1] + ... + bm y[n-m])] / b0
 * 
 * `xcoeff` represents numerator coefficients
 * `ycoeff` represents denominator coefficients
 * `xvals[c]` contains previous x-values from channel c:
 *      back = new values, front = old values
 *      (same thing for yvals)
 */
pub struct DiffEq {
    xcoeff: Vec<Float>,          // x[n] coefficients
    ycoeff: Vec<Float>,          // y[n] coefficients
    xvals: Vec<VecDeque<Float>>, // cached x-values
    yvals: Vec<VecDeque<Float>>, // cached y-values
    numch: ChannelCount          // number of channels
}

/*
 * build a linear Transform from transfer function coefficients
 *
 * `numer` = numerator coefficients (from z^0 to z^{-Inf})
 * `denom` = denominator coefficients
 */
impl DiffEq {
    pub fn new(numer: Vec<Float>, denom: Vec<Float>, numch: ChannelCount) -> Self {
        let channelct = numch as usize;
        let nx = numer.len();
        let ny = denom.len();
        let mut xvals = Vec::with_capacity(channelct);
        let mut yvals = Vec::with_capacity(channelct);
        for _ in 0..numch {
            xvals.push(VecDeque::from(vec![0.0; nx]));
            yvals.push(VecDeque::from(vec![0.0; ny]));
        }

        // reverse order of coefficients (x[n-Inf] to x[n])
        // so that they match x/yvals (oldest to newest)
        let mut xcoeff = numer;
        let mut ycoeff = denom;
        xcoeff.reverse();
        ycoeff.reverse();

        Self {
            xvals,
            yvals,
            xcoeff,
            ycoeff,
            numch
        }
    }

    /*
     * 1st order butterworth
     *
     *        W + W z^{-1}
     * H(z) = ------------------------
     *        (W + 1) + (W - 1) z^{-1}
     */
    pub fn butterworth1(fc: Float, fs: Float, numch: ChannelCount) -> Self {
        let wc = fc * TAU;
        let w = (wc * 0.5 / fs).tan();

        let numer = vec![w, w];
        let denom = vec![w + 1.0, w - 1.0];

        Self::new(numer, denom, numch)
    }

    /*
     * 2nd order butterworth (UNSTABLE AND PROBABLY WRONG)
     *
     *        W^2 + 2 W^2 z^{-1} w^2 z^{-2}
     * H(z) = ------------------------------------------------------------------
     *        1 + W sqrt2 + W^2 + 2(W^2 - 1) z^{-1} + (W^2 - W sqrt2 + 1) z^{-2}
     */
    pub fn butterworth2(fc: Float, fs: Float, numch: ChannelCount) -> Self {
        let wc = fc * TAU;
        let w = (wc * 0.5 / fs).tan();
        let w2 = w.powi(2);
        let wroot2 = w * SQRT_2;

        let numer = vec![w2, 2.0*w2, w2];
        let denom = vec![1.0 + wroot2 + w2, 2.0*(w2 - 1.0), w2 - wroot2 + 1.0];

        Self::new(numer, denom, numch)
    }
}

impl Transform for DiffEq {
    fn reset(&mut self) {
        self.xvals.clear();
        self.yvals.clear();
    }

    fn transform(&mut self, buf: &mut SampleBuffer<Int>) {
        let numch = buf.channels() as usize;
        assert!(self.numch == numch as ChannelCount,
            "channel count doesn't match! (self={}, buffer={})", self.numch, numch
        );
        let data = buf.data_mut();
        let mut data_new: Vec<Int> = vec![0; data.len()];

        let mut x_sum: Float; // x_sum = a0 x[n] + ... + ak x[n-k]
        let mut y_sum: Float; // y_sum = b1 y[n-1] + ... + bm y[n-m]
        let mut y_new: Float; // current y[n] value
        for ch in 0..numch {
            let x = self.xvals.get_mut(ch).unwrap();
            let y = self.yvals.get_mut(ch).unwrap();

            let sample_iter = zip(data.iter(), data_new.iter_mut())
                .skip(ch)
                .step_by(numch);
            for (samp, newsamp) in sample_iter {
                // get newest x-value (x[n]) and pop oldest x-value
                x.pop_front();
                x.push_back(*samp as Float);

                // compute sum[ x_0 x[n] ... x_k x[n-k] ]
                x_sum = zip(self.xcoeff.iter(), x.iter())
                    .map(|(&coeff, &xval)| {coeff * xval})
                    .sum();
                // compute sum[ y_1 y[n] ... y_m x[n-m] ]
                y.pop_front();
                y_sum = zip(self.ycoeff.iter(), y.iter())
                    .take(y.len()-1)
                    .map(|(&coeff, &yval)| {coeff * yval})
                    .sum();

                // compute y[n], pop oldest y-value, write to buffer
                y_new = (x_sum - y_sum) / self.ycoeff.get(y.len()-1).unwrap();
                y.push_back(y_new);
                *newsamp = y_new.round() as Int;
            }
        }

        // replace old buffer with new
        *data = data_new;
    }
}

/*
* ToMono: average all signals together
*/
pub struct ToMono;

impl Transform for ToMono {
    fn transform(&mut self, buf: &mut SampleBuffer<Int>) {
        let numch = buf.channels();
        let data = buf.data_mut();
        for chunk in data.chunks_mut(numch as usize) {
            let avg = chunk.iter().sum::<Int>() / numch as Int;
            chunk.fill(avg);
        }
    }

    fn reset(&mut self) {}
}

/*
* Pan: pan signal left or right
*/
pub struct Pan {
    left_wt: [Float; 2],
    right_wt: [Float; 2]
}

impl Pan {
    /*
     * Pan audio to the left.
     * pan_amt=1 --> pan all the way left
     * pan_amt=0 --> pan all the way right
     */
    pub fn pan_left(pan_amt: Float) -> Self {
        assert!(pan_amt >= 0.0 && pan_amt <= 1.0, "pan amount must be between 0-1");
        Self {
            left_wt: [1.0, pan_amt],
            right_wt: [0.0, 1.0 - pan_amt]
        }
    }

    /*
     * Same thing as pan_left, but pans in the other direction
     */
    pub fn pan_right(pan_amt: Float) -> Self {
        Self::pan_left(1.0 - pan_amt)
    }

    /*
     * Swap left and right channels
     */
    pub fn invert() -> Self {
        Self {
            left_wt: [0.0, 1.0],
            right_wt: [1.0, 0.0]
        }
    }
}

impl Transform for Pan {
    fn transform(&mut self, buf: &mut SampleBuffer<Int>) {
        let numch = buf.channels();
        assert!(numch == 2, "Sample buffer must be stereo");
        for chunk in buf.data_mut().chunks_exact_mut(numch as usize) {
            let samp_l: Int = zip(&self.left_wt, chunk.iter())
                .map(|(&x, &y)| {(x * y as Float).round() as Int})
                .sum();
            let samp_r: Int = zip(&self.right_wt, chunk.iter())
                .map(|(&x, &y)| {(x * y as Float).round() as Int})
                .sum();
            chunk[0] = samp_l;
            chunk[1] = samp_r;
        }
    }

    fn reset(&mut self) {}
}

#[allow(dead_code)]
fn energy<T:Num+Copy>(vec: &Vec<T>) -> T {
    vec
        .iter()
        .fold(zero::<T>(), |acc, &x| acc + x*x)
}

#[allow(dead_code)]
// bound on T means that T is castable to R
// bound on R means that R is a Float
fn rms<T:Num+AsPrimitive<R>, R:'static+num_traits::Float>(vec: &Vec<T>) -> R {
    let e = energy(vec).as_();
    e.sqrt()
}