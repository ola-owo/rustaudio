use std::f32::consts::*;
use std::collections::VecDeque;
use std::iter::zip;
use std::ops::Add;
use std::sync::Mutex;

// external crates
use num_traits::real::Real;
use num_traits::Zero;
use rustfft::{FftPlanner, num_complex::Complex};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

// local crates
use crate::buffers::{ChannelCount, SampleBuffer, SampleRate};
use crate::utils::*;

/// This trait describes objects that can transform an audio buffer.
/// e.g. filter, pan, gain

pub trait Transform<S>: Send {
    /// transform an owned SampleBuffer.
    fn transform(&mut self, buf: SampleBuffer<S>) -> SampleBuffer<S>;

    /// transform a SampleBuffer in place.
    // fn transform_inplace(&mut self, buf: &mut SampleBuffer<S>);

    /// transform a borrowed SampleBuffer and return a new buffer.
    // fn transform_borrow(&mut self, buf: &SampleBuffer<S>) -> SampleBuffer<S>;

    /// reset the Transform's internal state
    fn reset(&mut self);
}

/// Dummy transform that does nothing
pub struct PassThrough;

impl<S> Transform<S> for PassThrough {
    fn transform(&mut self, buf: SampleBuffer<S>) -> SampleBuffer<S> { buf }
    fn reset(&mut self) {}
}

/// Chain multiple transforms together
pub struct Chain<S> {
    chain: Vec<Box<dyn Transform<S>>>,
}

impl<S> Chain<S> {
    pub fn new() -> Self {
        Self {chain: vec![]}
    }

    /// convert a Transform into a Chain
    pub fn from(tf: impl Transform<S> + 'static) -> Self {
        Self {chain: vec![Box::new(tf)]}
    }

    /// add a Transform to the Chain
    pub fn push<T>(mut self, tf: T) -> Self
    where T: Transform<S> + 'static {
        self.chain.push(Box::new(tf));
        self
    }

    /// get chain length
    pub fn len(&self) -> usize {
        self.chain.len()
    }

    /// get a reference to the nth chain element
    pub fn get(&self, n: usize) -> Option<&dyn Transform<S>> {
        match self.chain.get(n) {
            Some(tf_box) => Some(tf_box.as_ref()),
            None => None
        }
    }

    /// get a mutable reference to the nth chain element
    pub fn get_mut(&mut self, n: usize) -> Option<&mut dyn Transform<S>> {
        match self.chain.get_mut(n) {
            Some(tf_box) => Some(tf_box.as_mut()),
            None => None
        }
    }

    /// remove the nth chain element (panic if out of bounds)
    pub fn remove(&mut self, n: usize) -> Option<Box<dyn Transform<S>>> {
        if n >= self.chain.len() {
            None
        } else {
            Some(self.chain.remove(n))
        }
    }

    /// Merge Chain `chain` into self.
    pub fn append(&mut self, mut chain: Chain<S>) {
        self.chain.append(&mut chain.chain);
    }
}

#[macro_export]
macro_rules! chain {
    ( $tf0:expr, $( $tf:expr ),* ) => {
        Chain::from($tf0)
        $(
            .push($tf)
        )*
    };
}

/// Merge 2 Chains together into a single Chain
impl<S> Add for Chain<S> {
    type Output = Chain<S>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut chain = self.chain;
        let mut chain2 = rhs.chain;
        chain.append(&mut chain2);
        Self { chain }
    }
}

impl<S> Transform<S> for Chain<S> {
    fn transform(&mut self, buf: SampleBuffer<S>) -> SampleBuffer<S> {
        let mut buf = buf;
        for tf_box in self.chain.iter_mut() {
            buf = tf_box.transform(buf);
        }
        buf
    }

    fn reset(&mut self) {
        for tf_box in self.chain.iter_mut() {
            tf_box.reset();
        }
    }
}

#[cfg(feature = "rayon")]
pub trait ParallelTransform<S>: Transform<S> + 'static + Send {}
#[cfg(feature = "rayon")]
impl<S, T: Transform<S> + 'static + Send> ParallelTransform<S> for T {}

#[cfg(not(feature = "rayon"))]
pub trait ParallelTransform<S>: Transform<S> + 'static {}
#[cfg(not(feature = "rayon"))]
impl<S, T: Transform<S> + 'static> ParallelTransform<S> for T {}
/// Chain transforms together in parallel.
/// All transforms are weighted equally (for now)
pub struct ParallelChain<S> {
    chain: Vec<Box<dyn ParallelTransform<S>>>,
}

impl<S> ParallelChain<S> {
    pub fn new() -> Self {
        Self {chain: vec![]}
    }

    /// convert a Transform into a single-element Chain
    pub fn from(tf: impl ParallelTransform<S>) -> Self {
        Self { chain: vec![Box::new(tf)] }
    }

    /// add a Transform to the chain
    pub fn push(mut self, tf: impl ParallelTransform<S>) -> Self {
        self.chain.push(Box::new(tf));
        self
    }

    /// get chain length
    pub fn len(&self) -> usize {
        self.chain.len()
    }

    /// get a reference to the nth chain element
    pub fn get(&self, n: usize) -> Option<&dyn ParallelTransform<S>> {
        match self.chain.get(n) {
            Some(tf_box) => Some(tf_box.as_ref()),
            None => None
        }
    }

    /// get a mutable reference to the nth chain element
    pub fn get_mut(&mut self, n: usize) -> Option<&mut dyn ParallelTransform<S>> {
        match self.chain.get_mut(n) {
            Some(tf_box) => Some(tf_box.as_mut()),
            None => None
        }
    }

    /// remove the nth chain element (panic if out of bounds)
    pub fn remove(&mut self, n: usize) -> Option<Box<dyn ParallelTransform<S>>> {
        if n >= self.chain.len() {
            None
        } else {
            Some(self.chain.swap_remove(n))
        }
    }
}

impl ParallelChain<Float> {
///  Create a wet/dry mix
/// 
/// Internally, combine tf (+amp) with PassThrough (+amp)
    pub fn wetdry(tf: impl ParallelTransform<Float>, wetness: Float) -> Self {
        assert!(wetness >= 0.0 && wetness <= 1.0, "wetness must be between 0 and 1");
        
        let wet = chain!(tf, Gain::new(wetness));
        let dry = Gain::new(1.0 - wetness);
        ParallelChain::from(wet).push(dry)
    }
}
/// Macro to chain together multiple transforms
/// 
/// parallel_chain(tf1, tf2, ...) -> ParallelChain::from(tf1).push(tf2).push(...)
#[macro_export]
macro_rules! parallel_chain {
    ( $tf0:expr, $( $tf:expr ),* ) => {
        ParallelChain::from($tf0)
        $(
            .push($tf)
        )*
    };
}

// Merge 2 ParallelChains together into a single ParallelChain
impl<S> Add for ParallelChain<S> {
    type Output = ParallelChain<S>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut chain = self.chain;
        let mut chain2 = rhs.chain;
        chain.append(&mut chain2);
        Self { chain }
    }
}

#[cfg(feature = "rayon")]
impl<S> Transform<S> for ParallelChain<S>
where S: Copy+Zero+Send+Sync {
    fn transform(&mut self, buf: SampleBuffer<S>) -> SampleBuffer<S> {
        let data_final_lock = Mutex::new(vec![S::zero(); buf.len()]);

        // run each transform, scale result, and add to data_final
        self.chain.par_iter_mut().for_each(|tf_box| {
            let buf_part = tf_box.transform(buf.clone());
            let mut data_final = data_final_lock.lock()
                .expect("data_final is poisoned due to panic in another thread");
            *data_final = vec_add(&data_final, buf_part.data());
        });

        // assign new data to buf
        buf.with_data(
            data_final_lock.into_inner()
            .expect("data_final is poisoned due to panic in another thread")
        )
    }

    fn reset(&mut self) {
        self.chain.par_iter_mut().for_each(|tf_box| {
            tf_box.reset();
        });
    }
}

#[cfg(not(feature = "rayon"))]
impl<S> Transform<S> for ParallelChain<S>
where S: Copy+Zero {
    fn transform(&mut self, buf: SampleBuffer<S>) -> SampleBuffer<S> {
        // data_final = final data vector (as float)
        let mut data_final = vec![S::zero(); buf.len()];

        // run each transform, scale result, and add to data_final
        for tf_box in self.chain.iter_mut() {
            let buf_part = tf_box.transform(buf.clone());
            data_final = vec_add(&data_final, buf_part.data());
        }

        // assign new data to buf
        buf.with_data(data_final)
    }

    fn reset(&mut self) {
        for tf_box in self.chain.iter_mut() {
            tf_box.reset();
        }
    }
}

// Gain: scale signal up or down.
pub struct Gain {
    gain: Float, // multiplier
}

#[allow(dead_code)]
impl Gain {
    pub fn new(gain: Float) -> Self {
        Self { gain }
    }
    
    /// invert signal polarity
    pub fn inverter() -> Self {
        Self { gain: -1.0 }
    }
    
    /// dB change = 20 * log10(gain)
    pub fn db(db: Float) -> Self {
        let gain = Self::db2gain(db);
        Self { gain }
    }

    /// convert dB to gain multiplier
    fn db2gain(db: Float) -> Float {
        10.0.powf(db / 20.0)
    }
}

impl Transform<Float> for Gain {
    fn transform(&mut self, buf: SampleBuffer<Float>) -> SampleBuffer<Float> {
        let scaled = vec_scale(buf.data(), self.gain);
        buf.with_data(scaled)
    }

    fn reset(&mut self) {}
}

/// Conv1d: Convolve signal with a kernel.
/// This can model any FIR filter
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

    /// Create digital butterworth filter,
    /// using a bilinear transform,
    ///     W_cont = 2 * fs * tan(w_disc / 2)
    /// then sampling the freq response,
    /// then an IFFT.
    /// 
    /// n   digital filter kernel length
    /// fc  cutoff freq (hz)
    /// ord analog filter order
    /// fs  sample rate (hz)
    pub fn butterworth(n: usize, fc: Float, ord: u32, fs: Float) -> Self {
        const OVERSAMPLE_FACTOR: usize = 15;
        let wc = hz2rads(fc); // cutoff freq in rad/s
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

impl Transform<Float> for Conv1d {
    // apply (causal) filter to buffer,
    // this will delay signal by k-1 samples
    fn transform(&mut self, buf: SampleBuffer<Float>) -> SampleBuffer<Float> {
        // note: multi-channel vectors are interleaved:
        // [left, right, left, right, ...]
        let numch = buf.channels();
        let data = buf.data();
        let k = self.kernel.len();
        let n = data.len();
        let npad = numch as usize * (k-1); // amount of padding - LEFT SIDE ONLY

        // make padded array (as float)
        let mut buf_padded = Vec::with_capacity(npad + data.len());
        if self.lastchunk.len() == npad {
            buf_padded.append(&mut self.lastchunk);
        } else {
            buf_padded.append(&mut vec![0.0; npad]);
        }
        buf_padded.extend(&mut data.iter());

        // make output array
        let mut output = Vec::with_capacity(data.len());

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
            output.push(res);
        }

        // save last chunk (n*(k-1)) samples
        self.lastchunk = data[data.len()-npad..].into();

        // move buffer pointer to output vector
        buf.with_data(output)
    }

    // delete cached data chunk
    fn reset(&mut self) {
        self.lastchunk = vec![];
    }
}

/* Represent systems as difference equations,
 * which seems to be way faster than Conv1d
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
    chcount: ChannelCount          // number of channels
}

/* build a linear Transform from transfer function coefficients
 *
 * `numer` = numerator coefficients (from z^0 to z^{-Inf})
 * `denom` = denominator coefficients
 */
impl DiffEq {
    const DEFAULT_CH_CAPACITY: usize = 2;

    pub fn new(numer: Vec<Float>, denom: Vec<Float>) -> Self {
        let chcount = 0 as ChannelCount;
        let xvals = Vec::with_capacity(Self::DEFAULT_CH_CAPACITY);
        let yvals = Vec::with_capacity(Self::DEFAULT_CH_CAPACITY);

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
            chcount
        }
    }

    /// use add_ch at the beginning of transform()
    /// to make sure x/yvals length matches number of buffer channels
    /// this avoids having to hardcode channelct inside DiffEq object
    fn add_ch(&mut self) {
        self.xvals.push(VecDeque::from(vec![0.0; self.xcoeff.len()]));
        self.yvals.push(VecDeque::from(vec![0.0; self.ycoeff.len()]));
        self.chcount += 1;
    }

    /// Merge two DiffEqs together in parallel,
    /// instead of using a ParallelChain.
    /// 
    /// BORING MATH:
    /// Assume system 1 has transfer coeffs Y(z): a0..ak and X(z): c0..cm
    ///    and system 2 has Y(z): b0..bp and X(z): d0..dq
    /// Then multiplying H1(z)*H2(z) gives:
    ///     [(a0..ak)(d0..dq) + (b0..bp)(c0..cm)] / [(c0..cm)(d0..dq)]
    /// Finally, multiplying/adding all of these polynomicals gives the diffeq
    ///   coeffs of the summed system
    pub fn merge_parallel(self, rhs: Self) -> Self {
        let x1 = self.xcoeff;
        let y1 = self.ycoeff;
        let x2 = rhs.xcoeff;
        let y2 = rhs.ycoeff;
        let x1_y2_prod = vec_mul(&x1, &y2);
        let x2_y1_prod = vec_mul(&x2, &y1);

        let xcoeff = vec_add(&x1_y2_prod, &x2_y1_prod);
        let ycoeff = vec_mul(&y1, &y2);
        Self::new(xcoeff, ycoeff)
    }

    /// Merge two DiffEqs together in series,
    /// instead of using a Chain.
    /// 
    /// BORING MATH:
    /// Combine into one big DiffEq by multiplying the x and y polynomial
    ///   coeffs between `self` and `other`.
    pub fn merge_series(self, other: Self) -> Self {
        let x1 = self.xcoeff;
        let y1 = self.ycoeff;
        let x2 = other.xcoeff;
        let y2 = other.ycoeff;

        let xcoeff = vec_mul(&x1, &x2);
        let ycoeff = vec_mul(&y1, &y2);
        Self::new(xcoeff, ycoeff)
    }

    /// 1st order butterworth
    /// 
    ///              W + W z^{-1}
    /// H(z) = ------------------------
    ///        (W + 1) + (W - 1) z^{-1}
    /// 
    pub fn butterworth1(fc: Float, fs: Float) -> Self {
        let wc = hz2rads(fc);
        let w = (wc * 0.5 / fs).tan();

        let numer = vec![w, w];
        let denom = vec![w + 1.0, w - 1.0];

        Self::new(numer, denom)
    }

    /// 2nd order butterworth (UNSTABLE AND PROBABLY WRONG)
    /// 
    ///                         W^2 + 2 W^2 z^{-1} w^2 + z^{-2}
    /// H(z) = ------------------------------------------------------------------
    ///        1 + W sqrt2 + W^2 + 2(W^2 - 1) z^{-1} + (W^2 - W sqrt2 + 1) z^{-2}
    pub fn butterworth2(fc: Float, fs: Float) -> Self {
        let wc = hz2rads(fc);
        let w = (wc * 0.5 / fs).tan();
        let w2 = w.powi(2);
        let wroot2 = w * SQRT_2;

        let numer = vec![w2, 2.0*w2, w2];
        let denom = vec![1.0 + wroot2 + w2, 2.0*w2 - 2.0, w2 - wroot2 + 1.0];

        Self::new(numer, denom)
    }

    /// All pass filter (aka phase shifter)
    /// 
    /// BORING MATH:
    /// all-pass has a pole and zero pair reflected across the unit circle
    /// e.g. pole = r exp(jw), zero = 1/r exp(jw)
    /// 
    /// for a pole at a and a zero at 1/a*:
    ///             1 - (1/a*) z^-1       -a* + z^-1
    /// H(z) = -a* _________________  =  ____________
    ///               1 - a z^-1          1 - a z^-1
    /// 
    /// to have real coefficients, combine (above) with another allpass with
    /// pole=a*, zero=1/a:
    /// 
    ///               a - b z^-1 + 1
    /// H_real(z) = ___________________
    ///              1 - b^-1 + a z^-2
    /// where a = |pole|^2, b = 2*Re[pole]
    /// 
    /// note: pole angle = frequency of maximum phase-shift,
    ///       pole magnitude ~ sharpness of phase-shift
    fn allpass_complex(pole: CFloat) -> Self {
        let a = pole.norm_sqr(); // |pole|^2
        let b = -2.0 * pole.re;  // -2 Re[pole]
        let numer = vec![a, b, 1.0];
        let denom = vec![1.0, b, a];
        Self::new(numer, denom)
    }

    /// all-pass with a real-valued pole (r), and a zero at 1/r
    fn allpass_real(pole: Float) -> Self {
        let numer = vec![-pole, 1.0];
        let denom = vec![1.0, -pole];
        Self::new(numer, denom)
    }

    /// Create an all-pass by specifying phase angle and r-value
    /// 
    /// phase = phase angle (rad) of pole
    /// r = pole magnitude
    pub fn allpass(phase: Float, r: Float) -> Self {
        let phase = phase.rem_euclid(TAU);
        if phase == 0.0 {
            Self::allpass_real(r)
        } else if phase == PI {
            Self::allpass_real(-r)
        } else {
            Self::allpass_complex(Complex::from_polar(r, phase))
        }
    }
}

impl From<Conv1d> for DiffEq {
    fn from(conv: Conv1d) -> Self {
        // set Y(z) = H(z) and X(z) = 1
        let numer = conv.kernel;
        let denom = vec![1.0];
        Self::new(numer, denom)
    }
}

impl Transform<Float> for DiffEq {
    fn reset(&mut self) {
        self.xvals.clear();
        self.yvals.clear();
        self.chcount = 0;
    }

    fn transform(&mut self, buf: SampleBuffer<Float>) -> SampleBuffer<Float> {
        let chcount = buf.channels();
        let chsize = chcount as usize;
        let data = buf.data();
        let mut data_new = vec![0.0; data.len()];

        let mut x_sum: Float; // x_sum = a0 x[n] + ... + ak x[n-k]
        let mut y_sum: Float; // y_sum = b1 y[n-1] + ... + bm y[n-m]
        let mut y_new: Float; // current y[n] value
        let ylen = self.ycoeff.len(); // number of y-coefficients

        // make sure channel count = buffer channels
        if self.chcount > chcount {
            self.reset();
        }
        while self.chcount < chcount {
            self.add_ch();
        }

        for ch in 0..chsize {
            let x_vec = self.xvals.get_mut(ch).unwrap();
            let y_vec = self.yvals.get_mut(ch).unwrap();

            let sample_iter = zip(data.iter(), data_new.iter_mut())
                .skip(ch)
                .step_by(chsize);
            for (samp, newsamp) in sample_iter {
                // get newest x-value (x[n]) and drop oldest value
                x_vec.pop_front();
                x_vec.push_back(*samp as Float);

                // compute sum[ x_0 x[n] ... x_k x[n-k] ]
                x_sum = zip(self.xcoeff.iter(), x_vec.iter())
                    .map(|(&coeff, &x)| {coeff * x})
                    .sum();
                // compute sum[ y_1 y[n] ... y_m x[n-m] ]
                y_vec.pop_front();
                y_sum = zip(self.ycoeff.iter(), y_vec.iter())
                    .take(ylen - 1)
                    .map(|(&coeff, &y)| {coeff * y})
                    .sum();

                // compute y[n], pop oldest y-value, write to buffer
                y_new = (x_sum - y_sum) / self.ycoeff.get(ylen - 1).unwrap();
                y_vec.push_back(y_new);
                *newsamp = y_new;
            }
        }

        // replace old buffer with new
        buf.with_data(data_new)
    }
}

/*
 * ToMono: average all signals together
 */
pub struct ToMono;

impl Transform<Float> for ToMono {
    fn transform(&mut self, buf: SampleBuffer<Float>) -> SampleBuffer<Float> {
        let n_ch = buf.channels() as usize;
        let mut buf_new = buf;
        let data_new = buf_new.data_mut();
        for chunk in data_new.chunks_exact_mut(n_ch) {
            let avg = chunk.iter().sum::<Float>() / n_ch as Float;
            chunk.fill(avg);
        }
        buf_new
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
    /// Pan audio to the left.
    /// 
    /// `pan_amt` ranges from 0 (no change) to 1 (all the way left)
    /// 
    /// Panic if pan_amt is not between 0 and 1.
    pub fn pan_left(pan_amt: Float) -> Self {
        assert!(pan_amt >= 0.0 && pan_amt <= 1.0, "pan amount must be between 0-1");
        Self {
            left_wt: [1.0, pan_amt],
            right_wt: [0.0, 1.0 - pan_amt]
        }
    }

    /// Same as pan_left but in the other direction.
    pub fn pan_right(pan_amt: Float) -> Self {
        assert!(pan_amt >= 0.0 && pan_amt <= 1.0, "pan amount must be between 0-1");
        Self {
            left_wt: [1.0 - pan_amt, 0.0],
            right_wt: [pan_amt, 1.0]
        }
    }

    /// Swap left and right channels
    pub fn inverter() -> Self {
        Self {
            left_wt: [0.0, 1.0],
            right_wt: [1.0, 0.0]
        }
    }
}

impl Transform<Float> for Pan {
    fn transform(&mut self, buf: SampleBuffer<Float>) -> SampleBuffer<Float> {
        let numch = buf.channels();
        assert!(numch == 2, "Sample buffer must be stereo");
        let mut buf_new = buf;
        for chunk in buf_new.data_mut().chunks_exact_mut(numch as usize) {
            let samp_l = zip(&self.left_wt, chunk.iter())
                .map(|(&x, &y)| {x * y})
                .sum();
            let samp_r = zip(&self.right_wt, chunk.iter())
                .map(|(&x, &y)| {x * y})
                .sum();
            chunk[0] = samp_l;
            chunk[1] = samp_r;
        }
        buf_new
    }

    fn reset(&mut self) {}
}

/// Apply multiple phase shifts with wet/dry control.
/// 
/// Internally, this is a Chain of evenly spaced all-pass filters,
/// all contained in a wet/dry ParallelChain.
pub struct Phaser {
    chain: ParallelChain<Float>
}

impl Phaser {
    /* Construct a new phaser
     *
     * n: number of poles (actually x2 bc poles are mirrored across real axis)
     * angle [deg]: phase angle
     * rval[0 to 1]: pole magnitude (smaller = sharper phase-warp)
     */
    pub fn new(n: u32, rval: Float, wetness: Float) -> Self {
        assert!(rval > 0.0, "pole magnitude must be positive");
        assert!(rval < 1.0, "pole magnitude > 1 is unstable");

        let angles = (0..n)
            .map(|i| i as Float * (PI / n as Float));
        let mut allpass_chain = Chain::new();
        for ang in angles {
            allpass_chain = allpass_chain.push(DiffEq::allpass(ang, rval));
        }
        let chain = ParallelChain::wetdry(allpass_chain, wetness);
        Self {chain}
    }
}

impl Transform<Float> for Phaser {
    fn transform(&mut self, buf: SampleBuffer<Float>) -> SampleBuffer<Float> {
        self.chain.transform(buf)
    }

    fn reset(&mut self) {
        self.chain.reset();
    }
}

/// Downsample a signal by keeping 1 out of every n samples.
pub struct Decimator {
    factor: u32,
    remainder: usize
}

impl Decimator {
    pub fn new(factor: u32) -> Self {
        Self { factor, remainder: 0 }
    }
}

impl Transform<Float> for Decimator {
    fn transform(&mut self, buf: SampleBuffer<Float>) -> SampleBuffer<Float> {
        let n_ch = buf.channels() as usize;
        let chunk_len = self.factor as usize;
        let chunk_sz = chunk_len * n_ch;

        // fast-forward to beginning of next chunk
        let n_skip_start = (chunk_len - self.remainder).rem_euclid(chunk_len) * n_ch;
        // skip entire buffer, if needed
        if n_skip_start >= buf.len() {
            self.remainder += buf.len() / n_ch;
            return buf.with_data(vec![])
        }
        let mut data = &buf.data()[n_skip_start..];

        // buffer sizes
        let buf_size = data.len();
        let buf_len = buf_size / n_ch;
        let n_chunks = buf_len / chunk_len;
        
        // if less than 1 full chunk exists, just take the 1st sample
        if n_chunks == 0 {
            self.remainder = buf_len;
            let data_new = data[..n_ch].into();
            return buf.with_data(data_new)
        }
        
        // take the first [n_ch] out of every [factor] elements,
        let mut data_new = Vec::with_capacity(n_chunks);
        let mut chunk: &[Float];
        for _i in 0..n_chunks {
            (chunk, data) = data.split_at(chunk_sz);
            data_new.extend_from_slice(&chunk[..n_ch]);
        }

        self.remainder = data.len() / n_ch;
        buf.with_data(data_new)
    }

    fn reset(&mut self) {
        self.remainder = 0;
    }
}

/// Resample a signal using sinc interpolation.
pub struct Resampler {
    fs: Float
}

impl Resampler {
    pub fn new(fs: SampleRate) -> Self {
        Self {fs: fs as Float}
    }
}

impl Transform<Float> for Resampler {
    fn transform(&mut self, buf: SampleBuffer<Float>) -> SampleBuffer<Float> {
        let fs2_fs1 = self.fs / buf.fs() as Float;
        let n_ch = buf.channels() as usize;
        let n_in = buf.len() / n_ch;
        let n_out = ((n_in as Float) * fs2_fs1).round() as usize;
        let data = buf.data();

        let mut data_new = vec![0.0; n_out * n_ch];
        for c in 0..n_ch {
            // interpolate the single-channel signal
            let data_ch = data.iter()
                .skip(c).step_by(n_ch)
                .map(|&x| x)
                .collect::<Vec<_>>();
            // copy interpolated samples into new buffer
            for (i, &x) in interp_sinc(&data_ch[..], n_out).iter().enumerate() {
                data_new[i*n_ch + c] = x;
            }
        }
        buf.with_data(data_new)
    }

    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;
    use hound::WavSpec;
    use rand;
    use approx::relative_eq;
    use crate::utils::Float;
    use crate::buffers::{ChannelCount, SampleBuffer, SampleRate};
    use super::*;

    const NUM_CH: ChannelCount = 2;
    const FS: SampleRate = 48000;
    const BUFF_LEN: usize = 1024;

    static TEST_WAVSPEC: WavSpec = WavSpec {
        channels: NUM_CH,
        sample_rate: FS,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Float
    };

    // buffer filled with random numbers
    // TODO: should this have a fixed seed?
    static TEST_BUFFER: LazyLock<SampleBuffer<Float>> = LazyLock::new(|| {
        let mut data = vec![0.0; BUFF_LEN];
        rand::fill(data.as_mut_slice());
        SampleBuffer::new(data, &TEST_WAVSPEC)
    });
    // delta function
    static TEST_BUFFER_DELTA: LazyLock<SampleBuffer<Float>> = LazyLock::new(|| {
        let mut data = vec![0.0; BUFF_LEN];
        data[0] = 1.0;
        SampleBuffer::new(data, &TEST_WAVSPEC)
    });
    // all zeros
    static TEST_BUFFER_ZEROS: LazyLock<SampleBuffer<Float>> = LazyLock::new(||
        SampleBuffer::new(vec![0.0; BUFF_LEN], &TEST_WAVSPEC)
    );

    #[test]
    fn test_passthrough() {
        let mut tf = PassThrough;
        assert_eq!(
            TEST_BUFFER.data(),
            tf.transform(TEST_BUFFER.clone()).data()
        );
    }

    #[test]
    fn test_gain() {
        let buf = Gain::new(2.0).transform(TEST_BUFFER.clone());
        assert!(
            buf.data().iter()
            .zip(TEST_BUFFER.data().iter())
            .all(|(&a,&b)| relative_eq!(0.5*a, b))
        )
    }

    #[test]
    fn test_gain_zero() {
        // test X * 0 = 0
        let buf = Gain::new(0.0).transform(TEST_BUFFER.clone());
        assert!(
            buf.data().iter()
            .all(|&s| relative_eq!(s, 0.0))
        );

        // test X + (-X) = 0
        let buf = Gain::inverter().transform(TEST_BUFFER.clone());
        assert!(
            buf.data().iter()
            .zip(TEST_BUFFER.data().iter())
            .all(|(a,b)| relative_eq!(a + b, 0.0))
        );
    }

    #[test]
    fn test_conv1d_impulse_resp() {
        let mut kernel = vec![0.0; 100];
        rand::fill(kernel.as_mut_slice());
        let mut conv = Conv1d::new(kernel);
        let buf = conv.transform(TEST_BUFFER_DELTA.clone());
        assert_eq!(
            buf.data().iter().sum::<Float>(),
            conv.kernel.iter().sum()
        );
    }

    #[test]
    fn test_pan() {
        // pan left
        let mut panl = Pan::pan_left(1.0);
        let buf = panl.transform(TEST_BUFFER.clone());
        assert!(
            buf.data().iter()
            .skip(1).step_by(2)
            .all(|&s| s == 0.0)
        );

        // pan right
        let mut panr = Pan::pan_right(1.0);
        let buf = panr.transform(TEST_BUFFER.clone());
        assert!(
            buf.data().iter()
            .step_by(2)
            .all(|&s| s == 0.0)
        );

        // pan invert
        let mut inv = Pan::inverter();
        let buf = inv.transform(TEST_BUFFER.clone());
        assert!(
            buf.data().chunks_exact(2)
            .zip(TEST_BUFFER.data().chunks_exact(2))
            .all(|(d1,d2)| d1[0] == d2[1] && d1[1] == d2[0])
        );
    }

    #[test]
    fn test_mono() {
        let mut mono = ToMono;
        let buf = mono.transform(TEST_BUFFER.clone());
        assert!(
            buf.data().chunks_exact(NUM_CH as usize)
            .all(|chunk| chunk.iter().all(|&s| s == chunk[0]))
        )
    }

}