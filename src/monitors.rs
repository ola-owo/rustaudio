// std lib imports
use std::{iter::zip, ops::Add};
use ndarray::{Array2, ArrayView2, Axis};
// external crates
use rustfft::{num_complex::Complex, FftNum, FftPlanner};
// local crates
use crate::buffers::SampleBuffer;

/* Monitor an audio buffer
 * 
 * monitor() takes a SampleBuffer and modifies the Monitor's interal state
 * reset() resets the internal state
 */
pub trait Monitor<S> {
    fn monitor(&mut self, buf: &SampleBuffer<S>);
    fn reset(&mut self);
}

// Dummy monitor that does nothing
pub struct PassThrough;

impl<S> Monitor<S> for PassThrough {
    fn monitor(&mut self, _buf: &SampleBuffer<S>) {}
    fn reset(&mut self) {}
}

// Discrete fourier transform
pub struct DFT<T:FftNum> {
    planner: FftPlanner<T>,
    output: Array2<Complex<T>> // shape = (nch, npts)
}

impl<T:FftNum+Default> DFT<T> {
    pub fn new() -> Self {
        let planner = FftPlanner::new();
        let output = Array2::default((0,0));
        Self { planner, output }
    }

    pub fn peek_output(&self) -> ArrayView2<Complex<T>> {
        self.output.view()
    }

    pub fn pop_output(&mut self) -> Array2<Complex<T>> {
        std::mem::take(&mut self.output)
    }
}

impl<T:FftNum+Default> Default for DFT<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T:FftNum+Default> Monitor<T> for DFT<T> {
    fn monitor(&mut self, buf: &SampleBuffer<T>) {
        let n = buf.len();
        let nch = buf.channels() as usize;
        let fft = self.planner.plan_fft_forward(n);
        if self.output.dim() != (nch,n) {
            self.output = Array2::<Complex<T>>::zeros((nch, n));
        }
        for (ch, mut fft_buf) in self.output.axis_iter_mut(Axis(0)).enumerate() {
            // copy per-channel sample buffer (buf.data) into fft buffer slice (fft_buf)
            let buf_iter = buf.data().iter()
                .skip(ch).step_by(nch)
                .map(|&x| Complex::new(x, T::zero()));
            let fft_buf_iter = fft_buf.iter_mut();
            for (arrayval, samp) in zip(fft_buf_iter, buf_iter) {
                *arrayval = samp;
            }

            // do FFT
            fft.process(fft_buf.as_slice_mut().unwrap());
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

// Chain of Monitors
pub struct Chain<S> {
    chain: Vec<Box<dyn Monitor<S>>>,
}

impl<S> Chain<S> {
    pub fn new() -> Self {
        Self {chain: vec![]}
    }

    pub fn from(tf: impl Monitor<S> + 'static) -> Self {
        Self {chain: vec![Box::new(tf)]}
    }

    // 'static means tf is an owned type (not a ref),
    // which is required for Box
    pub fn push(mut self, tf: impl Monitor<S> + 'static) -> Self {
        self.chain.push(Box::new(tf));
        self
    }

    pub fn len(&self) -> usize {
        self.chain.len()
    }

    // get a reference to the nth chain element
    pub fn get(&self, n: usize) -> Option<&dyn Monitor<S>> {
        match self.chain.get(n) {
            Some(tf_box) => Some(tf_box.as_ref()),
            None => None
        }
    }

    // get a mutable reference to the nth chain element
    pub fn get_mut(&mut self, n: usize) -> Option<&mut dyn Monitor<S>> {
        match self.chain.get_mut(n) {
            Some(tf_box) => Some(tf_box.as_mut()),
            None => None
        }
    }

    // remove the nth chain element (panic if out of bounds)
    pub fn remove(&mut self, n: usize) -> Option<Box<dyn Monitor<S>>> {
        if n >= self.chain.len() {
            None
        } else {
            Some(self.chain.remove(n))
        }
    }
}

#[macro_export]
macro_rules! monitor_chain {
    ( $tf0:expr, $( $tf:expr ),* ) => {
        Chain::from($tf0)
        $(
            .push($tf)
        )*
    };
}

// Merge 2 Chains together into a single Chain
impl<S> Add for Chain<S> {
    type Output = Chain<S>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut chain = self.chain;
        let mut chain2 = rhs.chain;
        chain.append(&mut chain2);
        Self { chain }
    }
}

impl<S> Monitor<S> for Chain<S> {
    fn monitor(&mut self, buf: &SampleBuffer<S>) {
        for mon in self.chain.iter_mut() {
            mon.monitor(buf);
        }
    }

    fn reset(&mut self) {
        for mon in self.chain.iter_mut() {
            mon.reset();
        }
    }
}

