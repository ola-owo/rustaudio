use std::marker::PhantomData;
use std::{iter::zip, sync::Arc};
use std::mem;

use crate::buffers::{ChannelCount, BufferSampler, SampleBuffer};
use rustfft::{num_complex::Complex, Fft, FftNum, FftPlanner};
use ndarray::{s, Array, Array2, Array3, Axis};

use crate::Float;

/* Monitor describes tools that monitor a signal stream.
 * Similar to Transform but without modifying the signal.
 */
pub trait Monitor {
    fn transform(&mut self, buf: &SampleBuffer<Float>);
    fn reset(&mut self);
}

pub struct FftBuf<T: FftNum> {
    planner: FftPlanner<T>,
    fft: Arc<dyn Fft<T>>
}

impl<T: FftNum> FftBuf<T> {
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::<T>::new();
        let fft = planner.plan_fft_forward(n);
        Self { planner, fft }
    }

    pub fn fft(&mut self, buf: &SampleBuffer<T>) -> Array2<Complex<T>> {
        let (nch, n) = buf.dim();

        // plan fft
        self.fft = self.planner.plan_fft_forward(n);

        // build output array, size = (# channels, # samples)
        let mut fft_res = Array2::<Complex<T>>::zeros((nch, n));
        for (ch, mut ax) in fft_res.axis_iter_mut(Axis(0)).enumerate() {
            // fill with data from buf (converted to Complex)
            let buf_iter = buf.data().iter()
                .skip(ch).step_by(nch)
                .map(|&x| Complex::new(x, T::zero()));
            let arr_iter = ax.iter_mut();
            for (samp, arrayval) in zip(buf_iter, arr_iter) {
                *arrayval = samp;
            }

            // do fft
            self.fft.process(ax.as_slice_mut().unwrap());
        }

        fft_res
    }
}

pub struct STFT<T,C>
where C: BufferSampler<T> + ExactSizeIterator {
    data: Array3<Complex<T>>, // size = (n_ch, n_times, n_pts)
    // sampler: C,
    sampler: PhantomData<C>,
    nch: usize,
    ntimes: usize,
    npt: usize,
}

impl<T,C> STFT<T,C>
where T: FftNum, C: BufferSampler<T> + ExactSizeIterator {
    fn new_arr(nch: usize, ntimes: usize, npt: usize) -> Array3<Complex<T>> {
        Array3::<Complex<T>>::zeros((nch, ntimes, npt))
    }

    pub fn new(sampler: &C) -> Self {
        let nch = sampler.nch() as usize;
        let ntimes = sampler.len();
        let npt = sampler.nsamp();
        
        let data = Self::new_arr(nch, ntimes, npt);

        Self {data, sampler: PhantomData, nch, ntimes, npt}
    }

    pub fn build(&mut self, sampler: C) -> Array3<Complex<T>> {
        // let mut n_read: usize = 0;
        let mut fft = FftBuf::<T>::new(sampler.nsamp());
        let mut fftout: Array2<Complex<T>>; // shape = (n_ch, n_pts)
        // println!("sampler total len = {}", &sampler.len());
        for (t, buf) in sampler.enumerate() {

            fftout = fft.fft(&buf);
            for ch in 0..self.nch {
                self.data.slice_mut(s![ch, t, ..])
                    // .assign(&fftout.slice(s![ch, ..]));
                    .assign(&fftout.index_axis(Axis(0), ch));
            }
            // for (mut data_ch, fft_ch) in zip(self.data.axis_iter_mut(Axis(0)), fftout.axis_iter(Axis(0))) {
            //     data_ch.index_axis(Axis(0), ch).assign(&fft_ch);
            // }
        }

        // return the stft array and replace self.data with an empty array
        let empty_arr = Self::new_arr(self.nch, self.ntimes, self.npt);
        mem::replace(&mut self.data, empty_arr)
    }
}