use std::marker::PhantomData;
use std::mem;

use crate::buffers::BufferSamples;
use rustfft::{num_complex::Complex, FftNum};
use ndarray::{s, Array2, Array3, Axis};

use crate::utils::Float;
use crate::monitors::{Monitor, DFT};

pub struct STFTBuilder<T,C>
where
    C: BufferSamples<T> + ExactSizeIterator,
    T: FftNum
{
    data: Array3<Complex<T>>, // shape = (n_ch, n_times, n_pts)
    fft: DFT<T>,
    sampler: PhantomData<C>,
    nch: usize,
    ntimes: usize,
    npt: usize,
}

impl<T,C> STFTBuilder<T,C>
where
    C: BufferSamples<T> + ExactSizeIterator,
    T: FftNum + Default
{
    fn new_arr(nch: usize, ntimes: usize, npt: usize) -> Array3<Complex<T>> {
        Array3::<Complex<T>>::zeros((nch, ntimes, npt))
    }

    pub fn new(sampler: &C) -> Self {
        let nch = sampler.nch() as usize;
        let ntimes = sampler.len();
        let npt = sampler.nsamp();
        let fft = DFT::new();
        
        let data = Self::new_arr(nch, ntimes, npt);

        Self {data, fft, sampler: PhantomData, nch, ntimes, npt}
    }

    pub fn build(&mut self, sampler: C) -> Array3<Complex<T>>
    where C: BufferSamples<T> + ExactSizeIterator {
        let mut fftout: Array2<Complex<T>>; // shape = (n_ch, n_pts)
        for (t, buf) in sampler.enumerate() {
            self.fft.monitor(&buf);
            let n_pts = buf.dim().1;
            fftout = self.fft.pop_output();
            self.data.slice_mut(s![.., t, ..])
                .assign(&fftout.slice(s![.., ..n_pts]));
        }

        // return the stft array and replace self.data with an empty array
        let empty_arr = Self::new_arr(self.nch, self.ntimes, self.npt);
        mem::replace(&mut self.data, empty_arr)
    }
}

pub struct STFTPlotData {
    pub stft_data: Array2<Float>,
    pub fvals: Vec<Float>,
    pub tvals: Vec<Float>
}

pub fn stft<S>(sampler:S) -> STFTPlotData
where S: BufferSamples<Float> + ExactSizeIterator {
    // build stft array
    // also save time and freq step-sizes for later
    let wavspec = sampler.wavspec();
    let fs = wavspec.sample_rate as Float;
    let tstep = sampler.step_size() as Float / fs;
    let fstep = fs as Float / sampler.buffer_size() as Float;
    let mut stft_builder = STFTBuilder::new(&sampler);
    let stft_data_raw = stft_builder.build(sampler);

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

    STFTPlotData {
        stft_data,
        fvals,
        tvals
    }
}