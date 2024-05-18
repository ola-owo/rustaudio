use std::marker::PhantomData;
use hound::{Sample, WavSpec};
use num_traits::{AsPrimitive,Float};

pub type ChannelCount = u16;
pub type SampleRate = u32;

pub const BUFFER_CAP: usize = 4096;

pub struct WavSamplesWrapper<I,F> {
    iter: I,
    ftype: PhantomData<F>
}

impl<I,F,S> WavSamplesWrapper<I,F>
where
    I: Iterator<Item = hound::Result<S>>,
    S: Sample+AsPrimitive<F>,
    F: Float+'static
{
    pub fn new(iter: I) -> Self {
        Self { iter, ftype: PhantomData }
    }
}

impl<I,F,S> Iterator for WavSamplesWrapper<I,F>
where
    I: Iterator<Item = hound::Result<S>>,
    S: Sample+AsPrimitive<F>,
    F: AsPrimitive<S>
{
    type Item = hound::Result<F>;

    fn next(&mut self) -> Option<Self::Item> {
        let res = self.iter.next()?;
        Some(res.and_then(
            |s| Ok(s.as_())
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I,F,S> ExactSizeIterator for WavSamplesWrapper<I,F>
where
    I: ExactSizeIterator<Item = hound::Result<S>>,
    S: Sample+AsPrimitive<F>,
    F: AsPrimitive<S>
{}

// BufferSamples is an iterator over SampleBuffers
pub trait BufferSamples<S>: Iterator<Item = SampleBuffer<S>> {
    // number of channels
    fn nch(&self) -> ChannelCount;

    // number of samples per channel
    fn nsamp(&self) -> usize;

    // total size of sample buffer
    fn buffer_size(&self) -> usize;

    // step size between buffers
    fn step_size(&self) -> usize;

    // underlying WavSpec
    fn wavspec(&self) -> &WavSpec;
}

/* ChunkSamples: basic BufferSamples impl with no overlap between buffers
 */
pub struct ChunkSamples<I,F> {
    sample_iter: I,
    buffer_size: usize,
    wavspec: WavSpec,
    float_type: PhantomData<F>
}

impl<I,F,S> ChunkSamples<I,F>
where
    I: Iterator<Item = hound::Result<S>>,
    S: Sample
{
    pub fn new(
        sample_iter: I,
        buffer_size: usize,
        wavspec: WavSpec
    ) -> Self {
        assert!(buffer_size > 0);
        Self {
            sample_iter,
            buffer_size,
            wavspec,
            float_type: PhantomData,
            // samp_type: PhantomData
        }
    }

}

impl<I,F,S> BufferSamples<F> for ChunkSamples<I,F>
where
    I: ExactSizeIterator<Item = hound::Result<S>>,
    S: Sample+AsPrimitive<F>,
    F: 'static+Float
{
    fn wavspec(&self) -> &WavSpec {
        &self.wavspec
    }

    fn nch(&self) -> ChannelCount {
        self.wavspec.channels
    }

    fn nsamp(&self) -> usize {
        self.buffer_size / self.wavspec.channels as usize
    }

    fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    fn step_size(&self) -> usize {
        self.buffer_size
    }
}

impl<I,F,S> Iterator for ChunkSamples<I,F>
where
    I: ExactSizeIterator<Item = hound::Result<S>>,
    F:'static+Float,
    S:Sample+AsPrimitive<F>
{
    type Item = SampleBuffer<F>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut vec = Vec::with_capacity(self.buffer_size);
        let mut sample: S;
        for _ in 0..self.buffer_size {
            if let Some(res) = self.sample_iter.next() {
                sample = res.expect("error while reading sample");
                vec.push(sample.as_());
            } else {
                break // end of iterator
            }
        }
        if vec.is_empty() {
            None
        } else {
            Some(SampleBuffer::new(vec, &self.wavspec))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.sample_iter.len().div_ceil(self.buffer_size);
        (len, Some(len))
    }
}

impl<I,F,S> ExactSizeIterator for ChunkSamples<I,F>
where
    I: ExactSizeIterator<Item = hound::Result<S>>,
    F:'static+Float,
    S:Sample+AsPrimitive<F>
{}

/* OverlapSamples: yields overlapping sample buffers
 */
pub struct OverlapSamples<I,F> {
    // sample_iter: WavSamples<'wr,R,S>,
    sample_iter: I,
    buffer_size: usize,
    step_size: usize,
    wavspec: WavSpec,
    is_empty: bool,
    last_chunk: Vec<F>,
}

impl<I,F> OverlapSamples<I,F> {
    pub fn new(
        sample_iter: I,
        buffer_size: usize,
        step_size: usize,
        wavspec: WavSpec
    ) -> Self {
        assert!(step_size > 0 && step_size <= buffer_size, "step size must be in range [1, buffer_size]");
        Self {
            sample_iter,
            step_size,
            buffer_size,
            wavspec,
            is_empty: false,
            last_chunk: Vec::with_capacity(buffer_size - step_size)
        }
    }
}

impl<I,F,S> BufferSamples<F> for OverlapSamples<I,F>
where
    I: ExactSizeIterator<Item = hound::Result<S>>,
    F:'static+Float,
    S:Sample+AsPrimitive<F>
{
    fn wavspec(&self) -> &WavSpec {
        &self.wavspec
    }

    fn nch(&self) -> ChannelCount {
        self.wavspec.channels
    }

    fn nsamp(&self) -> usize {
        self.buffer_size / self.wavspec.channels as usize
    }

    fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    fn step_size(&self) -> usize {
        self.buffer_size
    }
}

impl<I,F,S> Iterator for OverlapSamples<I,F>
where
    I: ExactSizeIterator<Item = hound::Result<S>>,
    F:'static+Float,
    S:Sample+AsPrimitive<F>
{
    type Item = SampleBuffer<F>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut sample: S;

        if self.is_empty {
            return None
        }
        
        let samps_to_load = self.buffer_size - self.last_chunk.len();
        let mut vec = Vec::with_capacity(self.buffer_size);
        vec.extend(self.last_chunk.drain(0..));

        for _ in 0..samps_to_load {
            if let Some(res) = self.sample_iter.next() {
                sample = res.expect("error while reading sample");
                vec.push(sample.as_());
            } else {
                self.is_empty = true;
                break // end of iterator
            }
        }

        // if wavsamples is exhausted, pad the buffer with 0s and don't fill last_chunk
        if self.is_empty {
            for _ in 0..(self.buffer_size - vec.len()) {
                vec.push(F::zero());
            }
        } else {
            self.last_chunk.extend(&vec[self.step_size..]);
        }

        Some(SampleBuffer::new(vec, &self.wavspec))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // 1st chunk takes BUFFER CAP samples,
        // all remaining chunks take STEP_SIZE samples
        let len = (self.sample_iter.len() - self.buffer_size)
            .div_ceil(self.step_size)
            + 2;
        (len, Some(len))
    }
}

impl<I,F,S> ExactSizeIterator for OverlapSamples<I,F>
where
    I: ExactSizeIterator<Item = hound::Result<S>>,
    F:'static+Float,
    S:Sample+AsPrimitive<F>
{}

/* JumpSamples: yields overlapping sample buffers
 */
pub struct JumpSamples<I,F> {
    // sample_iter: WavSamples<'wr,R,S>,
    sample_iter: I,
    buffer_size: usize,
    step_size: usize,
    wavspec: WavSpec,
    is_empty: bool,
    float_type: PhantomData<F>
}

impl<I,F> JumpSamples<I,F> {
    pub fn new(
        // sample_iter: WavSamples<'wr,R,S>,
        sample_iter: I,
        buffer_size: usize,
        step_size: usize,
        wavspec: WavSpec
    ) -> Self {
        assert!(step_size >= buffer_size, "step size must be in range [1, buffer_size]");
        Self {
            sample_iter,
            step_size,
            buffer_size,
            wavspec,
            is_empty: false,
            float_type: PhantomData
        }
    }
}

impl<I,F,S> BufferSamples<F> for JumpSamples<I,F>
where
    I: ExactSizeIterator<Item = hound::Result<S>>,
    F:'static+Float,
    S:Sample+AsPrimitive<F>
{
    fn wavspec(&self) -> &WavSpec {
        &self.wavspec
    }

    fn nch(&self) -> ChannelCount {
        self.wavspec.channels
    }

    fn nsamp(&self) -> usize {
        self.buffer_size / self.wavspec.channels as usize
    }

    fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    fn step_size(&self) -> usize {
        self.buffer_size
    }
}

impl<I,F,S> Iterator for JumpSamples<I,F>
where
    I: ExactSizeIterator<Item = hound::Result<S>>,
    F:'static+Float,
    S:Sample+AsPrimitive<F>
{
    type Item = SampleBuffer<F>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut sample: S;

        if self.is_empty {
            return None
        }

        // skip samples between chunks
        let nskip = self.step_size - self.buffer_size;
        for _ in 0..nskip.min(self.sample_iter.len()) {
            self.sample_iter.next();
        }

        // get chunk data
        let mut vec = Vec::with_capacity(self.buffer_size);
        for _ in 0..self.buffer_size {
            if let Some(res) = self.sample_iter.next() {
                sample = res.expect("error while reading sample");
                vec.push(sample.as_());
            } else {
                break // end of iterator
            }
        }
        if vec.is_empty() {
            None
        } else {
            Some(SampleBuffer::new(vec, &self.wavspec))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // 1st chunk takes BUFFER CAP samples,
        // all remaining chunks take STEP_SIZE samples
        let len = (self.sample_iter.len() - self.buffer_size)
            .div_ceil(self.step_size)
            + 2;
        (len, Some(len))
    }
}

impl<I,F,S> ExactSizeIterator for JumpSamples<I,F>
where
    I: ExactSizeIterator<Item = hound::Result<S>>,
    F:'static+Float,
    S:Sample+AsPrimitive<F>
{}

/* SampleBuffer: Represents a chunk of interleaved multichannel data
 * Data vector is owned by this object.
 * 
 * S: data type of samples in wav file
 * F: (float) data type of buffer
 */
#[derive(Clone)]
 pub struct SampleBuffer<S> {
    data: Vec<S>,
    numch: ChannelCount,
    fs: SampleRate,
}

impl<S> SampleBuffer<S> {
    pub fn new(data: Vec<S>, wavspec: &WavSpec) -> Self {
        Self {
            data,
            numch: wavspec.channels,
            fs: wavspec.sample_rate,
        }
    }

    pub fn with_data(self, data: Vec<S>) -> Self {
        Self {
            data,
            numch: self.numch,
            fs: self.fs
        }
    }

    // number of channels
    pub fn channels(&self) -> ChannelCount {
        self.numch
    }

    // sample rate (hz)
    pub fn fs(&self) -> SampleRate {
        self.fs
    }

    // reference to internal vec
    pub fn data(&self) -> &Vec<S> {
        &self.data
    }

    // mutable ref to internal vec
    pub fn data_mut(&mut self) -> &mut Vec<S> {
        &mut self.data
    }

    // total length of internal vec
    pub fn len(&self) -> usize {
        self.data.len()
    }

    // buffer dimensions: (n_channels, n_samples_per_channel)
    pub fn dim(&self) -> (usize, usize) {
        let nsamp_per_ch = self.data.len() / (self.numch as usize);
        (self.numch.into(), nsamp_per_ch)
    }
}
