use std::{io::{Read, Seek, Write}, marker::PhantomData};
use hound::{Sample, WavSamples, WavSpec, WavWriter};
use num_traits::{AsPrimitive,Float};

pub type ChannelCount = u16;
pub type SampleRate = u32;

// Converter between wav sample type (usually i16) and SampleBuffer type (float)
pub struct SampleConverter<S,F> {
    samp_type: PhantomData<S>,
    float_type: PhantomData<F>
}

impl<S,F> SampleConverter<S,F>
where F: Float+AsPrimitive<S>, S:'static+Sample+Copy+AsPrimitive<F> {
    pub fn new<'wr,R>(_wav_samples: &WavSamples<'wr,R,S>) -> Self {
        Self {
            samp_type: PhantomData::<S>,
            float_type: PhantomData::<F>
        }
    }

    // pub fn vec_to_float(v: &Vec<S>) -> Vec<F> {
    //     v.iter().map(|x| {x.as_()}).collect()
    // }

    // pub fn vec_to_samp(v: &Vec<F>) -> Vec<S> {
    //     v.iter().map(|x| {x.as_()}).collect()
    // }

    pub fn write_buffer<W: Write+Seek>(&self, writer: &mut WavWriter<W>, buf: &Vec<F>) {
        for s in buf.iter() {
            writer.write_sample(s.as_()).expect("failed to write buffer");
        }
    }
}

/*
 * ChunkedSampler: wrapper around WavSamples that loads chunks of samples.
 * Can have any # of channels
 */
pub struct ChunkedSampler<'wr,R,S,F> {
    sample_iter: &'wr mut WavSamples<'wr,R,S>,
    buffer_cap: usize,
    wavspec: &'wr WavSpec,
    float_type: PhantomData<F>,
    samp_type: PhantomData<S>
}

impl<'wr,R,S,F> ChunkedSampler<'wr,R,S,F>
where R:Read, S:'static+Sample+Copy, F: AsPrimitive<S> {
    pub fn new(
        sample_iter: &'wr mut WavSamples<'wr,R,S>,
        buffer_cap: usize,
        wavspec: &'wr WavSpec
    ) -> Self {
        Self {
            sample_iter,
            buffer_cap,
            wavspec,
            float_type: PhantomData,
            samp_type: PhantomData
        }
    }
}

impl<'wr,R,S,F> Iterator for ChunkedSampler<'wr,R,S,F>
where R:Read, S:Sample+AsPrimitive<F>, F:'static+Copy {
    type Item = SampleBuffer<F>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut vec = Vec::with_capacity(self.buffer_cap);
        let mut sample: S;
        for _ in 0..self.buffer_cap {
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
            Some(SampleBuffer::new(vec, self.wavspec))
        }
    }
}

/*
 * OverlapSampler: wrapper around WavSamples that loads chunks of samples.
 * Similar to ChunkedSampler but with an overlapping window
 */
pub struct OverlapSampler<'wr,R,S,F> {
    sample_iter: &'wr mut WavSamples<'wr,R,S>,
    buffer_cap: usize,
    step_size: usize,
    last_chunk: Vec<F>,
    wavspec: &'wr WavSpec,
    float_type: PhantomData<F>,
    samp_type: PhantomData<S>
}

impl<'wr,R,S,F> OverlapSampler<'wr,R,S,F>
where R:Read, S:'static+Sample+Copy, F: AsPrimitive<S> {
    pub fn new(
        sample_iter: &'wr mut WavSamples<'wr,R,S>,
        buffer_cap: usize,
        step_size: usize,
        wavspec: &'wr WavSpec
    ) -> Self {
        assert!(step_size > 0 && step_size <= buffer_cap, "step size must be in range [1, buffer_cap]");
        let last_chunk = Vec::with_capacity(buffer_cap - step_size);

        Self {
            sample_iter,
            buffer_cap,
            step_size,
            last_chunk,
            wavspec,
            float_type: PhantomData,
            samp_type: PhantomData
        }
    }
}

impl<'wr,R,S,F> Iterator for OverlapSampler<'wr,R,S,F>
where R:Read, S:Sample+AsPrimitive<F>, F:'static+Copy {
    type Item = SampleBuffer<F>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut sample: S;
        
        let samps_to_load = self.buffer_cap - self.last_chunk.len();
        let mut vec = Vec::with_capacity(self.buffer_cap);
        vec.extend(self.last_chunk.drain(0..));

        for _ in 0..samps_to_load {
            if let Some(res) = self.sample_iter.next() {
                sample = res.expect("error while reading sample");
                vec.push(sample.as_());
            } else {
                break // end of iterator
            }
        }

        self.last_chunk.extend(&vec[self.step_size..]);

        if vec.is_empty() {
            None
        } else {
            Some(SampleBuffer::new(vec, self.wavspec))
        }
    }
}

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
    // constructor
    pub fn new(data: Vec<S>, wavspec: &WavSpec) -> Self {
        Self {
            data,
            numch: wavspec.channels,
            fs: wavspec.sample_rate,
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
