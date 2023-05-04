use std::io::{Read,Write,Seek};
use hound::{WavSamples,Sample,WavWriter};

type ChannelCount = u16;
type SampleRate = u32;

/*
 * ChunkedSampler: wrapper around WavSamples that loads chunks of samples.
 * Can have any # of channels
 */
pub struct ChunkedSampler<'wr, R:Read, S:Sample> {
    sample_iter: &'wr mut WavSamples<'wr,R,S>,
    buffer_cap: usize,
    channels: ChannelCount,
    fs: SampleRate
}

impl<'wr, R:Read, S:Sample> ChunkedSampler<'wr,R,S> {
    pub fn new(
        sample_iter: &'wr mut WavSamples<'wr,R,S>,
        buffer_cap: usize,
        channels: ChannelCount,
        fs: SampleRate
    ) -> Self {
        Self {sample_iter, buffer_cap, channels, fs}
    }
}

impl<'wr, R:Read, S:Sample> Iterator for ChunkedSampler<'wr, R, S> {
    type Item = SampleBuffer<S>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut sample_read = false;
        let maxn = self.buffer_cap as i32;
        let mut vec = Vec::with_capacity(self.buffer_cap);
        let mut sample: S;
        for _ in 0..maxn {
            if let Some(res) = self.sample_iter.next() {
                sample = res.expect("error while reading sample");
                sample_read = true;
                vec.push(sample);
            } else {
                break // end of iterator
            }
        }
        if sample_read {
            Some(SampleBuffer::<S>::new(vec, self.channels, self.fs))
        } else {
            None
        }
    }
}


/*
 * SampleBuffer: Represents a chunk of interleaved multichannel data
 * Data vector is owned by this object.
 */
pub struct SampleBuffer<S> {
    data: Vec<S>,
    channels: ChannelCount,
    fs: SampleRate
}

impl<S> SampleBuffer<S> {
    pub fn new(data: Vec<S>, channels: ChannelCount, fs: SampleRate) -> Self {
        Self {data, channels, fs}
    }
    pub fn channels(&self) -> ChannelCount {
        self.channels
    }
    pub fn data(&self) -> &Vec<S> {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut Vec<S> {
        &mut self.data
    }
}


/* 
 * write_buffer: helper fn for writing a chunk to a WavWriter
 */
pub fn write_buffer<W:Write+Seek, S:Sample+Copy>(writer: &mut WavWriter<W>, buf: &Vec<S>) {
    for s in buf.iter() {
        writer.write_sample(s.clone()).expect("failed to write buffer");
    }
}

// pub trait Buffer<T> {
//     // constructor
//     fn new(data: &[T], channelct:ChannelCount) -> Self where Self: Sized;
//     // abstract methods
//     fn channelct(&self) -> ChannelCount;
//     fn len(&self) -> i32;
//     fn data(&self) -> &[T];
//     fn data_mut(&self) -> &mut [T];
//     fn chan(&self, channelct:ChannelCount) -> Vec<T>;
//     fn chan_mut(&self, channelct:ChannelCount) -> &mut Vec<T>;
// }
