pub mod monitors;
pub mod transforms;
pub mod buffers;
pub mod spectral;
pub mod plot;
pub mod utils;
pub mod fileio;

use std::{io::{Seek, Write}, path::Path};
use num_traits::{AsPrimitive, Float};
use hound::Sample;

use buffers::BufferSamples;
use transforms::Transform;
use fileio::*;

pub fn read_transform_write<B,T,W,F,S>(
    sampler: B,
    writer: &mut WavWriterAdapter<W,S,F>,
    tf: &mut T,
) -> Result<(), Box<dyn std::error::Error>>
where 
    B: BufferSamples<F>,
    T: Transform<F>,
    W: Write+Seek,
    F: Float+AsPrimitive<S>,
    S: Sample+'static+Copy,
{
    for mut buf in sampler {
        buf = tf.transform(buf);
        writer.write_buffer(buf.data())?;
    }
    Ok(())
}

pub fn spectrogram<B,P>(
    sampler: B,
    outpath: P,
    title: String,
) -> Result<(), Box<dyn std::error::Error>>
where
    B: BufferSamples<utils::Float> + ExactSizeIterator,
    P: AsRef<Path>,
{
    let stft_data = spectral::stft(sampler);
    plot::spectrogram_log(&outpath, title, &stft_data)?;
    Ok(())
}