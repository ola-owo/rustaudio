pub mod monitors;
pub mod transforms;
pub mod buffers;
pub mod spectral;
pub mod plot;
pub mod utils;
pub mod fileio;

use std::{io::{Seek, Write}, path::Path};

use buffers::{Sample, BufferSamples};
use num_traits::AsPrimitive;
use transforms::Transform;
use fileio::*;
use utils::Float;

pub fn read_transform_write<B,T,W,S>(
    sampler: B,
    writer: &mut WavWriterAdapter<W,S>,
    tf: &mut T,
) -> Result<(), Box<dyn std::error::Error>>
where 
    B: BufferSamples<Float>,
    T: Transform<Float>,
    W: Write+Seek,
    S: Sample,
    Float: AsPrimitive<S>
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