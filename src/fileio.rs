use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::marker::PhantomData;
use std::path::Path;

use hound::{read_wave_header, Sample, WavReader, WavSamples, WavSpec, WavWriter};
use num_traits::{AsPrimitive, Float};

use crate::buffers::{ChunkSamples, JumpSamples, OverlapSamples, WavSamplesWrapper};

/* WavReaderAdapter:
 * WavReader wrapper that converts samples to floats
 * 
 * S: data type of samples in wav file
 * F: internal float type
 */
pub struct WavReaderAdapter<R,S,F> {
    pub reader: WavReader<R>,
    samp_type: PhantomData<S>,
    float_type: PhantomData<F>
}

impl<S,F> WavReaderAdapter<BufReader<File>, S, F> {
    pub fn from_path<P:AsRef<Path>>(path: P) -> Result<Self,hound::Error> {
        let reader = WavReader::open(path)?;
        Ok(Self {
            reader,
            samp_type: PhantomData,
            float_type: PhantomData
        })
    }
}

impl<R,S,F> From<WavReader<R>> for WavReaderAdapter<R,S,F> {
    fn from(value: WavReader<R>) -> Self {
        Self {
            reader: value,
            samp_type: PhantomData,
            float_type: PhantomData
        }
    }
}

impl<R,S,F> WavReaderAdapter<R,S,F>
where
    S: Sample+AsPrimitive<F>,
    F: Float+'static,
    R: Read
{
    pub fn iter(&mut self) -> WavSamplesWrapper<WavSamples<'_,R,S>,F> {
        WavSamplesWrapper::new(self.reader.samples::<S>())
    }

    pub fn iter_chunk(&mut self, buf_sz: usize) -> ChunkSamples<WavSamples<R,S>,F> {
        let wavspec = self.reader.spec();
        ChunkSamples::new(self.reader.samples::<S>(), buf_sz, wavspec)
    }

    pub fn iter_overlap(&mut self, buf_sz: usize, step_sz: usize) -> OverlapSamples<WavSamples<R,S>,F> {
        let wavspec = self.reader.spec();
        OverlapSamples::new(self.reader.samples::<S>(), buf_sz, step_sz, wavspec)
    }

    pub fn iter_jump(&mut self, buf_sz: usize, step_sz: usize) -> JumpSamples<WavSamples<R,S>,F> {
        let wavspec = self.reader.spec();
        JumpSamples::new(self.reader.samples::<S>(), buf_sz, step_sz, wavspec)
    }
}

/* WavWriterAdapter:
 * WavWriter wrapper that converts samples to floats
 * 
 * S: data type of samples in wav file
 * F: internal float type
 */
pub struct WavWriterAdapter<W,S,F>
where W: Write+Seek {
    writer: WavWriter<W>,
    samp_type: PhantomData<S>,
    float_type: PhantomData<F>
}

impl<S,F> WavWriterAdapter<BufWriter<File>, S, F> {
    pub fn from_path<P:AsRef<Path>>(path: P, spec: WavSpec) -> Result<Self,hound::Error> {
        let writer = WavWriter::create(path, spec)?;
        Ok(Self {
            writer,
            samp_type: PhantomData,
            float_type: PhantomData
        })
    }
}

impl<W,S,F> WavWriterAdapter<W,S,F>
where
    W: Write+Seek,
    S: Sample+'static+Copy,
    F: Float+AsPrimitive<S>,
{
    pub fn write_sample(&mut self, samp: F) -> Result<(),hound::Error> {
        self.writer.write_sample(samp.round().as_())?;
        Ok(())
    }

    pub fn write_buffer(&mut self, buf: &Vec<F>) -> Result<(),hound::Error> {
        for s in buf.iter() {
            self.write_sample(*s)?;
        }
        Ok(())
    }
}

pub fn check_wav<P:AsRef<Path>>(path: P) -> Result<(),hound::Error> {
    let mut f = File::open(path)?;
    read_wave_header(&mut f)?;
    Ok(())
}

pub fn read_wav<P:AsRef<Path>>(path: P) -> Result<WavReader<BufReader<File>>, hound::Error> {
    let reader = WavReader::open(path)?;
    Ok(reader)
}
