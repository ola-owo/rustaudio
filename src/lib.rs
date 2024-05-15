pub mod transforms;
pub mod buffers;
pub mod spectral;
pub mod plot;
pub mod utils;

use std::io::{Write,Seek};
use num_traits::{AsPrimitive, Float};
use hound::{Sample, WavWriter};
use crate::{buffers::{BufferSamples,WavIO}, transforms::Transform};

// read, transform, and write a sampler
pub fn read_tf_write<S,F,B,T,W>(wav: WavIO<S,F>, sampler: B, mut tf: T, mut writer: WavWriter<W>)
where S: 'static+Sample+AsPrimitive<F>, F: 'static+AsPrimitive<S>+Float, B: BufferSamples<F>, T: Transform<F>, W: Write+Seek {
    for buf in sampler {
        let buf = tf.transform(buf);
        wav.write_buffer(&mut writer, buf.data());
    }
}