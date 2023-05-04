use std::fs::File;
use std::path::Path;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions, Decoder};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::errors::Error;
use symphonia::core::audio::SampleBuffer;
use fundsp::wave::Wave32;

const WAVFILE: &str = "./data/maggi.wav";

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

struct TrackDecoder {
    track_id: u32,
    reader: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>
}

fn main() {
    let path = Path::new(&WAVFILE);

    let mut track_decoder = make_track_decoder(path);

    //////
    // read & decode packets
    // let mut bufs = vec![None, None];
    let mut samplebuf = None;

    for i in 0..10 {
        // read next packet
        let packet = match track_decoder.reader.next_packet() {
            Err(Error::ResetRequired) => unimplemented!(), // tracklist changed since last read
            Err(err) => panic!("{}", err),
            Ok(packet) => packet
        };
        if packet.track_id() != track_decoder.track_id {
            continue
        }
        println!("packet {}, time: {}", i, packet.ts);

        // consume metadata
        while !track_decoder.reader.metadata().is_latest() {
            track_decoder.reader.metadata().pop();
        }

        // decode packet
        let decoded = track_decoder.decoder.decode(&packet).unwrap();
        let sigspec = decoded.spec();
        let nchan = sigspec.channels.count();
        println!("> {} channels @ {} Hz", nchan, sigspec.rate);

        // load decoded audio into sample buffer
        if samplebuf.is_none() {
            println!("> (Initializing sample buffer with capacity {}...)", decoded.capacity());
            samplebuf = Some(SampleBuffer::<f32>::new(decoded.capacity() as u64, *sigspec));
        }
        if let Some(buf) = &mut samplebuf {
            println!("> Copying into sample buffer ...");
            buf.copy_planar_ref(decoded);
            println!(">> {} samples copied.", buf.len());
        }
    }
}

fn make_track_decoder(path: &Path) -> TrackDecoder {
    let fileobj = File::open(path).expect("Couldn't open file");
    let stream = MediaSourceStream::new(Box::new(fileobj), Default::default());

    // Use the default options for metadata and format readers.
    let hint = Hint::new();
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    // Probe the media source and get a format reader
    let probed = symphonia::default::get_probe()
        .format(&hint, stream, &fmt_opts, &meta_opts)
        .expect("unsupported format");
    let reader = probed.format;

    // get the 1st readable audio track
    let track = reader.tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .expect("no supported audio tracks");
        
    // Use the default options for the decoder.
    let dec_opts: DecoderOptions = Default::default();

    // Create a decoder for the track.
    let decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .expect("unsupported codec");

    let track_id = track.id;
    println!("track ID: {}", track_id);

    TrackDecoder {
        track_id,
        reader,
        decoder,
    }

    // unimplemented!();
}
