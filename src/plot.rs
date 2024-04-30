use std::path::Path;
use std::iter::repeat;

use itertools::izip;
use num_traits::AsPrimitive;
use ndarray::{Array1, Array2, Axis};
use plotters::prelude::*;

use crate::buffers::SampleRate;
use crate::utils::{Float, interp_sinc};

#[allow(dead_code)]
pub fn spectrogram<P: AsRef<Path>>(
    fname: &P, arr: &Array2<Float>, fs: SampleRate, title: String
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(fname, (1440, 1080))
        .into_drawing_area();
    // root.fill(&WHITE)?;
    root.fill(&VulcanoHSL::get_color(0.0))?;

    // axis limits and step sizes
    let (ntimes, npts) = arr.dim();
    let fs = fs as f64;
    let fstep = fs * 0.5 / npts as f64;
    let fvals = (0..npts)
        .map(|x| x as f64 * fstep)
        .collect::<Vec<_>>();
    // TODO: change tvals and tstep to seconds
    let tvals: Vec<f64> = (0..ntimes)
        .map(|x| x as f64)
        .collect();

    let mut chart = ChartBuilder::on(&root)
        .caption(title, 40)
        .set_label_area_size(LabelAreaPosition::Top, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .build_cartesian_2d(0.0..ntimes as f64, 0.0..0.5*fs)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("Time point")
        .y_desc("Frequency (Hz)")
        .draw()?;

    chart.draw_series(
        (0..ntimes).map(|it| repeat(it).zip(0..npts))
            .flatten()
            .map(|(ixt,ixf)| {
                let t0 = tvals[ixt];
                let f0 = fvals[ixf];
                // let t1 = t0 + tstep;
                // let f1 = f0 + fstep;
                let y = arr[[ixt,ixf]];
                // let y = y.atan() * 2.0 * FRAC_1_PI;
                // let y = y.min(1.0);
                // Rectangle::new(
                //     [(t0, f0 + f64::MIN_POSITIVE), (t1, f1)],
                //     VulcanoHSL::get_color(y)
                // )
                Pixel::new(
                    (t0, f0 + f64::MIN_POSITIVE),
                    VulcanoHSL::get_color(y)
                )
            })
    )?;

    root.present()?;
    Ok(())
}

pub fn spectrogram_log<P: AsRef<Path>>(
    fname: &P, arr: &Array2<Float>, fvals: Vec<Float>, tvals: Vec<Float>, title: String
) -> Result<(), Box<dyn std::error::Error>> {
    const PAD_TOP: u32 = 40;
    const PAD_BOTTOM: u32 = 60;
    const PAD_LEFT: u32 = 60;
    const PAD_RIGHT: u32 = 30;

    let (ntimes, npts) = arr.dim();
    let img_width = ntimes as u32 / 8 + PAD_LEFT;
    let img_height = 15 * (npts as f64).log2().floor() as u32 + PAD_TOP + PAD_BOTTOM;

    let root = BitMapBackend::new(fname, (img_width, img_height))
        .into_drawing_area();
    // root.fill(&WHITE)?;
    root.fill(&VulcanoHSL::get_color(0.0))?;

    // axis limits and step sizes
    // let fs = fs as f64;
    // let fstep = fs * 0.5 / npts as f64;
    // let fvals = (0..npts)
    //     .map(|x| x as f64 * fstep)
    //     .collect::<Vec<_>>();
    // get log-spaced frequency values to show
    let log_max = (npts as f64).log2().floor();
    let fvals_log_ix = Array1::logspace(2.0, 1.0, log_max, log_max as usize);
    let fvals_log_ix = fvals_log_ix.map(|x| x.round() as usize - 1);
    let fvals_log = fvals_log_ix.map(|&i| fvals[i]); // log-spaced freq values
    let fvals_log_len = fvals_log.len();
    let npts_log_interp = 15 * fvals_log_len;
    let fvals_log_interp = Array1::<f64>::geomspace(fvals_log[0].as_(), fvals_log[fvals_log_len-1].as_(), npts_log_interp).unwrap();

    /*
    0.5 -> stft overlap 
    2*npts -> fft length (including negative freqs)
    1/fs -> secs per sample
    */
    // let tstep = 0.5 * npts as f64 * 2.0 / fs;
    // let tvals: Vec<_> = (0..ntimes)
    //     .map(|x| x as f64 * tstep)
    //     .collect();


    let mut chart = ChartBuilder::on(&root)
        .caption(title, 40)
        .set_label_area_size(LabelAreaPosition::Top, PAD_TOP)
        .set_label_area_size(LabelAreaPosition::Bottom, PAD_BOTTOM)
        .set_label_area_size(LabelAreaPosition::Left, PAD_LEFT)
        .set_label_area_size(LabelAreaPosition::Right, PAD_RIGHT)
        .build_cartesian_2d(tvals[0]..tvals[tvals.len()-1], (fvals_log_interp[0]..fvals_log_interp[npts_log_interp-1]).log_scale())?;
    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("Time (secs)")
        .y_desc("Frequency (Hz)")
        .draw()?;

    // let pts = (0..ntimes).map(|it| repeat(it).zip(0..npts))          
    //     .flatten()
    //     .map(|(ixt,ixf)| {
    //         let &t0 = tvals.get(ixt).unwrap();
    //         let &f0 = fvals.get(ixf).unwrap();
    //         let t1 = t0 + tstep;
    //         let f1 = f0 + fstep;
    //         let y = *arr.get((ixt, ixf)).unwrap() as f64;
    //         [(t0,0.0,f0), (t1, y, f1)]
    //     })
    //     .collect::<Vec<_>>();

    // chart.draw_series(
    //     (0..ntimes).map(|it| repeat(it).zip(fvals_log_ix.iter()))
    //         .flatten()
    //         .map(|(ixt, &ixf)| {
    //             let &t0 = tvals.get(ixt).unwrap();
    //             let &f0 = fvals.get(ixf).unwrap();
    //             // let t1 = t0 + tstep;
    //             // let f1 = f0 + fstep;
    //             let y = *arr.get((ixt, ixf)).unwrap() as f64;
    //             // Rectangle::new(
    //             //     [(t0, f0 + f64::MIN_POSITIVE), (t1, f1)],
    //             //     VulcanoHSL::get_color(y)
    //             // )
    //             Pixel::new(
    //                 (t0, f0),
    //                 VulcanoHSL::get_color(y)
    //             )
    //         })
    // )?;
    chart.draw_series(
        (0..ntimes).map(|i| {
            let hvals = arr.index_axis(Axis(0), i); // frequency spectrum H[f] at time i
            let hvals_logsp = fvals_log_ix.map(|&i| hvals[i]); // log-spaced H[f] values
            let mut hvals_logsp_interp = interp_sinc(hvals_logsp.as_slice().unwrap(), npts_log_interp);
            for h in hvals_logsp_interp.iter_mut() {
                *h = h.min(1.0);
            }
            let t = tvals[i];

            // repeat(t).zip(0..npts_log_interp)
            izip!(repeat(t), fvals_log_interp.iter(), hvals_logsp_interp)
                .map(|(t, &f, h)| {
                    // let &t0 = tvals.get(ixt).unwrap();
                    // let &f0 = fvals.get(ixf).unwrap();
                    // Rectangle::new(
                    //     [(t0, f0 + f64::MIN_POSITIVE), (t1, f1)],
                    //     VulcanoHSL::get_color(y)
                    // )
                    Pixel::new((t, f), VulcanoHSL::get_color(h))
                })
        }).flatten()
    )?;

    root.present()?;
    Ok(())
}