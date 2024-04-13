use std::path::Path;
use std::iter::repeat;

use plotters::prelude::*;
use ndarray:: Array2;

use crate::{buffers::SampleRate, Float};

pub fn spectrogram2d(fname: &Path, arr: &Array2<Float>, fs: SampleRate) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(fname, (1440, 1080))
        .into_drawing_area();
    root.fill(&WHITE)?;

    // axis limits and step sizes
    let (ntimes, npts) = arr.dim();
    let fstep = fs as f64 * 0.5 / npts as f64;
    let fvals = (0..npts)
        .map(|x| x as f64 * fstep)
        .collect::<Vec<_>>();
    // TODO: change tvals and tstep to seconds
    let tvals: Vec<f64> = (0..ntimes)
        .map(|x| x as f64)
        .collect();
    let tstep = 1.0;

    let mut chart = ChartBuilder::on(&root)
        .build_cartesian_2d(0.0..ntimes as f64, 0.0..(fs/2) as f64)?;
    chart
        .configure_mesh()
        .disable_mesh()
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
    chart.draw_series(
        (0..ntimes).map(|it| repeat(it).zip(0..npts))
            .flatten()
            .map(|(ixt,ixf)| {
                let &t0 = tvals.get(ixt).unwrap();
                let &f0 = fvals.get(ixf).unwrap();
                let t1 = t0 + tstep;
                let f1 = f0 + fstep;
                let y = *arr.get((ixt, ixf)).unwrap() as f64;
                Rectangle::new(
                    [(t0, f0), (t1, f1)],
                    VulcanoHSL::get_color(y)
                )
            })
    )?;

    Ok(())
}

pub fn spectrogram3d(fname: &Path, arr: &Array2<Float>, fs: SampleRate) {
    let root = BitMapBackend::new(fname, (1440, 1080))
        .into_drawing_area();
    root.fill(&WHITE).expect("couldn't set background color");

    // axis limits and step sizes
    let (ntimes, npts) = arr.dim();
    let fstep = fs as f64 * 0.5 / npts as f64;
    let fvals = (0..npts)
        .map(|x| x as f64 * fstep)
        .collect::<Vec<_>>();
    // TODO: change tvals and tstep to seconds
    let tvals: Vec<f64> = (0..ntimes)
        .map(|x| x as f64)
        .collect();
    let tstep = 1.0;

    let mut chart = ChartBuilder::on(&root)
        .build_cartesian_3d(0.0..ntimes as f64, 0.0..(fs/2) as f64, 0.0..1.0)
        .expect("couldn't create chart context");
    chart.configure_axes().draw().expect("couldn't draw axes");

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
    chart.draw_series(
        (0..ntimes).map(|it| repeat(it).zip(0..npts))
            .flatten()
            .map(|(ixt,ixf)| {
                let &t0 = tvals.get(ixt).unwrap();
                let &f0 = fvals.get(ixf).unwrap();
                let t1 = t0 + tstep;
                let f1 = f0 + fstep;
                let y = *arr.get((ixt, ixf)).unwrap() as f64;
                Cubiod::new(
                    [(t0,0.0,f0), (t1, y, f1)],
                    BLUE.filled(),
                    &BLACK
                )
            })
    ).unwrap();
}