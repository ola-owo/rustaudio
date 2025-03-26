#![allow(dead_code)]
//!
//! Random helper functions are defined here,
//! mostly related to math.
//!
use std::f32::consts::{PI, TAU};
use std::iter::repeat;
use std::ops::{Add, AddAssign, Mul, MulAssign};

use itertools::{EitherOrBoth, Itertools};
use rustfft::num_complex::Complex;
use num_traits::{AsPrimitive, Num, Zero};

use crate::buffers::SampleBuffer;

pub type Int = i32; // sample type to read wav samples into
pub type Float: = f32; // sample type (casted from Int) used for internal processing
pub type CFloat = Complex<Float>;

/// print summary stats of a sample buffer
fn chunk_summary(buf: &SampleBuffer<Float>) {
    let chunk1 = buf.data();
    let rms = chunk1.iter()
        .fold(0.0_f64, |acc, &x| acc + (x*x) as f64);
    let rms = (rms as f64 / chunk1.len() as f64).sqrt();
    println!("BUFFER INFO:");
    println!("> buffer size: {}", chunk1.len());
    println!("> rms = {}", rms);
    println!("> min = {}", chunk1.iter().fold(Float::INFINITY, |a, &b| a.min(b)));
    println!("> max = {}", chunk1.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b)));
}

/// Root-mean-square average of a vector
pub fn rms<T,R>(vals: &[T]) -> R
where T:Num+AsPrimitive<R>, R:'static+num_traits::Float {
    let e = energy(vals).as_();
    e.sqrt()
}

/// Signal energy
pub fn energy<T:Num+Copy>(vals: &[T]) -> T {
    vals.iter()
        .fold(T::zero(), |acc, &x| acc + x*x)
}

/// Convert degrees to radians
pub fn deg2rad<R>(deg: R) -> R
where R:'static+num_traits::Float+From<Float> {
    deg * (PI / 180.0).into()
}

/// convert radians to degrees
pub fn rad2deg<R>(rad: R) -> R
where R:'static+num_traits::Float+From<Float> {
    rad * (180.0 / PI).into()
}

/// convert hz to rad/s
pub fn hz2rads<R>(hz: R) -> R
where R:'static+num_traits::Float+From<Float> {
    hz * TAU.into()
}

/// Multiply a vector by a scalar
pub fn vec_scale<T>(v: &[T], k: T) -> Vec<T>
where T: Copy+Mul<T, Output=T> {
    v.iter()
        .map(|&x| x*k)
        .collect::<Vec<_>>()
}

/// Multiple a vector by a scalar (in place)
pub fn vec_scale_inplace<T>(v: &mut [T], k: T)
where T: Copy+MulAssign<T> {
    for x in v.iter_mut() {
        *x *= k;
    }
}

/// Add 2 vectors together
/// Output vec length is the max length of either input 
pub fn vec_add<T>(v1: &Vec<T>, v2: &Vec<T>) -> Vec<T>
where T: Add<Output=T> + Copy {
    v1.iter().zip_longest(v2.iter())
        .map(|it| match it {
            EitherOrBoth::Both(a,b) => *a + *b,
            EitherOrBoth::Left(a) => *a,
            EitherOrBoth::Right(b) => *b
        })
        .collect::<Vec<T>>()
}

/// Multiply 2 polynomials (represented as vectors) together
///
/// BORING MATH:
/// To multiply polynomials, multiply coefficient pairs between v1 and v2. 
/// v_out[k] = v1[0]*v2[k] + v1[1]*v2[k-1] + ... + v1[k]*v2[0]
pub fn vec_mul<T>(v1: &Vec<T>, v2: &Vec<T>) -> Vec<T>
where T: AddAssign + Mul<Output=T> + Zero + Copy {
    let mut x1: &T;
    let mut x2: &T;
    let mut vout = vec![T::zero(); v1.len() + v2.len()];
    for i1 in 0..v1.len() {
        x1 = &v1[i1];
        for i2 in 0..v2.len() {
            x2 = &v2[i2];
            vout[i1+i2] += *x1 * *x2;
        }
    }
    vout
}

/// Sinc interpolation
///
/// BORING MATH:
///   x(t) = sum{n: -inf->inf}( x[n] * sinc((t - nT)/T) )
/// substitute t=m*T2:
///   x[m*T2] = sum{n}( x[n] * sinc((mT2 - nT) / T) )
///           = sum{n}( x[n] * sinc(m(T2/T) - n) )
pub fn interp_sinc<T>(v_in: &[T], n_out: usize) -> Vec<T>
where
    T: 'static + num_traits::Float + AsPrimitive<f64>,
    for<'a> &'a T: Mul<T, Output=T>,
    f64: AsPrimitive<T>
{
    let n_in = v_in.len();
    let t2_t1 = (n_in - 1) as f64 / (n_out - 1) as f64; // ratio of T2/T1

    (0..n_out)
        .map(|m| {
            repeat(m).zip(v_in.iter().enumerate())
            .map(|(m,(n,xn))| xn.as_() * sinc::<f64>(m as f64 * t2_t1 - n as f64))
            .sum::<f64>()
            .as_()
    }).collect()
}

/// normalized sinc function sin(pi x) / (pi x)
pub fn sinc<T>(x: T) -> T
where T: 'static+num_traits::Float, f32: AsPrimitive<T> {
    if x == 0.0.as_() {
        1.0.as_()
    } else {
        let pi_x = x * PI.as_();
        pi_x.sin() / pi_x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_energy() {
        let v = vec![-1.4, 0.0, 2.6, 4.5];
        assert_eq!(energy(&Vec::<Float>::new()), 0.0);
        assert_relative_eq!(energy(&v), 28.97);
    }

    #[test]
    fn test_rms() {
        let v = vec![-1.4, 0.0, 2.6, 4.5];
        assert_eq!(rms::<_,Float>(&Vec::<Float>::new()), 0.0);
        assert_relative_eq!(rms::<_,f64>(&v), 5.382378656319156_f64);
    }

    #[test]
    fn test_deg2rad() {
        assert_eq!(deg2rad(0.0), 0.0);
        assert_eq!(deg2rad(90.0), 0.5 * PI);
        assert_eq!(deg2rad(180.0), PI);
        assert_eq!(deg2rad(-90.0), -0.5 * PI);
    }

    #[test]
    fn test_rad2deg() {
        assert_eq!(rad2deg(0.0), 0.0);
        assert_eq!(rad2deg(0.5 * PI), 90.0);
        assert_eq!(rad2deg(PI), 180.0);
        assert_eq!(rad2deg(-0.5 * PI), -90.0);
    }

    #[test]
    fn test_hz2rads() {
        assert_eq!(hz2rads(0.0), 0.0);
        assert_eq!(hz2rads(0.5), PI);
        assert_eq!(hz2rads(1.0), 2.0 * PI);
        assert_eq!(hz2rads(-1.0), -2.0 * PI);
    }

    #[test]
    fn test_vec_scale() {
        let v = vec![11.19, -9.23, -11.34, -4.72];
        assert_relative_eq!(
            vec_scale(&v[..], 2.0)[..],
            vec![22.38, -18.46, -22.68, -9.44]);
        assert_relative_eq!(
            vec_scale(&v[..], -0.5)[..],
            vec![-5.595, 4.615, 5.67, 2.36]);
    }

    #[test]
    fn test_vec_scale_inplace() {
        let v1 = vec![11.19, -9.23, -11.34, -4.72];
        let mut v2 = v1.clone();
        vec_scale_inplace(&mut v2[..], 2.0);
        assert_relative_eq!(v2[..], vec![22.38, -18.46, -22.68, -9.44]);
        vec_scale_inplace(&mut v2[..], 0.5);
        assert_relative_eq!(v2[..], v1[..]);
    }

    #[test]
    fn test_vec_add() {
        let v1 = vec![-8.24, -6.36, 9.53, 4.39];
        let v2 = vec![5.45, -6.68, 8.46, -11.54];
        let v3 = vec![-8.39, 3.56, 16.68];
        let v4 = vec![-7.17, -10.99, -2.48, -12.46, 0.17];

        // same length
        assert_relative_eq!(
            vec_add(&v1, &v2)[..],
            vec![-2.79, -13.04, 17.99, -7.15]);
        // RHS is shorter
        assert_relative_eq!(
            vec_add(&v2, &v3)[..],
            vec![-2.94, -3.12, 25.14, -11.54]);
        // RHS is longer
        assert_relative_eq!(
            vec_add(&v2, &v4)[..],
            vec![-1.72, -17.67, 5.98, -24.0, 0.17]);
    }
}