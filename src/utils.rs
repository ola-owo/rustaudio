#![allow(dead_code)]
use std::f32::consts::{PI, TAU};
use std::iter::repeat;
use std::ops::{Add, AddAssign, Mul};

use itertools::{EitherOrBoth, Itertools};
use rustfft::num_complex::Complex;
use num_traits::{AsPrimitive, Num, Zero};

pub type Float: = f32; // type used for internal processing
pub type Int = i16; // type of samples in wav file
pub type CFloat = Complex<Float>;

/////////////////////////////
// Random helper functions //
/////////////////////////////

/* Root-mean-square average of a vector
*
* bound on T means that T is castable to R
* bound on R means that R is a Float
*/
pub fn rms<T,R>(vec: &Vec<T>) -> R
where T:Num+AsPrimitive<R>, R:'static+num_traits::Float {
    let e = energy(vec).as_();
    e.sqrt()
}

/* Signal energy */
pub fn energy<T:Num+Copy>(vec: &Vec<T>) -> T {
    vec
        .iter()
        .fold(T::zero(), |acc, &x| acc + x*x)
}

/* Convert degrees to radians */
pub fn deg2rad<R>(deg: R) -> R
where R:'static+num_traits::Float+From<Float> {
    deg * (PI / 180.0).into()
}

/* convert radians to degrees */
pub fn rad2deg<R>(rad: R) -> R
where R:'static+num_traits::Float+From<Float> {
    rad * (180.0 / PI).into()
}

/* convert hz to rad/s */
pub fn hz2rads<R>(hz: R) -> R
where R:'static+num_traits::Float+From<Float> {
    hz * TAU.into()
}

/* Add 2 vectors together
 * Output vec length is the max length of either input 
 */
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

/* Multiply 2 polynomials (represented as vectors) together
 *
 * BORING MATH:
 * To multiply polynomials, multiply coefficient pairs between v1 and v2. 
 * v_out[k] = v1[0]*v2[k] + v1[1]*v2[k-1] + ... + v1[k]*v2[0]
 */
pub fn vec_mul<T>(v1: &Vec<T>, v2: &Vec<T>) -> Vec<T>
where T: AddAssign + Mul<Output=T> + Zero + Copy {
    let mut x1: &T;
    let mut x2: &T;
    let mut y: &mut T;
    let mut vout = vec![T::zero(); v1.len() + v2.len()];
    for i1 in 0..v1.len() {
        x1 = v1.get(i1).unwrap();
        for i2 in 0..v2.len() {
            x2 = v2.get(i2).unwrap();

            y = vout.get_mut(i1+i2).unwrap();
            *y += *x1 * *x2;
        }
    }
    vout
}

/* Sinc interpolation
 *
 * BORING MATH
 * x(t) = sum_{n: -inf->inf} [x[n] * sinc((t - nT)/T)]
 * substitute t=mT2:
 * x[mT2] = sum_n{ x[n] * sinc((mT2 - nT) / T) }
 *        = sum_n{ x[n] * sinc(m(T2/T) - n) }
 */
pub fn interp_sinc<T>(v_in: &[T], n_out: usize) -> Vec<T>
where T: 'static + num_traits::Float + AsPrimitive<f64>,
for<'a> &'a T: Mul<T, Output=T>,
f64: AsPrimitive<T> {
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

// normalized sinc function sin(pi x) / (pi x)
pub fn sinc<T>(x: T) -> T
where T: 'static+num_traits::Float, f32: AsPrimitive<T> {
    if x == 0.0.as_() {
        1.0.as_()
    } else {
        let pix = x * PI.as_();
        pix.sin() / pix
    }
}