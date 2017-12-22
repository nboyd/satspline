#![allow(non_snake_case)]
#![allow(unused_must_use)]

extern crate rblas;
mod nonnan;
mod wgam;
use wgam::*;
use std::slice;
mod loss;
mod gam;
use gam ::*;
use loss::*;

#[no_mangle]
pub unsafe extern fn make_training_dataset(a : *mut*mut f64, n : usize, d: usize) -> Box<TrainingData> {
    let mut A = Vec::with_capacity(n);
    for idx in 0..n{
        A.push((slice::from_raw_parts(*(a.offset(idx as isize)), d)).iter().cloned().collect::<Vec<_>>());
    }
    Box::new(TrainingData::new(A))
}

#[no_mangle]
pub unsafe extern fn make_gam(d : usize) -> Box<GAM> {
    Box::new(GAM::new(d))
}

#[no_mangle]
pub unsafe extern fn num_knots(gam : &GAM) -> usize {
    gam.splines.iter().map(|s| s.weights.len()).sum()
}

#[no_mangle]
pub unsafe extern fn offset(gam : &GAM) -> f64 {
    gam.offset
}

#[no_mangle]
pub unsafe extern fn get_knots(gam : &GAM, knot_idx : *mut usize, t : *mut f64, w : *mut f64)  {
    let k = gam.splines.iter().map(|s| s.weights.len()).sum();
    let mut knot_idx = slice::from_raw_parts_mut(knot_idx,k);
    let mut t = slice::from_raw_parts_mut(t,k);
    let mut w = slice::from_raw_parts_mut(w,k);
    let mut idx = 0;
    for f_idx in 0..gam.splines.len(){
        for j_idx in 0..gam.splines[f_idx].locations.len(){
            t[idx] = gam.splines[f_idx].locations[j_idx];
            w[idx] = gam.splines[f_idx].weights[j_idx];
            knot_idx[idx] = f_idx;
            idx += 1;
        }
    }

}

#[no_mangle]
pub unsafe extern fn fit_gam(gam : &mut GAM, td : &TrainingData, y_ptr : * mut f64,  lambda : f64, tol : f64, use_offset : bool, loss : &LossFn) -> f64
{
    let y = slice::from_raw_parts(y_ptr, td.training_examples.len());
    match *loss{
        LossFn::Squared => inf_cg(gam, td, y, lambda, tol, use_offset, &sq_loss),
        LossFn::Logistic => inf_cg(gam, td, y, lambda, tol, use_offset, &log_loss),
        LossFn::Huber(delta) => inf_cg(gam, td, y, lambda, tol, use_offset, &(|o,z| pseudo_huber(o,z,delta))),
    }
}

#[no_mangle]
pub unsafe extern fn apply_gam(g : &GAM, v : *mut f64, d: usize) -> f64  {
    // construct slice
    let slice = slice::from_raw_parts(v, d);
    //apply gam
    g.apply(slice)
}
//this seems... wrong.
#[no_mangle]
pub fn return_log_loss() -> Box<LossFn> {
    Box::new(LossFn::Logistic)
}
#[no_mangle]
pub fn return_sq_loss() -> Box<LossFn> {
    Box::new(LossFn::Squared)
}
#[no_mangle]
pub fn return_huber_loss(delta : f64) -> Box<LossFn> {
    Box::new(LossFn::Huber(delta))
}
