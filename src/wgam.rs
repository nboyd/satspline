use rblas::math::Mat;
use rblas::Dot;
use rblas::matrix::Matrix;
use rblas::attribute::*;
use rblas::matrix::ll::cblas_d;
use rblas::matrix_vector::ops::Gemv;
use std::f64;
use nonnan::*;
use gam::*;

pub struct TrainingData {
    d : usize,
    pub training_examples : Vec<Vec<f64>>,
    sort_vecs : Vec<Vec<(usize,f64)>>
}

impl TrainingData {
    pub fn new(t_data : Vec<Vec<f64>>) -> TrainingData {
        let d = t_data[0].len();
        let mut s = (0..d).map(|i| t_data.iter().map(|x| x[i]).enumerate().collect::<Vec<(usize,f64)>>()).collect::<Vec<_>>();
        for v in &mut s{
            v.sort_by(|&a,&b| a.1.partial_cmp(&b.1).expect("Invalid training data!"));
        }
        TrainingData{ d: d, training_examples : t_data, sort_vecs: s}
    }

    pub fn apply(&self, g : &GAM, o_vec : &mut[f64]) {
        for (o,v) in o_vec.iter_mut().zip(self.training_examples.iter()){
            *o = g.apply(&v);
        }
    }
}

fn apply_loss<L>(o : &[f64], y : &[f64], d : &mut[f64], dd : &mut[f64], loss : &L) -> f64
where L: Fn(f64,f64) -> (f64,f64,f64) {
    let mut r = 0.0;
    for idx in 0..o.len(){
        let (l_i,first_derivative_i,second_derivative_i) = loss(o[idx],y[idx]);
        r += l_i;
        d[idx] = first_derivative_i;
        dd[idx] = second_derivative_i;
    }
    r
}

pub fn inf_cg<L>(gam : &mut GAM, td : &TrainingData, y : &[f64],  lambda : f64, tol : f64, use_offset : bool, loss : &L) -> f64
where L: Fn(f64,f64) -> (f64,f64,f64) {
    let d = td.d;
    assert!(d == gam.splines.len(),"GAM and training data have different d!");
    let n = td.training_examples.len();
    // allocate vectors to store output, derivative and second derivative
    let (mut o, mut first_derivative_loss, mut dd_loss) = (vec![0.0;n],vec![0.0;n],vec![0.0;n]);
    // fit the offset term
    optimize_weights(gam, &td, &y, loss, lambda, tol, use_offset);
    loop {
        // compute output on training data.
        td.apply(&gam,&mut o);
        // compute loss and derivatives
        apply_loss(&o,&y,&mut first_derivative_loss, &mut dd_loss, loss);
        // get best knots for each feature
        let potential_knots = td.sort_vecs.iter().map(|sv| Spline::min_max_knots(sv, &first_derivative_loss)).enumerate();
        // best knots
        let best_knots = potential_knots.min_by_key(|p| NonNan::new((p.1).1+(p.1).3)).unwrap();
        // check optimality conditions
        let bound = Dot::dot(&o, &first_derivative_loss) - (lambda/2.0)*((best_knots.1).1 + (best_knots.1).3);
        if bound <= tol{
            return (best_knots.1).1 + (best_knots.1).3;
        }
        // println!("BOUND: {}", bound);
        // add knots
        gam.splines[best_knots.0].add_knot((best_knots.1).0);
        gam.splines[best_knots.0].add_knot((best_knots.1).2);
        // update weights
        optimize_weights(gam, &td, &y, loss, lambda, tol, use_offset);
    }
}

fn optimize_weights<L>(gam : &mut GAM, td : &TrainingData, y : &[f64], loss : &L, lambda : f64, tol : f64, use_offset : bool)
where L : Fn(f64,f64) -> (f64,f64,f64) {
    let n = td.training_examples.len();
    // check if no knots...
    let num_knots = gam.splines.iter().map(|s| s.weights.len()).sum::<usize>();
    if num_knots == 0 {
        if use_offset{
            let mut o = vec![0.0; n];
            let mut first_derivative_loss = vec![0.0; n];
            let mut dd_loss = vec![0.0; n];
            apply_loss(&o,&y,&mut first_derivative_loss, &mut dd_loss, loss);
            gam.offset = find_offset(& mut o, y, loss,& mut first_derivative_loss, & mut dd_loss , gam.offset)
        }
        return;
    }
    // linearize knots... (worth it?)
    // yes.
    let mut knot_locations = vec![0.0;num_knots];
    let mut w = vec![0.0;num_knots];
    let mut offset = gam.offset;
    let mut knot_fidx = vec![0;num_knots];
    let mut idx = 0;
    for f_idx in 0..gam.splines.len(){
        for j_idx in 0..gam.splines[f_idx].locations.len(){
            knot_locations[idx] = gam.splines[f_idx].locations[j_idx];
            w[idx] = gam.splines[f_idx].weights[j_idx];
            knot_fidx[idx] = f_idx;
            idx += 1;
        }
    }

    let mut groups = vec![(0,0);0];
    idx = 0;
    for f_idx in 0..gam.splines.len(){
        groups.push((idx, idx+gam.splines[f_idx].locations.len()));
        idx = idx+gam.splines[f_idx].locations.len();
    }

    let mut O : Mat<f64> = Mat::new(n,num_knots);
    let mut O_p : Mat<f64> = Mat::new(n,num_knots);
    // fill O
    unsafe{
        for t_idx in 0..n {
            let x = &td.training_examples[t_idx];
            let ptr = O.as_mut_ptr().offset((t_idx*num_knots) as isize);
            for knot_idx in 0..num_knots{
                let f_idx = knot_fidx[knot_idx];
                let v = x[f_idx] - knot_locations[knot_idx];
                *(ptr.offset(knot_idx as isize)) = if v >= 0.0 {v} else {0.0};
            }
        }
    }
    let mut g = vec![0.0; num_knots];
    let mut o = vec![0.0; n];
    let mut first_derivative_loss = vec![0.0; n];
    let mut dd_loss = vec![0.0; n];
    //proximal-newton loop
    for _ in 1..10 {
        // println!("prox-newton!");
        // compute output at current weights
        f64::gemv(Transpose::NoTrans, &1.0, &O, &w, &0.0, &mut o);
        //add offset...
        if use_offset {
            for v in &mut o{
                (*v) += offset;
            }
        }
        // compute loss and derivatives
        apply_loss(&o,&y,&mut first_derivative_loss, &mut dd_loss, loss);
        // calculate gradient...
        f64::gemv(Transpose::Trans, &1.0, &O,&first_derivative_loss, &0.0, &mut g);
        //check optimality conditions...
        let score = {
            let up_scores = groups.iter()
                        .map(|&(start_idx, end_idx)| (start_idx..end_idx)
                                .map(|i| g[i])
                                .min_by_key(|v| NonNan::new(*v)).unwrap_or(0.0)
                            );
            let down_scores = groups.iter()
                        .map(|&(start_idx, end_idx)| (start_idx..end_idx)
                                .map(|i| -g[i])
                                .min_by_key(|v| NonNan::new(*v)).unwrap_or(0.0)
                            );
            up_scores.zip(down_scores).map(|(u,d)| u+d).min_by_key(|v| NonNan::new(*v)).unwrap()
        };
        let b_o : f64 = first_derivative_loss.iter().sum();
        let bound = Dot::dot(&first_derivative_loss,&o) - (lambda/2.0)*score;
        // println!("prox_bound: {}", bound);
        if bound  <= tol/2.0 {
            break;
        }

        // collect offset terms
        let H_cc = dd_loss.iter().sum::<f64>();
        let mut C = vec![0.0;num_knots];
        f64::gemv(Transpose::Trans, &1.0, &O,&dd_loss, &0.0, &mut C);

        // calculate hessian
        // compute sqrt dd_loss
        for idx in 0..n{
            dd_loss[idx] = dd_loss[idx].sqrt();
        }
        //scale O to get square-root of hessian
        unsafe{
            for t_idx in 0..n{
                let ptr_o = O.as_ptr().offset((t_idx*num_knots) as isize);
                let ptr_o_p = O_p.as_mut_ptr().offset((t_idx*num_knots) as isize);
                for knot_idx in 0..num_knots{
                    *(ptr_o_p.offset(knot_idx as isize)) = *(ptr_o.offset(knot_idx as isize))*dd_loss[t_idx];
                }
            }
        }
        let mut H : Mat<f64> = Mat::new(num_knots,num_knots);
        unsafe{
            cblas_d::syrk(O_p.order(), Symmetry::Upper, Transpose::Trans, num_knots as i32, n as i32, 1.0, O_p.as_ptr(), num_knots as i32, 0.0, H.as_mut_ptr(), num_knots as i32);
        }

        let vtw = Dot::dot(&C, &w);
        //lazy hack
        let mut H_p = vec![vec![0.0; num_knots]; num_knots];
        for idx in 0..num_knots{
                H_p[idx][idx] = H[idx][idx];
            for ip in 0..idx{
                H_p[idx][ip] = H[ip][idx];
                H_p[ip][idx] = H[ip][idx];
            }
        }
        let mut H = H_p;
        let mut b = vec![0.0; num_knots];
        for knot_idx in 0..num_knots{
            for knot_idx_prime in 0..num_knots{
                b[knot_idx] += H[knot_idx][knot_idx_prime]*w[knot_idx_prime];
            }
            b[knot_idx] -= g[knot_idx];
        }

        //use_offset => do partial minimization...
        let b_o = -b_o + vtw + H_cc*offset;

        if use_offset {
            // modify H and b... schur complement bro
            for i in 0..num_knots{
                b[i] += C[i]*offset;
            }
            for i in 0..num_knots{
                for j in 0..num_knots{
                    H[i][j] -= (1.0/H_cc)*C[i]*C[j];
                }
            }
            let mul = -(1.0/H_cc)*b_o;
            for i in 0..num_knots{
                b[i] += C[i]*mul;
            }
        }
        // quadratic objective
        min_quad(&mut w, &groups, &b, &H, lambda, tol);
        if use_offset {
            let mut btx_star = 0.0;
            for i in 0..num_knots{
                btx_star += C[i]*w[i];
            }
            offset = (1.0/H_cc)*(b_o - btx_star);
        }
    }
    //write new weights... and filter out zero-weight knots.
    let mut idx = 0;
    for f_idx in 0..gam.splines.len(){
            gam.splines[f_idx].weights = Vec::with_capacity(gam.splines[f_idx].weights.len());
            let old_locations = gam.splines[f_idx].locations.clone();
            gam.splines[f_idx].locations = Vec::with_capacity(old_locations.len());
        for j_idx in 0..old_locations.len(){
            if w[idx].abs() >= 1E-5{
                gam.splines[f_idx].weights.push(w[idx]);
                gam.splines[f_idx].locations.push(old_locations[j_idx]);
            }
            idx += 1;
        }
    }
    gam.offset = offset;
}

fn min_quad(w: &mut[f64], groups : &[(usize,usize)], b : &[f64], K : &[Vec<f64>], lambda : f64, tol : f64) {
    // set K*w
    let num_knots = w.len();
    let mut K_w = vec![0.0; num_knots];
    for i in 0..num_knots{
        for j in 0..num_knots{
            K_w[i] += K[i][j]*w[j];
        }
    }
    let mut iter_count = 0;
    for _ in 0..1000000 {
          iter_count += 1;
          let mut score = f64::INFINITY;
          let mut w_dot_g = 0.0;
          let mut w_K_w = 0.0;
          let mut b_w = 0.0;
          let mut i_all = 0;
          let mut j_all = 0;
          for &(start_idx, stop_idx) in groups.iter(){
              let mut i = 0;
              let mut j = 0;
              let mut v_i = f64::INFINITY;
              let mut v_j = f64::INFINITY;
              for idx in start_idx..stop_idx {
                  b_w += b[idx]*w[idx];
                  let g = K_w[idx]-b[idx];
                  w_dot_g += g*w[idx];
                  w_K_w += K_w[idx]*w[idx];
                  let v_idx_i = g;
                  let v_idx_j = -g;
                  if v_idx_i <= v_i {
                      v_i = v_idx_i;
                      i = idx;
                  }
                  if v_idx_j <= v_j {
                      v_j = v_idx_j;
                      j = idx;
                  }
              }
              if v_i + v_j <= score {
                  score = v_i + v_j;
                  i_all = i;
                  j_all = j;
              }
          }
          let bound = w_dot_g-(lambda/2.0)*score;
          if  bound < tol/2.0 {
              if iter_count >= 100000 {
                  println!("iters: {}", iter_count);
              }
              break;
          }
          // d = (lambda/2.0)*(e_i_all - e_j_all)
          let d_K_d = ((lambda*lambda)/4.0)*(K[i_all][i_all] -2.0*(K[i_all][j_all]) + K[j_all][j_all]);
          let w_K_d = (lambda/2.0)*(K_w[i_all] - K_w[j_all]);
          let b_d = (lambda/2.0)*(b[i_all] - b[j_all]);
          let alpha = min_1d_quad_interval(d_K_d + w_K_w - 2.0*w_K_d,-( b_d - b_w - w_K_d + w_K_w));
          // take step...
          for idx in 0..num_knots{
              K_w[idx] *= 1.0 - alpha;
              K_w[idx] += alpha*(lambda/2.0)*K[i_all][idx];
              K_w[idx] -= alpha*(lambda/2.0)*K[j_all][idx];
              w[idx] *= 1.0 - alpha;
          }
          w[i_all] += (lambda/2.0)*alpha;
          w[j_all] -= (lambda/2.0)*alpha;
    }
}

fn find_offset<L>(output : &mut [f64], y : &[f64], loss : &L, d_loss : &mut[f64], dd_loss: &mut[f64], offset_init: f64) -> f64
where L: Fn(f64,f64) -> (f64,f64,f64){
    let mut offset = offset_init;
    loop {
        //compute gradients
        apply_loss(&output,&y,d_loss, dd_loss, loss);
        let grad_o = d_loss.iter().sum::<f64>();
        if grad_o.abs() <= 1E-3 {
            break;
        }
        let second_grad_o = dd_loss.iter().sum::<f64>();
        let step = grad_o/second_grad_o;
        offset = offset - step;
        for v in output.iter_mut(){
            (*v) -= step;
        }
    }
    offset
}

/// Solves `min_x`
/// 0.5*a*x^2 + b*x
/// s.t. 0 <= x <= 1
/// a must be >= 0.
fn min_1d_quad_interval(a : f64, b : f64) -> f64{
  let unconstrained_min = -b/a;
  unconstrained_min.min(1.0).max(0.0)
}
