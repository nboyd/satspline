
fn sign(v : f64) -> f64 {
    if v < 0.0{
        -1.0
    } else if v == 0.0 {
        0.0
    } else {
        1.0
    }
}

fn abs(v : f64) -> f64 {
    if v <= 0.0 {
        -v
    } else {
        v
    }
}
/// Solves the convex optimization problem:
/// min_x 0.5*x'Kx - b'x + lambda |x|_1
/// terimates when the directional derivative in l_1 norm is <= tol in abs value
pub fn lasso(k : &[&[f64]], b : &[f64], lambda : f64, tol : f64, o : &mut[f64], x : &mut[f64] ){
    let n = b.len();
    // o = k*x
    loop {
        // compute directional derivative
        let mut min_ind = 0;
        let mut min_dd = 0.0;
        for ind in 0..n {
            let g = o[ind] - b[ind];
            let dd = if x[ind] == 0.0{
                 g.abs() - lambda
             } else if x[ind] > 0.0{
                 (g + lambda).abs()
             } else {
                 (g - lambda).abs()
             };
            if dd > min_dd{
                min_dd = dd;
                min_ind = ind;
            }
        }
        // println!("min_ind = {}", min_ind);
        // println!("dd = {}", min_dd);
        if min_dd < tol {
            return;
        }
        // do a step of coordinate descent for min_ind
        let x_old = x[min_ind];
        let quad_term = k[min_ind][min_ind];
        let lin_term = -(o[min_ind] - quad_term*x_old - b[min_ind]);
        //soft thresholding...
        let x_new = if lin_term <= lambda && lin_term >= -lambda {0.0}
            else {(lin_term - sign(lin_term)*lambda)/quad_term};
        //update o
        let delta = x_new - x_old;
        for (o_v, k_v) in o.iter_mut().zip(k[min_ind].iter()) {
            *o_v += delta*(*k_v);
        };
        //update x
        x[min_ind] = x_new;
        // println!("o = {:?}", o);
        // println!("x = {:?}", x);
    }
    println!("did not converge!");
    return;
}
