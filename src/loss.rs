#[link_name = "m"]
   extern {
       pub fn log1p(n: f64) -> f64;
   }

pub fn ln1p(n : f64) -> f64 {
    unsafe {
        log1p(n)
    }
}

#[derive(Debug)]
pub enum LossFn {
    Squared,
    Logistic,
    Huber(f64),
}

pub fn sq_loss(o : f64, y : f64) -> (f64,f64,f64){
    (0.5*(o-y)*(o-y), o-y, 1.0)
}

pub fn log_loss(o : f64, y : f64) -> (f64,f64,f64){
        let o = o*y;
    if o > 0.0 {
        let o_exp = (-o).exp();
        let logit = 1.0/(1.0 + o_exp);
        (ln1p(o_exp), y*(logit-1.0), logit*(1.0-logit))
    } else {
        let o_exp = o.exp();
        let logit = o_exp/(1.0 + o_exp);
        (-o +ln1p(o_exp), y*(logit-1.0), logit*(1.0-logit))
    }
}
/// `L_delta(e) = delta*(sqrt(1+x^2/delta) -1)`
pub fn pseudo_huber(o : f64, y : f64, delta : f64) -> (f64,f64,f64) {
    let x = o -y ;
    let sqrt_term = (1.0 + x*x/delta).sqrt();
    let denom_sqrt = delta+x*x;
    (delta*(sqrt_term -1.0), x/sqrt_term, delta*delta*sqrt_term/(denom_sqrt*denom_sqrt))
}
