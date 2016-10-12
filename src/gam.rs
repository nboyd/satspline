use std::cmp::Ordering;
use std::f64;
use std::fmt;

fn cmp_f64(a : &f64, b : &f64) -> Ordering {a.partial_cmp(b).expect("NaN!")}

#[derive(Debug, Clone, Default)]
pub struct Spline {
    pub locations : Vec<f64>,
    pub weights : Vec<f64>,
}

impl Spline {
    pub fn new() -> Spline {
        Default::default()
    }

    pub fn apply(&self, x : f64) -> f64 {
        let mut r = 0.0;
        for (w,l) in self.weights.iter().zip(self.locations.iter()){
            r += w*((x - l).max(0.0));
        }
        r
    }

    pub fn add_knot(&mut self, knot_location : f64) {
        match self.locations.binary_search_by(|v| cmp_f64(v,&knot_location)) {
            Ok(_) => {},
            Err(i) => {self.locations.insert(i,knot_location);
                        self.weights.insert(i,0.0);},
        };
    }

    pub fn min_max_knots(x_sp : &Vec<(usize,f64)>, v : &[f64]) -> (f64,f64,f64,f64) {
        let mut down_score = 0.0;
        let mut down_t = 1.0;
        let mut up_score = 0.0;
        let mut up_t = 1.0;
        let mut v_cs = 0.0;
        let mut vx_cs = 0.0;
        let mut last_x = 1.0;
        let n = v.len();
        // Bugs. Undoubtedly.
        for i in (0..n).rev() {
            let x = x_sp[i].1;
            assert!(0.0 <= x && x <= 1.0, "Training data must lie in [0,1]!");
            if last_x == x {
                v_cs += v[x_sp[i].0];
                vx_cs += x*v[x_sp[i].0];
                continue;
            }
            //find best t in interval (x, last_x) #### NOOOO.
            let t = if v_cs >= 0.0 {last_x} else {x};
            let score = vx_cs - t*v_cs;
            if score < down_score{
                down_score = score;
                down_t = t;
            }
            //find best t in interval (x, last_x)
            let t = if v_cs <= 0.0 {last_x} else {x};
            let score = vx_cs - t*v_cs;
            if -score < up_score{
                up_score = -score;
                up_t = t;
            }

            last_x = x;
            v_cs += v[x_sp[i].0];
            vx_cs += x*v[x_sp[i].0];
        }
        // check t = 0.0 .... (might not be in the data...)
        if vx_cs <= down_score{
            down_score = vx_cs;
            down_t = 0.0;
        }
        if -vx_cs <= up_score{
            up_score = -vx_cs;
            up_t = 0.0;
        }
        (up_t, up_score,down_t,down_score)
    }
}

#[derive(Debug,Clone)]
pub struct GAM {
    pub splines : Vec<Spline>,
    pub offset : f64
}

impl GAM {
    pub fn apply(&self, x : &[f64]) -> f64 {
        assert!(x.len() == self.splines.len(),
            format!("GAM for d={} used on input of dimension {}",self.splines.len(), x.len()));
        let mut r = 0.0;
        for (v, spline) in x.iter().zip(self.splines.iter()){
            r += spline.apply(*v)
        }
        r + self.offset
    }

    pub fn new(d : usize) -> GAM {
        GAM{ splines : (0..d).map(|_| Spline::new()).collect::<Vec<Spline>>(), offset : 0.0}
    }
}

impl fmt::Display for GAM {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GAM:\n");
        for (idx,spline) in self.splines.iter().enumerate(){
            write!(f, "\t {} \t", idx);
            spline.fmt(f);
            write!(f, "\n");
        }
        write!(f, "\toffset = {}", self.offset)
    }
}

impl fmt::Display for Spline {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Spline[");
        for (l,w) in self.locations.iter().zip(self.weights.iter()){
            write!(f, "({}, {}), ", l, w);
        }
        write!(f, "]")
    }
}
