use std::cmp::Ordering;

#[derive(PartialEq,PartialOrd)]
pub struct NonNan(pub f64);

impl NonNan {
    pub fn new(val: f64) -> NonNan {
        if val.is_nan() {
            panic!("NaN value!");
        } else {
            NonNan(val)
        }
    }
}

impl Eq for NonNan {}

impl Ord for NonNan {
    fn cmp(&self, other: &NonNan) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
