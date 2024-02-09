// #[macro_use]
// extern create cpython

// use cpython::{Python, PyResult};

const values: [f64;5] = [10.0,20.0,30.0,40.0,50.0];

pub fn mean(numbers: &[f64]) -> f64{
    let mut sum: f64 = 0.0;
    for &elements in numbers{
        sum += elements;
    }
    return sum / numbers.len() as f64;
}

fn main(){
    let res: f64 = mean(&values);
    println!("Mean: {}",res);
}