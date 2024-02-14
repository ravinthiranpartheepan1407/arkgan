pub fn calc_sigmoid(x: &[f64]) -> Vec<f64>{
    let mut sigmoid_arr: Vec<f64> = vec![];
    for &elements in x{
        let constant: f64 = 1.0;
        let calc = ((constant) / (constant + f64::exp(-elements))) as f64;
        sigmoid_arr.push(calc);
    }
    return sigmoid_arr;
}