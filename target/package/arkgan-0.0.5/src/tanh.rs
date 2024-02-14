pub fn calc_tanh(x: &[f64]) -> Vec<f64>{
    let mut tanh_values: Vec<f64> = vec![];
    for &elements in x{
        let nom: f64 = f64::exp(elements) - f64::exp(-elements) as f64;
        let denom: f64 = f64::exp(elements) + f64::exp(-elements) as f64;
        let calc: f64 = nom / denom as f64;
        tanh_values.push(calc);
    }
    return tanh_values;
}
