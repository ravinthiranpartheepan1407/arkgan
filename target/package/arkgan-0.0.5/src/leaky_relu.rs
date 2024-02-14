pub fn calc_leaky_relu(x: &[f64]) -> Vec<f64>{
    let neg_slope: f64 = 0.01;
    let mut leaky_relu_val: Vec<f64> = vec![];
    for &elements in x{
        if elements >= 0.0{
            leaky_relu_val.push(elements);
        }else{
            // println!("Negative Slope: {}", neg_slope);
            let calc = neg_slope * elements;
            leaky_relu_val.push(calc);
        }
    }
    return leaky_relu_val;
}