pub fn calc_relu(x: &[f64]) -> Vec<f64>{
    let mut relu_val: Vec<f64> = vec![];
    for &elements in x{
        if elements > 0.0{
            relu_val.push(elements);
        }else{
            relu_val.push(0.0);
        }
    }
    return relu_val;
}