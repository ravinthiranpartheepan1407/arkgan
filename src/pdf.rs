pub fn calc_pdf(values: &[f64]) -> Vec<f64>{
    // Calculate Mean
    let mut sum: f64 = 0.0;
    let mut sqr_diff: f64 = 0.0;
    let base: f64 = 2.0;
    let mut exp_val: Vec<f64> = vec![];
    let mut out: Vec<f64> = vec![];
    for &elements in values{
        sum += elements;
    }
    let mean: f64 = sum / (values.len() as f64);
    // Calculate Standard Deviation
    for &elements in values{
        let diff: f64 = elements - sum;
        let sqr_inst: f64 = diff.powi(2);
        sqr_diff += sqr_inst;
    }
    let std: f64 = sqr_diff.sqrt() / (values.len() as f64);
    for &elements in values{
        let calc:f64 = elements - mean;
        let sqr_cal: f64 = calc.powi(2);
        let div_std: f64 = sqr_cal / (&base * (std.powi(2)));
        let exp_cal: f64 = base.powf(-(div_std));
        exp_val.push(exp_cal);
    }
    let constant: f64 = 1.0;
    let pi: f64 = std::f64::consts::PI;    
    for &elements in &exp_val{
        let sqrt_pi = constant / (base * pi);
        let res: f64 = sqrt_pi * elements;
        out.push(res);
    }
    return out;
}