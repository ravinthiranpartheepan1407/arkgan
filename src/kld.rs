pub fn calc_kld(p: &[f64], q: &[f64]) -> f64{
    let mut sum_kld: f64 = 0.0;
    let mut log_vec: Vec<f64> = vec![];
    // Calculate Log Ratios
    for (idx, &elements) in p.iter().enumerate() {
        if q[idx] != 0.0 && elements != 0.0 {
            let log_ratio_calc: f64 = elements / q[idx];
            let log_calc: f64 = if log_ratio_calc > 0.0 {
                f64::log2(log_ratio_calc)
            } else {
                0.0 // Logarithm of non-positive number is undefined
            };
            log_vec.push(log_calc);
        } else {
            log_vec.push(0.0);
        }
    }
    for &elements in &log_vec{
        sum_kld += elements;
        println!("KLD Sum: {:?}", elements);
    }
    return sum_kld;
}

