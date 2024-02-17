pub fn wasserstein_dist(p: &[f64], q: &[f64]) -> f64{
    // Sort the distributions
    // Sort the distributions P
    let mut sort_p: Vec<f64> = vec![];
    let mut sort_q: Vec<f64> = vec![];
    let mut p_sum: f64 = 0.0;
    let mut q_sum: f64 = 0.0;
    let mut cdf_p: f64 = 0.0;
    let mut cdf_q: f64 = 0.0;
    let mut p_cdf: Vec<f64> = vec![];
    let mut q_cdf: Vec<f64> = vec![];
    let mut wass_dist: f64 = 0.0;
    for &element in p.iter() {
        if sort_p.is_empty() || element >= *sort_p.last().unwrap() {
            sort_p.push(element);
        } else {
            let mut instances = 0;
            while instances < sort_p.len() && sort_p[instances] < element {
                instances += 1;
            }
            sort_p.insert(instances, element);
        }
    }
    // Sort the distributions Q
    let mut sort_q: Vec<f64> = vec![];
    for &element in q.iter() {
        if sort_q.is_empty() || element >= *sort_q.last().unwrap() {
            sort_q.push(element);
        } else {
            let mut instances = 0;
            while instances < sort_q.len() && sort_q[instances] < element {
                instances += 1;
            }
            sort_q.insert(instances, element);
        }
    }
    // Calculate CDF for Sorted P and Q 
    for &elements in p{
        p_sum += elements;
    }
    for &elements in q{
        q_sum += elements;
    }
    // CDF for P
    for &elements in &sort_p{
        let cdf_p_calc: f64 = elements / p_sum;
        cdf_p += cdf_p_calc;
        p_cdf.push(cdf_p);
    }
    // CDF for Q
    for &elements in &sort_q{
        let cdf_q_calc: f64 = elements / q_sum;
        cdf_q += cdf_q_calc;
        q_cdf.push(cdf_q);
    }
    //  Calculate Wasserstein Distance
    for(idx, &elements) in p_cdf.iter().enumerate(){
        let wass_calc: f64 = elements - q_cdf[idx];
        let abs_wass: f64 = wass_calc.abs();
        wass_dist += abs_wass;
    }
    return wass_dist;
}