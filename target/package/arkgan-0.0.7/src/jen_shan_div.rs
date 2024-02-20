pub fn calc_jen_shan_div(x: &[f64], y: &[f64]) -> f64{
    let mut avg_dist: Vec<f64> = vec![];
    let mut x_kbd = 0.0;
    let mut y_kbd: f64 = 0.0;
    for(idx, &elements) in x.iter().enumerate(){
        println!("{} {}",elements,y[idx]);
        let sum = elements + y[idx];
        let avg = sum / 2 as f64;
        avg_dist.push(avg);
    }
    println!("{:?}", avg_dist);
    // Compute Kullback Divergence X,Y
    for (idx, &elements) in x.iter().enumerate(){
        let div: f64 = elements / avg_dist[idx];
        // println!("Div: {} / {} : {}", elements,avg_dist[idx],div);
        let calc: f64 = if div > 0.0 {div} else {f64::EPSILON};
        let kdv_calc = elements * f64::log(calc, 2.0);
        x_kbd += kdv_calc;
        // println!("Kullback Divergence X: {}",kdv_calc);
    }
    for (idy, &elements) in y.iter().enumerate(){
        let div: f64 = elements / avg_dist[idy];
        // println!("Div: {} / {} : {}", elements,avg_dist[idy],div);
        let calc: f64 = if div > 0.0 {div} else {f64::EPSILON};
        let kdv_calc = elements * f64::log(calc, 2.0);
        y_kbd += kdv_calc;
        // println!("Kullback Divergence Y: {}",kdv_calc);
    }
    let out: f64 = 0.5 * (x_kbd + y_kbd) as f64;
    return out;
}

