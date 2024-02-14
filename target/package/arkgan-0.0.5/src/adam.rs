pub fn exec_adam(x: &[f64], y: &[f64], lr: f64, b1: f64, b2: f64, itr: i32) -> (Vec<f64>,Vec<f64>){
// Calculate Gradient
    let mut x_sum: f64 = 0.0;
    let mut y_sum: f64 = 0.0;
    let mut nom_val: f64 = 0.0;
    let mut denom_val:f64 = 0.0;
    let mut loss_val: f64 = 0.0;
    let mut m_slope_val: f64 = 0.0;
    let mut b_intercept_val: f64 = 0.0;
    let mut m_steps: f64 = 0.0;
    let mut v_steps: f64 = 0.0;
    let mut slope_out: Vec<f64> = vec![];
    let mut intr_out: Vec<f64> = vec![];
    let ep: f64 = 0.00000001;
    for &elements in x{
        x_sum += elements;
    }
    for &elements in y{
        y_sum += elements;
    }
    let x_mean: f64 = x_sum;
    let y_mean: f64 = y_sum;
    // Calculate slope
    for(idx, &elements) in x.iter().enumerate(){
        let nom_calc = ((elements - x_mean) * (y[idx] - y_mean)) as f64;
        nom_val += nom_calc;
        let denom_calc = elements - x_mean as f64;
        let denom_sqr: f64 = denom_calc.powi(2) as f64;
        denom_val += denom_sqr;
    }
    let mut slope: f64 = nom_val / denom_val as f64;
    // Calculate Intercept
    let mut intercept: f64 = y_mean - (slope * x_mean) as f64;
    // Calculate Loss
    for (idx, &elements) in x.iter().enumerate(){
        let loss_calc: f64 = y[idx] - ((slope * elements) + intercept) as f64;
        let loss_sqr: f64 = loss_calc.powi(2) as f64;
        loss_val += loss_sqr;
    }
    let loss: f64 = (1.0 / ((2.0 * x.len() as f64) * loss_val)) as f64; 
    // Calculate Slope Loss
    for(idx, &elements) in x.iter().enumerate(){
        let m_loss_calc: f64 = - (y[idx] * ((slope * elements) + intercept)) * elements as f64;
        m_slope_val += m_loss_calc;
    }
    let m_loss: f64 = (1.0 / x.len() as f64) * m_slope_val as f64;
    // Calculate Intercept Loss
    for(idx, &elements) in x.iter().enumerate(){
        let b_intercept_calc: f64 = - (y[idx] * ((slope * elements) + intercept)) as f64;
        b_intercept_val += b_intercept_calc;
    }
    let b_loss: f64 = (1.0 / x.len() as f64) * b_intercept_val as f64;
    // Update Moment Estimates
    let mut instances: i32 = 0;
    while instances < itr{
        // Moving Averages of Gradient bt
        let m_step_calc: f64 = ((b1 * m_steps) + (1.0 - b1)) * m_loss as f64;
        m_steps = m_step_calc;
        // Moving averages of squared gradient of vt
        let v_step_calc: f64 = ((b2 * v_steps) + (1.0 - b2)) * (m_loss.powi(2)) as f64;
        v_steps = v_step_calc;
        // Bias-Corrected Moment Estimates
        // Gradient Steps
        let m_steps_val: f64 = m_steps / (1.0 - b1.powi(instances + 1)) as f64;
        let v_steps_val: f64 = v_steps / (1.0 - b2.powi(instances + 1)) as f64;
        // Update Parameters
        let upd_m: f64 = slope - (lr * m_steps_val) / (v_steps_val.sqrt() + ep);
        slope = upd_m;
        let upd_b: f64 = intercept - (lr * m_steps_val) / (v_steps_val.sqrt() + ep);
        intercept = upd_b;

        slope_out.push(slope);
        intr_out.push(intercept);

        instances += 1;
    }
    return (slope_out, intr_out);
}