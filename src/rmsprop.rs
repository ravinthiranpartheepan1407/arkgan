pub fn exec_rmsprop(x: &[f64], y: &[f64], lr:f64, decay: f64, itr: i32) -> (Vec<f64>, Vec<f64>){
    let mut x_sum: f64 = 0.0;
    let mut y_sum:f64 = 0.0;
    let mut nom: f64 = 0.0;
    let mut denom: f64 = 0.0;
    let mut loss_val: f64 = 0.0;
    let ep: f64 = 0.00000001;
    let mut grad_sum_m: f64 = 0.0;
    let mut grad_sum_b: f64 = 0.0;
    let mut grad_slope: Vec<f64> = vec![];
    let mut grad_intercept: Vec<f64> = vec![];
    let mut slope_loss: Vec<f64> = vec![];
    let mut intercept_loss: Vec<f64> = vec![];
    let mut sqrd_gradi_m: Vec<f64> = vec![];
    let mut sqrd_gradi_b: Vec<f64> = vec![];
    let mut res_slope: Vec<f64> = vec![];
    let mut res_intercept: Vec<f64> = vec![];

    for &elements in x{
        x_sum += elements;
    }
    for &elements in y{ 
        y_sum += elements;
    }
    let x_mean: f64 = x_sum;
    let y_mean: f64 = y_sum;
    // Calculate Slope (m)
    for(idx, &elements) in x.iter().enumerate(){
        let nom_calc: f64 = ((elements - x_mean) * (y[idx] - y_mean)) as f64;
        nom += nom_calc;
        let denom_calc: f64 = elements - x_mean as f64;
        let denom_sqr: f64 = denom_calc.powi(2) as f64;
        denom += denom_sqr;
    }
    let mut slope:f64 = nom / denom as f64;
    let mut intercept: f64 = y_mean * (slope * x_mean) as f64;
    let mut instances: i32 = 0;
    while instances < itr{
        let mut sqrd_grd_m: f64 = slope;
        let mut sqrd_grd_b: f64 = intercept;
        // Calculate Loss
        for(idx, &elements) in x.iter().enumerate(){
            let loss_calc: f64 = ((y[idx]) - ((slope * elements) + intercept));
            let loss_pow: f64 = loss_calc.powi(2);
            loss_val += loss_pow;
        }
        let loss: f64 = 1.0 / (2.0 * x.len() as f64) * loss_val as f64;
        // Calculate Gradient
            // Calculate slope loss
        for(idx, &elements) in x.iter().enumerate(){
            let m_loss_calc: f64 = -(y[idx] - (slope * elements)) * elements  as f64;
            slope_loss.push(m_loss_calc);
        }
            // Calculate Intercept Loss
        for(idx, &elements) in x.iter().enumerate(){
            let b_loss_calc: f64 = -(y[idx] - ((slope * elements) + intercept)) as f64;
            intercept_loss.push(b_loss_calc);
        }
        // Calculate Squared Gradients (m)
        for &elements in &slope_loss{
            let upd_sqrd_m: f64 = ((decay * sqrd_grd_m) + ((1.0 - decay) * elements.powi(2))) as f64; 
            sqrd_gradi_m.push(upd_sqrd_m);
        }
        // Calculate Squared Gradients (b)
        for &elements in &intercept_loss{
            let upd_sqrd_b: f64 = ((decay * sqrd_grd_b) + ((1.0 - decay) * elements.powi(2))) as f64; 
            sqrd_gradi_b.push(upd_sqrd_b);
        }
        // Update Parameters
        for &elements in &slope_loss{
            let slope_calc: f64 = slope - (lr / f64::sqrt((elements + ep))) * elements as f64;
            res_slope.push(slope_calc);
        }
        for &elements in &intercept_loss{
            let intercept_calc: f64 = intercept - (lr / f64::sqrt((elements + ep))) * elements as f64;
            res_intercept.push(intercept_calc);
        }
        // Root mean square for slope
        for(idx, &elements) in x.iter().enumerate(){
            let grad_m_calc: f64 = (elements * y[idx]) + (slope * (elements.powi(2))) as f64;
            grad_sum_m += grad_m_calc;
        }
        let mut m_v_org: f64 = 0.0;
        let mut grad_m_g: f64 = (1.0/x.len() as f64) * grad_sum_m as f64;
        // Update (v0) RMSProp update rule
        let mut upd_m_v_org: f64 = (decay * m_v_org) + ((1.0 - decay) * grad_sum_m) as f64;
        m_v_org = upd_m_v_org;
        // Calculate RMS Gradient for m
        let mut m_rmd_grad: f64 = f64::sqrt(upd_m_v_org);
        let mut m_rms_m_calc: f64 = (slope  - (lr / (m_rmd_grad + ep))) * grad_m_g as f64; 
        sqrd_grd_m = m_rms_m_calc;
        
        // Root mean square for intercept
        for(idx, &elements) in x.iter().enumerate(){
            let grad_b_calc: f64 = (y[idx]) + (slope * (elements.powi(2))) as f64;
            grad_sum_b += grad_b_calc;
        }
        let mut b_v_org: f64 = 0.0;
        let mut grad_b_g: f64 = (1.0/x.len() as f64) * grad_sum_b as f64;
        // Update (v0) RMSProp update rule
        let mut upd_b_v_org: f64 = (decay * b_v_org) + ((1.0 - decay) * grad_sum_b) as f64;
        b_v_org = upd_b_v_org;
        // Calculate RMS Gradient for m
        let mut b_rmd_grad: f64 = f64::sqrt(upd_b_v_org);
        let mut b_rms_m_calc: f64 = (slope  - (lr / (b_rmd_grad + ep))) * grad_b_g as f64; 
        sqrd_grd_b = b_rms_m_calc;
        instances += 1;
    }
    return (res_slope, res_intercept);
}