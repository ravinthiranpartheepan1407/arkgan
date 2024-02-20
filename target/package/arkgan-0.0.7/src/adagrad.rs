pub fn exec_adagrad(x: &[f64], y: &[f64], lr: f64, itr: i32) ->(Vec<f64>, Vec<f64>){
    let mut x_sum: f64 = 0.0;
    let mut y_sum: f64 = 0.0;
    let mut nom: f64 = 0.0;
    let mut denom: f64 = 0.0;
    let mut sum_grad_m: f64 = 0.0;
    let mut sum_grad_b: f64 = 0.0;
    let mut slope_out: Vec<f64> = vec![];
    let mut inter_out: Vec<f64> = vec![];
    let ep: f64 = 0.00000001;
    for &elements in x{
        x_sum += elements;
    }
    for &elements in y{
        y_sum += elements;
    }
    let x_mean: f64 = x_sum / x.len() as f64;
    let y_mean: f64 = y_sum / y.len() as f64;
    // Calculate slope
    for(idx, &elements) in x.iter().enumerate(){
        let nom_calc: f64 = ((elements - x_mean) * (y[idx] - y_mean)) as f64;
        nom +=  nom_calc;
        let denom_calc:f64 = elements - x_mean as f64;
        let sqr_denom: f64 = denom_calc.powi(2);
        denom += sqr_denom;
    }
    let mut slope: f64 = nom /  denom as f64;
    let mut intercept: f64 = y_mean - (slope * x_mean) as f64;
    for _ in 0..itr{
        let mut m_loss: Vec<f64> = vec![];
        let mut b_loss: Vec<f64> = vec![];
        for (xi, yi) in x.iter().zip(y.iter()) {
            let predicted = slope * xi + intercept;
            let loss_m = -2.0 * xi * (yi - predicted);
            let loss_b = -2.0 * (yi - predicted);
            m_loss.push(loss_m);
            b_loss.push(loss_b);
        }

        // Calculate squared gradients and update sum of squared gradients
        let grad_m: f64 = m_loss.iter().sum();
        let grad_b: f64 = b_loss.iter().sum();
        sum_grad_m += grad_m * grad_m;
        sum_grad_b += grad_b * grad_b;

        // Update parameters of slope and intercept
        slope -= lr / (sum_grad_m + ep).sqrt() * grad_m;
        intercept -= lr / (sum_grad_b + ep).sqrt() * grad_b;

        // Store the updated values
        slope_out.push(slope);
        inter_out.push(intercept);
    }
    return (slope_out, inter_out);
}