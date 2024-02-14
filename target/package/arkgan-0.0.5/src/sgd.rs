// const X: [f64;5] = [0.37,0.51,-0.97,0.54,0.76];
// const Y: [f64;5] = [0.72,0.13,0.43,-0.46,0.61];

pub fn exec_sgd(x: &[f64], y: &[f64], lr: f64, itr: i32) -> (Vec<f64>,Vec<f64>){
    let mut x_sum: f64 = 0.0;
    let mut y_sum: f64 = 0.0;
    let mut nom: f64 = 0.0;
    let mut denom: f64 = 0.0;
    let mut m_loss:Vec<f64> = vec![];
    let mut b_loss:Vec<f64> = vec![];
    let mut upd_m: Vec<f64> = vec![];
    let mut upd_b: Vec<f64> = vec![];
    // Calculate MeaVn
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
    let mut slope: f64 = nom / denom as f64;
    let mut intercept: f64 = y_mean - (slope * x_mean) as f64; 
    // Calculate Loss
    let mut instances: i32 = 0;
    while instances < itr{
        // let mut mse: f64 = 0.0;
        m_loss.clear();
        b_loss.clear();
        for(idx, &elements) in x.iter().enumerate(){
            let prediction = slope * elements + intercept;
            let error = prediction - y[idx];

            // Update parameters using SGD
            slope -= lr * error * elements as f64;
            intercept -= lr * error as f64;
        }
        // let mse_loss: f64 = (1.0 / (2.0 * x.len() as f64)) * mse as f64;
        // Calculate Gradient
            // Slope loss (m_loss)
        for(idx, &elements) in x.iter().enumerate(){
            let m_loss_calc: f64 = -(y[idx] - ((slope * elements) + intercept)) * slope as f64;
            m_loss.push(m_loss_calc);
        }
        for(idx, &elements) in y.iter().enumerate(){
            let b_loss_calc: f64 = -(elements - ((slope * x[idx]) + intercept)) as f64;
            b_loss.push(b_loss_calc);
        }
        // Update Parameters of slope and intercept
        for &elements in &m_loss{
            let upd_m_calc: f64 = slope - (lr * elements) as f64;
            upd_m.push(upd_m_calc);
        }
        for &elements in &b_loss{
            let upd_b_calc: f64 = slope - (lr * elements) as f64;
            upd_b.push(upd_b_calc);
        }
        instances += 1;
    }
    return (upd_m, upd_b);
}

// fn main(){
//     let (slope, intercept) = exec_sgd(&X,&Y,0.01,100);
//     println!("SGD Slope: {:?}", slope);
//     println!("--------------------------------------------------------");
//     print!("SGD Intercept: {:?}", intercept);
// }