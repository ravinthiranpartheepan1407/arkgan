pub fn discriminator(x: &[f64], y: &[f64]) -> (f64,f64){
    let mut x_sum: f64 = 0.0;
    let mut y_sum: f64 = 0.0;
    let mut add_nom: f64 = 0.0;
    let mut denom: f64 = 0.0;
    let mut x_discriminator_out: f64 = 0.0;
    let mut y_discriminator_out: f64 = 0.0;
    for &elements in x{
        x_sum += elements;
    }
    for &elements in y{
        y_sum += elements;
    }
    let x_mean = x_sum / x.len() as f64;
    let y_mean: f64 = y_sum / y.len() as f64;
    // Calculate slope (m)
    for (idx, &element) in x.iter().enumerate() {
        let nom: f64 = element * y[idx];
        add_nom += nom;
        let denom_calc: f64 = element - x_mean;
        let pow_denom: f64 = denom_calc.powi(2);
        denom += pow_denom;
    }
    let slope: f64 = add_nom / denom;
    // Calculate Intercept
    let intercept: f64 = y_mean - (slope * x_mean as f64);
    // Calculate Discriminator
    for &elements in x{
        let x_discriminator: f64 = (slope * elements) - intercept;
        x_discriminator_out += x_discriminator;
    }
    // println!("X Discriminator: {}",x_discriminator_out);
    for &elements in y{
        let y_discriminator: f64 = (slope * elements) - intercept;
        y_discriminator_out += y_discriminator;
    } 
    return (x_discriminator_out, y_discriminator_out);
}