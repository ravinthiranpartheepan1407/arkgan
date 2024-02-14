pub fn calc_adv_loss(x: &[f64], y: &[f64], sample_size: i64) -> (Vec<f64>,Vec<f64>,f64){
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
    // Calculate Generator
    let generator: f64 = (slope * sample_size as f64) + intercept;
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
    // println!("Y Discriminator: {}",y_discriminator_out);
    // Calculate adversarial loss
    let mut x_adv: Vec<f64> = vec![];
    let mut y_adv: Vec<f64> = vec![];
    let constant: f64 = 1.0;
    for &_elements in x{
        let x_discriminator: f64 = (slope * _elements) - intercept;
        x_discriminator_out += x_discriminator;

        // Ensure the logarithm arguments are valid
        let log_arg1 = if x_discriminator > 0.0 { x_discriminator } else { f64::EPSILON };
        let log_arg2 = if (constant - generator) > 0.0 { (constant - generator) } else { f64::EPSILON };

        let x_adv_loss = -0.5 * f64::log2(log_arg1) + f64::log2(log_arg2);
        x_adv.push(x_adv_loss);
        // println!("X_ADV_LOSS: {}", x_adv_loss);
    }
    for &_elements in y{
        let y_discriminator: f64 = (slope * _elements) - intercept;
        y_discriminator_out += y_discriminator;

        // Ensure the logarithm arguments are valid
        let log_arg1 = if y_discriminator > 0.0 { y_discriminator } else { f64::EPSILON };
        let log_arg2 = if (constant - generator) > 0.0 { (constant - generator) } else { f64::EPSILON };

        let y_adv_loss = -0.5 * f64::log2(log_arg1) + f64::log2(log_arg2);
        y_adv.push(y_adv_loss);
        // println!("Y_ADV_LOSS: {}", y_adv_loss);
    }
    let log_gen = if generator > 0.0 {generator} else {f64::EPSILON};
    let adv_gen: f64 = -0.5 * f64::log2(log_gen);

    return (x_adv, y_adv, adv_gen);
}
