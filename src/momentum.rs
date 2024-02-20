pub fn exec_momentum(x: &[f64], y: &[f64], lr:f64, beta: f64, weight: f64, itr: i32) -> Vec<f64>{
    let mut gen_loss: f64 = weight;
    // X for input data
    // Y for target data
    let mut v: f64 = 0.0;
    let mut gen_old: f64 = 1.0;
    let mut gen_param: Vec<f64> = vec![];
    for(idx, &elements) in x.iter().enumerate(){
        let calc_loss: f64 = ((gen_loss * elements) - y[idx]) * elements as f64; 
        gen_loss = calc_loss;
    }
    // Update the momentum (v)
    for _ in 0..itr{
        let calc_momentum: f64 = beta * v + (1.0 - beta) * gen_loss as f64;
        v = calc_momentum;
        let calc_gen_new: f64 = gen_old - lr * v as f64;
        gen_param.push(calc_gen_new);
    }
    // for _ in 0..itr{
    //     let calc_gen_new: f64 = gen_old - lr * v as f64;
    //     gen_param.push(calc_gen_new);
    // }
    return gen_param;
}