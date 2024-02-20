pub fn lr_scheduler(lr: f64, step_decay: f64, steps: &[usize]) -> Vec<f64>{
    let mut lr_rate: Vec<f64> = vec![];
    let mut init_lr:f64 = lr;
    let mut max_step: usize = 0;
    let mut key: usize = 0;
    // Get max step val from vector
    for &elements in steps{
        if max_step < elements{
            max_step = elements;  
        }else{
            max_step = max_step;
        }
    }
    for elements in 0..max_step+1{
        if elements == steps[key]{
            init_lr = step_decay * init_lr as f64;
            // let set_lr: f64 = init_lr as usize;
            lr_rate.push(init_lr);
            key += 1;
        }else{
            lr_rate.push(init_lr);
        }
    }
    return lr_rate;
}