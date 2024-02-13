pub fn calc_iou(x: &[f64], y: &[f64]) -> f64 {
    let mut same_elements: Vec<f64> = vec![];
    let mut diff_elements_x: Vec<f64> = vec![];
    let mut diff_elements_y: Vec<f64> = vec![];
    // for &elements in x{
    //     for &idx in y{
    //         if elements == idx{
    //             println!("Same Elements: {} {}",elements, idx);
    //             same_elements.push(elements);
    //         }else{
    //             println!("Different Elements: {} {}", elements, idx);
    //             diff_elements_x.push(elements);
    //         }   
    //     }
    // }
    for &elements_x in x{
        let mut is_same = false;
        for &elements_y in y{
            if elements_x == elements_y{
                same_elements.push(elements_x);
                is_same = true;
            }
        }
        if !is_same{
            diff_elements_x.push(elements_x);
        }
    }
    for &elements_y in y{
        if !x.contains(&elements_y){
            diff_elements_y.push(elements_y);
        }
    }
    println!("{:?} {:?} {:?}",same_elements, diff_elements_x, diff_elements_y);
    let intersect: f64 = same_elements.len() as f64;
    let union: f64 = (x.len() + y.len()) as f64;
    let out: f64 = (intersect / union) as f64;
    return out;
}