use crate::results;

use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray::Array2;
use results::save_npy_bool;
use results::vec_to_csv;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::Uniform;


fn create_array(l: usize) -> Array2<i32> {
    let mut rng = rand::thread_rng();
    let mat = 2 * Array::random_using((l, l), Uniform::new(0, 2), &mut rng) - 1;
    mat
}


fn flip_cluster(mat: &mut Array2<i32>, cluster: &[(usize, usize)]) {
    // Flip all spins in a cluster. Used for the Wolff algorithm.
    for &(i, j) in cluster {
        mat[[i, j]] *= -1;
    }
}

fn iterate_wolff(mat: &mut Array2<i32>, t: f32, _h: f32) {
    // Wolff algorithm
    let l = mat.shape()[0];
    let mut tracker = Array2::zeros((l, l));

    let mut rng = rand::thread_rng();
    let i = rand::thread_rng().gen_range(0..l);
    let j = rand::thread_rng().gen_range(0..l);
    let spin = mat[[i, j]];
    let mut stack = vec![(i, j)];
    tracker[[i, j]] = 1;

    let mut cluster = vec![(i, j)];

    while !stack.is_empty() {
        let (i, j) = stack.pop().unwrap();
        let neighbors = vec![
            (i, (j + 1) % l),
            (i, (j + l - 1) % l),
            ((i + 1) % l, j),
            ((i + l - 1) % l, j),
            ((i + 1) % l, (j + 1) % l),
            ((i + l - 1) % l, (j + l - 1) % l),
        ];

        for pair in neighbors {
            let (l, m) = pair;
            if mat[[l, m]] == spin && tracker[[l, m]] == 0 && rng.gen::<f32>() < (1.0 - (-2.0 / t).exp()) {
                cluster.push((l, m));
                stack.push((l, m));
                tracker[[l, m]] = 1;
            }
        }
    }

    flip_cluster(mat, &cluster);
}



fn iterate(mat: &mut Array2<i32>, t: f32, h: f32) {
    let (l, _) = mat.dim();
    let mut order1: Vec<usize> = (0..l).collect();
    let mut order2: Vec<usize> = (0..l).collect();
    let mut rng = rand::thread_rng();
    order1.shuffle(&mut rng);
    order2.shuffle(&mut rng);

    for &i in &order1 {
        for &j in &order2 {
            let spin_current = mat[[i, j]] as f32;
            let spin_new = -spin_current;
            let neighbour_sum = (
                mat[[i, (j + 1) % l]] + mat[[i, (j + l - 1) % l]] +
                mat[[(i + 1) % l, j]] + mat[[(i + l - 1) % l, j]] +
                mat[[(i + 1) % l, (j + 1) % l]] + mat[[(i + l - 1)%l, (j + l - 1) % l]]
            ) as f32;
            let e_current = -spin_current * neighbour_sum - spin_current * h;
            let e_new = -spin_new * neighbour_sum - spin_new * h;
            let e_diff = e_new - e_current;
            if e_diff < 0.0 || rng.gen_range(0.0..1.0) <= (-e_diff / t).exp() {
                mat[[i, j]] = spin_new as i32;   
            }
        }
    }
}


fn calc_macro_config(mat: &Array2<i32>, h: f32) -> (f32, f32){
    let (l, _) = mat.dim();
    let mut energy: f32 = 0.0;
    let mag: f32 = mat.sum() as f32;

    for i in 0..l{
        for j in 0..l{
            let neighbour_sum = (
                mat[[i, (j + 1) % l]] + mat[[i, (j + l - 1) % l]] +
                mat[[(i + 1) % l, j]] + mat[[(i + l - 1) % l, j]] +
                mat[[(i + 1) % l, (j + 1) % l]] + mat[[(i + l - 1) % l, (j + l - 1) % l]]
            ) as f32;
            energy += - mat[[i,j]] as f32 * (neighbour_sum + h);
        }
    }
    (energy, mag)
}


pub fn simulate(steps: usize, l: usize, t: f32, h :f32, sim_dir: &str, stats_out: bool, wolff: bool) {
    let mut mat = create_array(l);
    let mut time_vec: Vec<(usize, f32, f32)> = Vec::new();

    let iterate_function = if wolff {
        iterate_wolff
    } else {
        iterate
    };

    for step in 0..steps {
        iterate_function(&mut mat, t, h);
        if  stats_out{
            let (energy, mag) = calc_macro_config(&mat, h);
            time_vec.push((step, energy, mag));
        }
    }
    if stats_out{
        vec_to_csv(&time_vec, &format!("{}/data.csv", sim_dir));
    }
    let output_file = &format!("{}/final.npy", sim_dir);
    save_npy_bool(&mat, output_file);

}