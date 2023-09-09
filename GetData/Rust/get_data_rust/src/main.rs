extern crate ndarray;
extern crate ndarray_rand;
extern crate rand_distr;
extern crate ndarray_npy;
extern crate rayon;
use std::env;
use rayon::prelude::*;
use rand::seq::SliceRandom;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand_distr::Uniform;
use rand::Rng;
use ndarray_npy::write_npy;
use std::time::Instant;
use std::fs;


fn create_array(l: usize) -> Array2<i32> {
    let mut rng = rand::thread_rng();
    let mat = Array::random_using((l, l), Uniform::new(0, 2), &mut rng);
    let mat = mat.map(|x| if *x == 0 { -1 } else { *x });
    mat
}
fn save_npy(mat: &Array2<i32>, output_file: &str) {
    write_npy(output_file, mat).expect("Failed to write NPY file");
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
            let neighbour_sum =( mat[[i, (j + 1) % l]] + mat[[i, (j + l - 1) % l]]
                + mat[[(i + 1) % l, j]] + mat[[(i + l - 1) % l, j]]) as f32;
            let e_current = -spin_current * neighbour_sum - spin_current * h;
            let e_new = -spin_new * neighbour_sum - spin_new * h;
            let e_diff = e_new - e_current;
            if e_diff < 0.0 || rng.gen_range(0.0..1.0) <= (-e_diff / t).exp() {
                mat[[i, j]] = spin_new as i32;
            }
        }
    }
}



fn simulate(steps: usize, l: usize, t: f32, h :f32, output_file: &str) {
    let mut mat = create_array(l);
    for _ in 0..steps {
        iterate(&mut mat, t, h);
    }
    save_npy(&mat, output_file);

}

fn dir_validate(directory_path: &str){
    // Check if the directory exists
    if !fs::metadata(&directory_path).is_ok() {
        if let Err(err) = fs::create_dir_all(&directory_path) {
            eprintln!("Error creating directory: {}", err);
            std::process::exit(1);
        } else {
            println!("Directory created: {}", directory_path);
        }
    } else {
        println!("Directory already exists: {}", directory_path);
    }
}

fn main() {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() != 7 {
        eprintln!("Usage: {} <steps> <l> <t> <h> <num_simulations> <output directory>", args[0]);
        std::process::exit(1);
    }

    let steps: usize = args[1].parse().expect("Invalid steps");
    let l: usize = args[2].parse().expect("Invalid l");
    let t: f32 = args[3].parse().expect("Invalid t");
    let h: f32 = args[4].parse().expect("Invalid h");
    let num_simulations: usize = args[5].parse().expect("Invalid num_simulations");
    let output_dir: &str = &args[6];
    dir_validate(output_dir);

    let start_time = Instant::now(); 
    println!("Starting {} simulations", num_simulations);
    (1..=num_simulations)
        .into_par_iter() // Create a parallel iterator
        .for_each(|i| {
            let output_file = format!("{}/output{}.npy",output_dir, i);
            simulate(steps, l, t, h, &output_file);
        });

    let end_time = Instant::now(); 
    let duration = end_time - start_time;

    println!("{:?} simulations took {:?}", num_simulations, duration);
}