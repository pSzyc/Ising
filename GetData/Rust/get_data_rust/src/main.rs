mod results;
mod simulation;

use std::env;
use rayon::prelude::*;
use std::time::Instant;
use results::parameters_to_csv;
use results::dir_validate;
use simulation::simulate;

fn main() {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 7 && args.len() != 8 {
        eprintln!("Usage: {} <steps> <num_simulations> <output directory> <t> <h> <l> [-stat|--s]", args[0]);
        std::process::exit(1);
    }
    let steps: usize = args[1].parse().expect("Invalid steps");
    let num_simulations: usize = args[2].parse().expect("Invalid num_simulations");
    let output_dir: &str = &args[3];
    let t: f32 = args[4].parse().expect("Invalid t");
    let h: f32 = args[5].parse().expect("Invalid h");
    let l: usize = args[6].parse().expect("Invalid l");
    let stats_out: bool = args.iter().any(|x| x == "-stat" || x == "--s");

    dir_validate(output_dir);
    parameters_to_csv(&args, output_dir);

    let start_time = Instant::now(); 
    println!("Starting {} simulations", num_simulations);

    (1..=num_simulations)
        .into_par_iter() // Create a parallel iterator
        .for_each(|i| {
            let sim_dir = &format!("{}/output{}",output_dir, i);
            dir_validate(sim_dir);
            simulate(steps, l, t, h, sim_dir, stats_out);
        });

    let end_time = Instant::now(); 
    let duration = end_time - start_time;
    println!("{:?} simulations took {:?}", num_simulations, duration);

}

