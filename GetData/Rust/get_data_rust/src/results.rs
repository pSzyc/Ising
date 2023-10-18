use ndarray::Array2;
use std::fs;
use csv::Writer;
use ndarray_npy::write_npy;

pub fn save_npy(mat: &Array2<i32>, output_file: &str) {
    write_npy(output_file, mat).expect("Failed to write NPY file");
}

pub fn dir_validate(directory_path: &str){
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

pub fn vec_to_csv(data: &Vec<(usize, f32, f32)>, filename: &str){
    let file = fs::File::create(filename).expect("Failed to create file");
    let mut writer = Writer::from_writer(file);
    for record in data.iter() {
        writer.write_record(&[
            record.0.to_string(),
            record.1.to_string(),
            record.2.to_string(),
        ]).expect("Failed to write record to CSV");
    }
    writer.flush().expect("Failed to flush CSV writer");
}

pub fn parameters_to_csv(args: &Vec<String>, output_dir: &str){
    let par_file = format!("{}/parameters.csv",output_dir);
    let file = fs::File::create(par_file).expect("Failed to create file");
    let mut writer = Writer::from_writer(file);
    writer.write_record(&[
        "Steps".to_string(),
        "Simulatiton Number".to_string(),
        "Temperature".to_string(),
        "Magnetic Field".to_string(),
        "Mattize Size".to_string(),
        "Wolff".to_string()
    ]).expect("Failed to write record to CSV");
    let wolff: bool = args.iter().any(|x| x == "-wolff" || x == "--w");

    writer.write_record(&[
        args[1].to_string(),
        args[2].to_string(),
        args[4].to_string(),
        args[5].to_string(),
        args[6].to_string(),
        wolff.to_string()
    ]).expect("Failed to write record to CSV");
    writer.flush().expect("Failed to flush CSV writer");
}


