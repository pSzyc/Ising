Code repository for a bachelor's thesis "Application of Machine Learning to Ising model".<br> 
The thesis was about comparing VAE and GAN models by trying to recreate 2D Ising configurations with appropriate probability distribution. <br>
The ML code is under `/Model` directory where as MC simulations are under `/GetData`  
To run the simulations you have to have installed Cargo and Rust or Python.<br>
(https://doc.rust-lang.org/cargo/getting-started/installation.html)<br>
For executing example simulation in Python please run script as follows:
```console
python GetData/Python/get_data.py 500 100 Data/Test 2.9 0 64 5
```
for doing the same in Rust:
```console
cd GetData/Rust/get_data_rust
cargo run 500 100 Data/Test 2.9 0 64
```
It is worth noticing that Rust code is significantly faster. To execute above commands it was needed
- 531.46 seconds with Python
- 28.99 seconds with Rust and Cargo

The python algorithm has been modified with replacing for loop iteration with numpy array operations which gave significant improvment when it comes to time of the execution (
100 simulations took 5.63 seconds) but made the simulation getting stuck in some stable state. Run to replicate:
```console
python GetData/Python/get_data.py 500 100 Data/Test 2.9 0 64 5 -np 1
```
