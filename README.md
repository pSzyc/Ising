Tu run the simulations you have to have installed Cargo and Rust or Python.<br>
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
