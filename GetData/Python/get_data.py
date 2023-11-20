#  Code used in simulating the data. Repository used in creating this file:
#  https://github.com/DanielJCase/Ising-Deep-Learning/blob/master/Ising_genData.ipynb
import click
from pathlib import Path
from simulate import simulate
from multiprocessing import Pool
import time

@click.command()
@click.argument('steps', type=int)
@click.argument('n_samples', type=int)
@click.argument('output_file', type=str, default="Data")
@click.argument('temp', type=float, default=2.73)
@click.argument('H', type=float, default=0.)
@click.argument('L', type=int, default=32)
@click.argument('n_workers', type=int, default=5)
@click.option('-w', '--wolff_sim', default=False, type=bool)
@click.option('-s', '--stats', default=False, type=bool)
def get_data(steps, n_samples, output_file, temp, h, l, n_workers, wolff_sim, stats):
    output_file = Path(output_file)
    output_file.mkdir(parents=True, exist_ok=True)
    start_time = time.time()  # Start measuring time
    if wolff_sim and h != 0:
        raise ValueError("Wolff can't simulate external magnetic field. Set it to 0!")
    with Pool(n_workers) as pool:
        pool.starmap(simulate, [(steps, l, temp , h, output_file / f"output{i + 1}", wolff_sim, stats)  for i in range(n_samples)])
   
    with open(f"{output_file}/parameters.csv", 'w') as f:
        f.write("Steps,Simulatiton Number,Temperature,Magnetic Field,Mattize Size,Wolff\n")
        f.write(f"{steps},{n_samples},{temp},{h},{l},{wolff_sim}")
    
    end_time = time.time()  # End measuring time
    duration = end_time - start_time
    print(f"{n_samples} simulations took {duration} seconds")

if __name__ == '__main__':
    get_data()