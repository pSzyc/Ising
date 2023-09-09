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
@click.argument('temp', type=float, default=2.9)
@click.argument('H', type=float, default=0.)
@click.argument('L', type=int, default=64.)
@click.argument('n_workers', type=int, default=5)
def get_data(steps, n_samples, output_file, temp, h, l, n_workers):
    output_file = Path(output_file)
    output_file.mkdir(parents=True, exist_ok=True)
    start_time = time.time()  # Start measuring time
    with Pool(n_workers) as pool:
        pool.starmap(simulate, [(steps, l, temp , h, output_file / f"out-{i}")  for i in range(n_samples)])
    end_time = time.time()  # End measuring time
    duration = end_time - start_time
    print(f"{n_samples} Simulation took {duration} seconds")

if __name__ == '__main__':
    get_data()
