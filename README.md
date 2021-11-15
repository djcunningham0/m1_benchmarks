# M1 benchmarks

Benchmarking some basic data science tasks, primarily with the new Apple Silicon (M1) Macs.
Results shown here are from my own testing -- your results may vary slightly.

## Benchmarking tests

The following tests are implemented in this repo.
Each subdirectory has a README file with additional details.

* **pandas:** test some basic data operations (e.g., joins) -- this is mainly a memory-intensive task
* **LightGBM:** fit a multiclass classification model using lightgbm -- this is mainly a CPU-intensive task
* **TensorFlow:** fit a few neural networks with different architecture using GPU acceleration -- this is mainly a GPU-intensive task

If you have additional ideas for tests, feel free to open an issue or submit a PR.

*Note:* I have not been able to install XGBoost on an M1 Mac.
If anyone has a solution, please post in the open issue.
I would be interested in testing that library.

## How to run the benchmarks

To run any the benchmarks, clone this repository and then run any of the benchmark_\*.py scripts in the subdirectory.
The script will print out information about how long it took to run the relevant tasks.

The required environment setup is different for each test -- see the README files in each subdirectory for environment details.
(I used a separate environment for each benchmark, but you may be able to use a single conda environment for all of the tests.)

## Results

Again, these are only the results from the machines that were available to me.
Feel free to post your own results in the open issue thread if you have different hardware available to test on.

The table below shows the number of seconds recorded in each benchmark script.

| Hardware                                                                           |   Pandas: join (average) |   Pandas: pivot |   LightGBM |   Tensorflow: CNN (per epoch) | TensorFlow: CycleGAN (per epoch)   |
|------------------------------------------------------------------------------------|--------------------------|-----------------|------------|-------------------------------|------------------------------------|
| 2021 14-inch MacBook Pro M1 Pro 8-core CPU, 14-core GPU, 16 GB memory (base model) |                     5.28 |            9.68 |      25.79 |                          6.06 | 453.33                             |
| 2021 14-inch MacBook Pro M1 Pro 10-core, CPU 16-core GPU 16, GB memory             |                     5.39 |           12.93 |      21.98 |                          5.9  | 410.66                             |
| 2019 16-inch MacBook Pro Intel i7 6-core CPU 16 GB memory                          |                    11.95 |           23.83 |      34.47 |                         23.01 | 1605.84                            |
| 2013 13-inch MacBook Pro Intel i5 2-core CPU 8 GB memory                           |                    42.32 |           37.56 |     127.07 |                         97.01 | lol no                             |

*Note: the 2013 13-inch MacBook Pro is only included because it's the computer I'm replacing.
It's obviously much less powerful than the newer machines (it isn't even able to use GPU acceleration in TensorFlow).*
