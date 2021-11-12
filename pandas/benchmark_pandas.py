"""
Test how long it takes to perform some data operations using pandas. Specifically:
* Join two large dataframes on multiple keys
* Pivot a large dataframe
"""

import pandas as pd
import numpy as np
import string
import time


np.random.seed(0)


## parameters for join benchmark
N_ROWS_1 = 500_000
N_COLS_1 = 500

N_ROWS_2 = 500_000
N_COLS_2 = 500

N_KEY_COLS = 100
KEY_OPTIONS = list(string.ascii_uppercase)
JOIN_COLS = list(range(N_KEY_COLS))

# parameters for pivot benchmark
N_PIVOT_ROWS = 10_000
N_INDEX_COLS = 20
N_PIVOT_COLS = 20
N_VALUE_COLS = 40


def generate_random_df(n_rows: int, n_cols: int, n_key_cols: int) -> pd.DataFrame:
    print(f"generating dataframe with {n_rows:,} rows, {n_cols:,} cols, {n_key_cols:,} key cols...", end=" ", flush=True)
    start = time.time()
    data = {}
    for i in range(n_key_cols):
        data[i] = np.random.choice(KEY_OPTIONS, size=n_rows, replace=True)

    for j in range(n_key_cols, n_cols):
        data[j] = np.random.random(n_rows)

    out = pd.DataFrame(data)
    end = time.time()
    print(f"{end - start} seconds", flush=True)
    return out


def time_join(how: str) -> float:
    start = time.time()
    _ = df_1.merge(df_2, on=JOIN_COLS, how=how)
    end = time.time()
    elapsed = end - start
    print(f"{how} join: {elapsed} seconds")
    return elapsed


def time_pivot(df: pd.DataFrame, n_index_cols: int, n_cols: int, n_value_cols: int) -> float:
    start = time.time()
    df.columns = [str(x) for x in df.columns]
    _ = df.pivot(
        index=list(df.columns[0:n_index_cols]),
        columns=list(df.columns[n_index_cols:n_index_cols+n_cols]),
        values=list(df.columns[n_index_cols+n_cols:n_index_cols+n_cols+n_value_cols]),
    )
    end = time.time()
    elapsed = end - start
    print(f"pivot: {elapsed} seconds")
    return elapsed


df_1 = generate_random_df(N_ROWS_1, N_COLS_1, N_KEY_COLS)
df_2 = generate_random_df(N_ROWS_2, N_COLS_2, N_KEY_COLS)

print("")
join_times = []
join_times.append(time_join("left"))
join_times.append(time_join("right"))
join_times.append(time_join("inner"))
join_times.append(time_join("outer"))
print(f"mean join time: {np.mean(join_times)} seconds\n")

key_pivot_cols = N_PIVOT_COLS + N_INDEX_COLS
total_pivot_cols = key_pivot_cols + N_VALUE_COLS
pivot_df = generate_random_df(N_PIVOT_ROWS, total_pivot_cols, key_pivot_cols)

time_pivot(
    df=pivot_df,
    n_index_cols=N_INDEX_COLS,
    n_cols=N_PIVOT_COLS,
    n_value_cols=N_VALUE_COLS,
)
