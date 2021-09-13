def combine(eda, vector):
    eda_df = pd.read_pickle(eda)
    eda_df[pd.isnull(eda_df)] = 0.

    vector_df = pd.read_pickle(vector)

    return pd.concat([eda_df, vector_df], axis=1)


if __name__ == "__main__":
    import pandas as pd
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'eda_training_data_file', help='file containing eda training data')
    parser.add_argument(
        'vector_training_data_file', help='file containing vector training data')
    parser.add_argument(
        'data_output_file', help='file to contain combined baseline model')
    args = parser.parse_args()

    print("Combining data...")
    combined = combine(args.eda_training_data_file, args.vector_training_data_file)

    pickle.dump(combined, open(args.data_output_file, 'wb'))
