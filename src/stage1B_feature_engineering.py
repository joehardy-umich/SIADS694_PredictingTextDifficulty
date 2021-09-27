class DynamicImport():
    def __init__(self, import_path):
        self.module = __import__(import_path)

    def get(self, function_or_class_name):
        return getattr(self.module, function_or_class_name)


def get_data(input_data, helper_module):
    helper_module = DynamicImport(helper_module)
    LemmaTokenizer = helper_module.get("LemmaTokenizer")
    tfidf = helper_module.get("tfidf")

    df = pd.read_csv(input_data)

    sparse_df = tfidf.fit_transform(df['original_text'])
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(sparse_df)
    return tfidf_df, df['label']


if __name__ == "__main__":
    import pandas as pd

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_data', help='file containing input training data')
    parser.add_argument('helper_module',
                        help='file containing params and functions to help transform input training data')
    parser.add_argument('data_output', help='file to contain output feature representation')
    parser.add_argument('data_labels', help='file to contain output labels')

    args = parser.parse_args()

    data, labels = get_data(args.input_data, args.helper_module)

    data.to_pickle(args.data_output)
    labels.to_pickle(args.data_labels)
