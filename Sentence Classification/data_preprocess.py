import pandas as pd


def read_data(train_file):
    train_df = pd.read_csv(train_file, sep='\t')
    
    # test_df = pd.read_csv(test_file, sep="\t")
    return train_df["Phrase"].values, train_df["Sentiment"].values


if __name__ == "__main__":
    #X:sentence Y:semantic label
    X_data, y_data = read_data("dataset/train.tsv")
    print("train size", len(X_data))
