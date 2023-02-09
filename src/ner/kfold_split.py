if __name__ == "__main__":
    import os
    from utils import conll2pandas, pandas2json

    print("[ INFO ] Loading Conll Dataset")
    conll_data = conll2pandas("data/ner/second_ner.conll")

    save_path = "data/ner/"
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, random_state=0, shuffle=True)

    for i, (train_index, test_index) in enumerate(kf.split(conll_data)):
        path_fold = save_path + "/" + "fold-" + str(i) + "/"  # PATH TO SAVE
        os.makedirs(path_fold)  # CREATE THE FOLDER VERSION AND SUBFOLDER

        # get the data from indexes
        train_data, test_data = conll_data.loc[train_index], conll_data.loc[test_index]

        pandas2json(train_data, os.path.join(path_fold, "train.json"))
        pandas2json(test_data, os.path.join(path_fold, "eval.json"))
    print("[ INFO ] Completed!")
