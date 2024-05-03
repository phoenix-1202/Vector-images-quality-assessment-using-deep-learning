import pickle


def save_data(arr_embs, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(arr_embs, f)


def read_files(file_path):
    with open(file_path, 'rb') as file:
        emb_array = pickle.load(file)

    return emb_array
