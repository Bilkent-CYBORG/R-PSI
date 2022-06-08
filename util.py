import pickle

def pickle_read(file_name):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except Exception as ex:
        print(ex)
        return None


def pickle_save(file_name, data):
    try:
        with open(file_name, 'wb') as f:
            return pickle.dump(data, f)
    except Exception as ex:
        print(ex)
        return None

