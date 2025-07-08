from sklearn.model_selection import train_test_split

def split_data(texts, labels, test_size=0.2, random_state=42):
    return train_test_split(texts, labels, test_size=test_size, random_state=random_state)