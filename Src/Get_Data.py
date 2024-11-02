import pickle
import numpy as np
from collections import defaultdict

train_path = './data/train.txt'
test_path = './data/test.txt'
attribute_path = './data/itemAttribute.txt'
idx_path = './data/node_idx.pkl'

def get_train_data(train_path, item_index):
    """
    从训练数据文件中读取数据，返回以用户ID为键，值为[物品ID, 评分]列表的字典和以物品ID为键，值为[用户ID, 评分]列表的字典。

    :param train_path: 训练数据文件路径
    :param item_index: 物品ID映射索引的字典
    :return: 以用户ID为键，值为[物品ID, 评分]列表的字典和以物品ID为键，值为[用户ID, 评分]列表的字典
    """
    data_user, data_item = defaultdict(list), defaultdict(list)
    with open(train_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            user_id, num = map(int, line.strip().split('|'))
            for _ in range(num):
                line = f.readline()
                item_id, score = line.strip().split()
                item_id, score = int(item_id), float(score)
                score = score / 10  # 将评分映射到0-10之间
                data_user[user_id].append([item_index[item_id], score])
                data_item[item_index[item_id]].append([user_id, score])
    return data_user, data_item

def get_attribute_data(attribute_path, node_idx):
    """
    从物品属性文件中读取数据，返回以物品ID为键，值为属性列表的字典。

    :param attribute_path: 物品属性文件路径
    :param node_idx: 物品ID映射索引的字典
    :return: 以物品ID为键，值为属性列表的字典
    """
    data_attribute = defaultdict(list)
    with open(attribute_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            item_id, attr1, attr2 = line.strip().split('|')
            attr1 = 0 if attr1 == 'None' else 1
            attr2 = 0 if attr2 == 'None' else 1
            item_id = int(item_id)
            if item_id in node_idx:
                data_attribute[node_idx[item_id]].extend([attr1, attr2])
    return data_attribute

def get_test_data(test_path):
    """
    从测试数据文件中读取数据，返回以用户ID为键，值为物品ID列表的字典。

    :param test_path: 测试数据文件路径
    :return: 以用户ID为键，值为物品ID列表的字典
    """
    data_test = defaultdict(list)
    with open(test_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            user_id, num = map(int, line.strip().split('|'))
            for _ in range(num):
                line = f.readline()
                item_id = int(line.strip())
                data_test[user_id].append(item_id)
    return data_test

def split_data(data_user, ratio=0.85, shuffle=True):
    """
    将数据按照给定的比例分割成训练集和验证集。

    :param data_user: 以用户ID为键，值为[物品ID, 评分]列表的字典
    :param ratio: 训练集所占比例，默认为0.85
    :param shuffle: 是否打乱数据，默认为True
    :return: 分割后的训练集和验证集
    """
    train_data, valid_data = defaultdict(list), defaultdict(list)
    for user_id, items in data_user.items():
        if shuffle:
            np.random.shuffle(items)
        train_data[user_id] = items[:int(len(items) * ratio)]
        valid_data[user_id] = items[int(len(items) * ratio):]
    return train_data, valid_data

def load_pkl(pkl_path):
    """
    :param pkl_path: .pkl文件路径
    :return: 加载的数据
    """
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def store_data(pkl_path, data):
    """
    :param pkl_path: .pkl文件路径
    :param data: 要保存的数据
    """
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    print('Start...')
    with open(idx_path, 'rb') as f:
        node_idx = pickle.load(f)
    user_data, item_data = get_train_data(train_path, node_idx)
    store_data(train_path.replace('.txt', '_user.pkl'), user_data)
    store_data(train_path.replace('.txt', '_item.pkl'), item_data)

    attr_data = get_attribute_data(attribute_path, node_idx)
    store_data(attribute_path.replace('.txt', '.pkl'), attr_data)

    test_data = get_test_data(test_path)
    store_data(test_path.replace('.txt', '.pkl'), test_data)
    print('Get data Succeeded!')
