import pickle
import numpy as np

train_user_pkl = './data/train_user.pkl'
train_item_pkl = './data/train_item.pkl'

user_num = 19835
item_num = 455705
ratings_num = 5001507


def get_bias(train_data_user, train_data_item):
    """
    计算评分均值和用户、物品偏差。
    :param train_data_user: 以用户ID为键，[物品ID, 评分]为值的字典
    :param train_data_item: 以物品ID为键，[用户ID, 评分]为值的字典
    :return: 全局评分均值，用户偏差，物品偏差
    """
    miu = 0.0  # 初始化全局评分均值
    bx = np.zeros(user_num, dtype=np.float64)  # 初始化用户偏差数组
    bi = np.zeros(item_num, dtype=np.float64)  # 初始化物品偏差数组

    # 计算用户偏差和全局评分均值
    for user_id in train_data_user:
        sum = 0.0  # 当前用户的评分总和
        for item_id, score in train_data_user[user_id]:
            miu += score  # 累加全局评分
            sum += score  # 累加用户评分
        bx[user_id] = sum / len(train_data_user[user_id])  # 计算当前用户的平均评分
    miu /= ratings_num  # 计算全局评分均值

    # 计算物品偏差
    for item_id in train_data_item:
        sum = 0.0  # 当前物品的评分总和
        for user_id, score in train_data_item[item_id]:
            sum += score  # 累加物品评分
        bi[item_id] = sum / len(train_data_item[item_id])  # 计算当前物品的平均评分

    bx -= miu  # 减去全局评分均值，得到用户偏差
    bi -= miu  # 减去全局评分均值，得到物品偏差
    return miu, bx, bi  # 返回全局评分均值、用户偏差和物品偏差


if __name__ == '__main__':
    print('Loading data...')
    # 读取以用户ID为键，[物品ID，评分]为值的字典
    with open(train_user_pkl, 'rb') as f:
        train_user_data = pickle.load(f)
    # 读取以物品ID为键，[用户ID，评分]为值的字典
    with open(train_item_pkl, 'rb') as f:
        train_item_data = pickle.load(f)
    print('Data loaded.')

    # 计算总体偏差，用户偏差，物品偏差
    miu, bias_user, bias_item = get_bias(train_user_data, train_item_data)

    print('Saving data...')
    # 保存用户偏差
    with open('./data/Bias_user.pkl', 'wb') as f:
        pickle.dump(bias_user, f)
    # 保存物品偏差
    with open('./data/Bias_item.pkl', 'wb') as f:
        pickle.dump(bias_item, f)
    print('Data saved.')
