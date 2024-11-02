import pickle

train_path = './data/train.txt'

def allude_index(train_path):
    item_set = set()  # 物品的集合
    with open(train_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            _, num = map(int, line.strip().split('|'))  # 用户ID,物品数量
            for _ in range(num):
                line = f.readline()  # 逐行读取
                item_id, _ = map(int, line.strip().split())  # 获得物品ID
                item_set.add(item_id)  # 添加到集合

    item_index = {node: idx for idx, node in enumerate(sorted(item_set))}  # 为每个物品ID分配一个唯一的索引，存储到字典item_index中
    return item_index


if __name__ == '__main__':
    item_index = allude_index(train_path)  # 获取物品ID索引字典
    with open('./data/node_idx.pkl', 'wb') as f:
        pickle.dump(item_index, f)  # 使用pickle模块保存索引字典到文件
    print('succeed!')
