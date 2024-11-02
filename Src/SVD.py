import os
import time
import psutil
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from Get_Data import split_data, load_pkl, store_data

test_pkl = './data/test.pkl'
bx_pkl = './data/Bias_user.pkl'
bi_pkl = './data/Bias_item.pkl'
idx_pkl = './data/node_idx.pkl'

class SVD:
    def __init__(self, model_path='./model',
                 data_path='./data/train_user.pkl',
                 lr=5e-3,
                 lamda1=1e-2,
                 lamda2=1e-2,
                 lamda3=1e-2,
                 lamda4=1e-2,
                 factor=50):

        # 加载偏置
        self.bx = load_pkl(bx_pkl)  # 用户偏置
        self.bi = load_pkl(bi_pkl)  # 物品偏置

        # 设置模型参数
        self.lr = lr  # 学习率
        self.lamda1 = lamda1  # 正则化系数
        self.lamda2 = lamda2
        self.lamda3 = lamda3
        self.lamda4 = lamda4
        self.factor = factor  # 隐向量维度

        # 加载索引和数据
        self.idx = load_pkl(idx_pkl)
        self.train_user_data = load_pkl(data_path)
        self.train_data, self.valid_data = split_data(self.train_user_data)
        self.test_data = load_pkl(test_pkl)

        # 计算全局评分
        self.globalmean = self.get_globalmean()

        # 初始化用户和物品矩阵
        self.Q = np.random.normal(0, 0.1, (self.factor, len(self.bi)))  # 矩阵Q(物品)
        self.P = np.random.normal(0, 0.1, (self.factor, len(self.bx)))  # 矩阵P(用户)

        # 模型路径
        self.model_path = model_path

    def get_globalmean(self):
        """
        计算全局平均分。
        """
        score_sum, count = 0.0, 0
        for user_id, items in self.train_user_data.items():
            for item_id, score in items:
                score_sum += score
                count += 1
        return score_sum / count

    def predict(self, user_id, item_id):
        """
        预测用户对物品的评分。
        """
        pre_score = self.globalmean + \
                    self.bx[user_id] + \
                    self.bi[item_id] + \
                    np.dot(self.P[:, user_id], self.Q[:, item_id])
        return pre_score

    def loss(self, is_valid=False):
        """
        计算模型的损失函数。

        Args:
            is_valid (bool): 是否为验证集，默认为False。

        Returns:
            float: 损失值。
        """
        loss, count = 0.0, 0
        data = self.valid_data if is_valid else self.train_data
        for user_id, items in data.items():
            for item_id, score in items:
                loss += (score - self.predict(user_id, item_id)) ** 2
                count += 1
        if not is_valid:
            loss += self.lamda1 * np.sum(self.P ** 2)
            loss += self.lamda2 * np.sum(self.Q ** 2)
            loss += self.lamda3 * np.sum(self.bx ** 2)
            loss += self.lamda4 * np.sum(self.bi ** 2)
        loss /= count
        return loss

    def rmse(self):
        """
        计算均方根误差。
        """
        rmse, count = 0.0, 0
        for user_id, items in self.train_user_data.items():
            for item_id, score in items:
                rmse += (score - self.predict(user_id, item_id)) ** 2
                count += 1
        rmse /= count
        rmse = np.sqrt(rmse)
        return rmse

    def train(self, epochs=10, save=False, load=False):
        """
        训练模型。
        """
        if load:
            self.load_weight()
        print('Start training...')

        for epoch in range(epochs):
            for user_id, items in tqdm(self.train_data.items(), desc=f'Epoch {epoch + 1}'):
                for item_id, score in items:
                    error = score - self.predict(user_id, item_id)

                    # 更新用户和物品的偏置
                    self.bx[user_id] += self.lr * (error - self.lamda3 * self.bx[user_id])
                    self.bi[item_id] += self.lr * (error - self.lamda4 * self.bi[item_id])

                    # 更新用户和物品的隐向量
                    self.P[:, user_id] += self.lr * (error * self.Q[:, item_id] - self.lamda1 * self.P[:, user_id])
                    self.Q[:, item_id] += self.lr * (error * self.P[:, user_id] - self.lamda2 * self.Q[:, item_id])

            # 每个epoch的训练和验证集损失
            print(f'Epoch {epoch + 1} train loss: {self.loss():.6f} valid loss: {self.loss(is_valid=True):.6f}')

        print('Training finished.')

        if save:
            self.save_weight()

    def test(self, write_path='./result/result.txt', load=True):
        """
        测试模型。
        Args:
            write_path (str): 结果保存路径，默认为 './result/result.txt'。
            load (bool): 是否加载已有的模型参数，默认为 True。
        Returns:
            dict: 预测评分结果。
        """
        if load:
            self.load_weight()  # 加载已有模型参数
        print('Start testing...')

        # 初始化一个字典，用于存储预测评分结果
        predict_score = defaultdict(list)

        # 遍历测试数据中的每个用户和对应的物品列表
        for user_id, item_list in self.test_data.items():
            for item_id in item_list:
                if item_id not in self.idx:  # 如果物品不在索引中
                    pre_score = self.globalmean * 10  # 使用全局平均分的10倍作为默认评分
                else:
                    new_id = self.idx[item_id]  # 获取物品的新索引
                    pre_score = self.predict(user_id, new_id) * 10  # 预测评分并乘以10

                    # 将评分限制在0到100之间
                    if pre_score > 100.0:
                        pre_score = 100.0
                    elif pre_score < 0.0:
                        pre_score = 0.0

                # 将预测结果加入字典
                predict_score[user_id].append((item_id, pre_score))

        print('Testing finished.')

        def write_result(predict_score, write_path):
            """
            将预测结果写入文件。
            """
            print('Start writing...')
            with open(write_path, 'w') as f:
                for user_id, items in predict_score.items():
                    f.write(f'{user_id}|6\n')
                    for item_id, score in items:
                        f.write(f'{item_id} {score}\n')
            print('Writing finished.')

        if write_path:
            write_result(predict_score, write_path)

        return predict_score


    def load_weight(self):
        """
        加载模型参数。
        """
        print('Loading weight...')
        self.bx = load_pkl(self.model_path + '/bx.pkl')
        self.bi = load_pkl(self.model_path + '/bi.pkl')
        self.P = load_pkl(self.model_path + '/P.pkl')
        self.Q = load_pkl(self.model_path + '/Q.pkl')
        print('Loading weight finished.')

    def save_weight(self):
        """
        保存模型参数。
        """
        print('saving weight...')
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        store_data(self.model_path + '/bx.pkl', self.bx)
        store_data(self.model_path + '/bi.pkl', self.bi)
        store_data(self.model_path + '/P.pkl', self.P)
        store_data(self.model_path + '/Q.pkl', self.Q)
        print('done.')

    def precision_recall_f1(self, true_score, threshold=50):
        """
        计算精确率、召回率和F1分数。

        Args:
            true_score (dict): 真实评分结果字典。
            threshold (float): 评分的阈值，大于等于该值被认为是正例，默认为50。

        Returns:
            float: 精确率、召回率和F1分数的元组。
        """
        true_positive, false_positive, false_negative = 0, 0, 0
        for user_id, items in true_score.items():
            for item_id, true_s in items:
                pred_score = self.predict(user_id, item_id) * 10  # 预测评分并乘以10

                if true_s >= threshold and pred_score >= threshold:
                    true_positive += 1
                elif true_s >= threshold and pred_score < threshold:
                    false_negative += 1
                elif true_s < threshold and pred_score >= threshold:
                    false_positive += 1

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return precision, recall, f1

if __name__ == '__main__':
    svd = SVD()

    #start_time = time.time()
    # start_memory = psutil.virtual_memory().used
    #
    #
    # svd.train(epochs=10, save=True, load=True)
    #
    # end_time = time.time()
    # training_time = end_time - start_time

    # print(f"Training time: {training_time} seconds")


    svd.test(write_path='./result/result_svd.txt')

    rmse = svd.rmse()

    print(f'RMSE: {rmse:.6f}')

    #precision, recall, f1 = svd.precision_recall_f1()
    #print(f'Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}')