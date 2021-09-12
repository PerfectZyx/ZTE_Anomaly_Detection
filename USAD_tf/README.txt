# 文件夹说明：
# # processed：存放训练及测试数据
# # model：存放已训练的模型参数
# # results：存放训练结果
# # # anomaly_detect.png：异常检测结果图
# # # test_score.png：测试集分数图
# # # train_score.png：训练集分数图
# # # test_score.npy：测试集分数
# # # train_score.npy：训练集分数
# # # model_result：模型训练相关参数，异常检测异常点
# # usad：存放算法实现相关文件

# 运行
python main.py --dataset 数据集名称 --参数名 参数（详细参数见usad/config.yml）
