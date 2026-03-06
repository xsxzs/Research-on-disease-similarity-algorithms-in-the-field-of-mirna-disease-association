import warnings
from train import *
from param import *

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 MDformer miRNA-疾病关联预测模型启动")
    print("=" * 60)
    
    args = parse_args()
    print(f"📊 使用数据集: {args.path}")
    print(f"🖥️  计算设备: {args.device}")
    print(f"🔄 交叉验证折数: {args.kfolds}")
    print(f"📈 训练轮数: {args.epoch}")
    print(f"🎯 学习率: {args.lr}")
    
    repeat = 1
    warnings.filterwarnings("ignore")
    for i in range(repeat):
        print(f"\n📋 开始第{i+1}次完整训练...")
        # ******************5-cv训练代码******************
        averages = fold_valid(args)
        print(f"\n🏆 最终平均结果: {averages}")

    print("\n✅ 训练完成!")



#AUC	    0.9504	受试者工作特征曲线下面积，衡量分类性能
#AUPR	    0.9489	精确率-召回率曲线下面积，衡量不平衡数据性能
#Accuracy	0.8806	准确率，正确预测的比例
#F1-Score	0.8800	F1分数，精确率和召回率的调和平均
#Precision	0.8845	精确率，预测为正例中实际为正例的比例
#Recall  	0.8761	召回率，实际正例中被正确预测的比例