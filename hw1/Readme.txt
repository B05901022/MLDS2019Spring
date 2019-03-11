套件需求:

Pytorch 1.0.1
Numpy 1.15.4
scikit-learn 0.20.1
matplotlib 3.0.2
argparse 1.1

Source code:

位於 https://github.com/B05901022/MLDS2019Spring 上，git clone後打開hw1即為此次1-1和1-2之Source code(分別位於hw1-1,hw1-2資料夾)

Reproduce:

1-1 (函數擬合)

1、執行 python3 hw1-1_function.py (-type="deep") 可執行模型訓練並產生結果(-type可接受"shallow","medium","deep")
2、執行 python3 hw1-1_function_plot.py 可依據結果繪圖(需要shallow,medium,deep模型)

1-1 (Cifar10)

1、執行 python3 hw1-1_train.py 可執行模型訓練並產生結果(-type可接受"shallow","medium","deep")
2、執行 python3 hw1-1_train_plot.py 可依據結果繪圖(需要shallow,medium,deep模型)

1-2 (task1)

1、執行 python3 hw1-2_task1.py 可執行模型訓練並產生結果(需注意由於random seed並未固定，因此此程式無法完全reproduce)
2、執行 python3 hw1-2_task1_plot.py 可將weight取PCA後繪圖(reproduce所需之資料存放於All_Weigh.zipt中，將檔案解壓縮至python檔所在資料夾後即可執行)
3、執行 python3 hw1-2_task1_loss_acc_plot.py 可繪出accuracy及loss之訓練曲線圖(reproduce所需之資料存放於CNN_shallow_loss_acc.zip中，
   將檔案解壓縮至python檔所在資料夾後即可執行)
