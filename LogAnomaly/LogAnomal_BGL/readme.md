# 环境

python>=3.6 keras numpy tensorflow

## 运行

<!--train: python train_log_deeplog.py --train_file train_log_seqence.txt --model_dir weights/test/ --template_index_map_path template_to_int.txt

test: python detect_log_deeplog.py --test_file test_log_seqence.txt --model_dir weights/test/ --template_index_map_path template_to_int.txt --result_file precision_recall.txt-->

依次运行下面几个命令

* python log_main.py --train_file train_log_seqence.txt --test_file test_log_seqence.txt --model_dir weights/test/ --template_index_map_path template_to_int.txt   --result_file precision_recall.txt





