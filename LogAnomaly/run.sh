cd ./ft_tree
python -u ft_tree.py -data_path ../data/bgl.log -template_path ../middle/bgl_log.template -fre_word_path ../middle/bgl_log.fre
echo "----------------------------------------------------"
python -u matchTemplate.py -template_path ../middle/bgl_log.template -fre_word_path ../middle/bgl_log.fre -log_path ../data/bgl.log -out_seq_path ../data/bgl_log.seq -match_model 1
echo "----------------------------------------------------"
cd ../data
python removeAnomaly.py -input_seq bgl_log.seq -input_label bgl.label
echo "----------------------------------------------------"
cd ../template2Vector/src/preprocess/
python wordnet_process_mwb.py -data_dir ../../../middle/ -template_file bgl_log.template -syn_file bgl_log.syn -ant_file bgl_log.ant
echo "----------------------------------------------------"
cd ../../../middle
python changeTemplateFormat.py -input bgl_log.template
echo "----------------------------------------------------"
cd ../template2Vector/src
make -j8
./lrcwe -train ../../middle/bgl_log.template_for_training -synonym ../../middle/bgl_log.syn -antonym ../../middle/bgl_log.ant -output ../../model/bgl_log.model -save-vocab ../../middle/bgl_log.vector_vocab -belta-rel 0.8 -alpha-rel 0.01 -belta-syn 0.4 -alpha-syn 0.2 -alpha-ant 0.3 -size 32 -min-count 1
echo "----------------------------------------------------"
python template2Vec.py -template_file ../../middle/bgl_log.template -word_model ../../model/bgl_log.model -template_vector_file ../../model/bgl_log.template_vector -dimension 32
echo "----------------------------------------------------"
cd ../../LogAnomal_BGL
python -u train_vector_2LSTM.py -train_file ../data/bgl_log.seq_normal -seq_length 10 -model_dir ../weights/vector_matrix/ -onehot 1 -template2Vec_file ../model/bgl_log.template_vector -template_file ../middle/bgl_log.template -count_matrix 1 
echo "----------------------------------------------------"
python -u detect_vector_2LSTM.py -test_file ../data/bgl_log.seq -template_index_map_path ../data/bgl_log.seq_normal_map -seq_length 10 -model_dir ../weights/vector_matrix/ -n_candidates 15 -windows_size 3 -step_size 1 -onehot 1 -result_file ../results/bgl_log_precision_recall.txt -label_file ../data/bgl.label -template2Vec_file ../model/bgl_log.template_vector -template_file ../middle/bgl_log.template -count_matrix 1 
echo "----------------------------------------------------"
