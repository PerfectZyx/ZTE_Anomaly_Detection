#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2019-01-10 21:29
# * Last modified : 2019-01-11 14:58
# * Filename      : template2vec.py
# * Description   :
'''
这部分的代码是操作已经获得的template vector
'''
# **********************************************************
import os
from gensim.models.word2vec import Word2Vec
import gensim
import numpy as np
class Template2Vec:
    def __init__(self, model_file, template_file, is_binary=False):
        #读取现有的template2vec文件
        print('reading template2vec model')
        model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary = is_binary)
        template_to_index = {}#index(用于lstm)从0开始，template（即模板号）从1开始
        index_to_template = {}#index是int型的
        
        
        template_num = 0 
        with open(template_file) as IN:
            for line in IN:
                template_num += 1
        
        
        template_to_index = dict((str(i+1), i) for i in range(template_num)) #template是模板号
        index_to_template = dict((i, str(i+1)) for i in range(template_num))
        
        template_matrix = []
        for i in range(template_num):
            key = str(i+1)
            template_matrix.append(model[key])
        self.template_matrix = np.mat(template_matrix)
        #print(type(self.template_matrix),self.template_matrix.shape)
        
        vector_template_tuple =[(model[key], key) for key in template_to_index]#向量与模板号的映射关系
        self.model = model
        self.template_to_index = template_to_index
        self.index_to_template = index_to_template
        self.template_num = len(template_to_index)
        self.dimension = len(model['1'])
        self.vector_template_tuple = vector_template_tuple #向量与模板号的映射关系
        print(' Template2Vec.dimension:', self.dimension)
        print(' Template2Vec.template_num:', self.template_num)

    def word_to_most_similar(self, y_word, topn = 1):
        '''
         input:  word
         output: tuple(template_index,similarity)
         与word最相似的，不包括word本身。
        '''
        index = self.model.most_similar(positive = y_word,topn = topn)
        return index 
    
    def vector_to_most_similar(self, y_vector, topn = 1):
        '''
         input: vector
         output: 
         与vector最相似的word，因为预先不知道vector对应的words，所以包含其本身。top1应该是vector对应的word。
        '''
        temp_dict = {}
        for t in self.vector_template_tuple:
            template_index = t[1]
            vector = t[0]
            temp_dict[template_index] = self.cos(y_vector, vector)
        sorted_final_tuple=sorted(temp_dict.items(),key=lambda asd:asd[1] ,reverse=True)
        return sorted_final_tuple[:topn] 
    
    
    def cos(self, vector1, vector2):
        #计算两个向量的相似度
        '''
        dot_product = 0.0;
        normA = 0.0;
        normB = 0.0;
        for a,b in zip(vector1,vector2):
            dot_product += a*b
            normA += a**2
            normB += b**2
        if normA == 0.0 or normB==0.0:
            return None
        else:
            return dot_product / ((normA*normB)**0.5)
        '''
        return float(np.sum(vector1*vector2))/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
        
        
        
    def get_cosine_matrix(self, _matrixB):
        '''
            矩阵矢量化操作，按行计算余弦相似度
            返回值RES为A矩阵每行对B矩阵每行向量余弦值
            RES[i,j] 表示A矩阵第i行向量与B矩阵第j行向量余弦相似度
        '''
        _matrixA = self.template_matrix
        _matrixA_matrixB = _matrixA * _matrixB.reshape(len(_matrixB),-1)
        # 按行求和，生成一个列向量, 即各行向量的模
        _matrixA_norm = np.sqrt(np.multiply(_matrixA,_matrixA).sum())
        _matrixB_norm = np.sqrt(np.multiply(_matrixB,_matrixB).sum())
        return np.divide(_matrixA_matrixB, _matrixA_norm * _matrixB_norm.transpose())

    def vector_to_most_similar_back(self, vectorB, topn = 1):
        '''
         input: vector
         output: 
         与vector最相似的word，因为预先不知道vector对应的words，所以包含其本身。top1应该是vector对应的word。
        '''
        cosine_matrix = self.get_cosine_matrix(vectorB)
        sort_dict = {}
        for i, sim in enumerate(cosine_matrix):
            template_num = str(i+1)
            sort_dict[template_num] = sim
        sorted_final_tuple=sorted(sort_dict.items(),key=lambda asd:asd[1] ,reverse=True)
        return sorted_final_tuple[:topn] 
        
        
if __name__ == '__main__':
    temp2Vec_file = '../../model/bgl_log_20w.template_vector'#../../model/bgl_log_20w.template_vector
    template_file = '../../middle/bgl_log_20w.template'
    t = Template2Vec(temp2Vec_file, template_file)

    print(t.word_to_most_similar(['26'],topn = 3))
    print(t.vector_to_most_similar(t.model['26'],topn = 4))
    print(t.vector_to_most_similar(t.model['26'],topn = 1)[0][0])
    
    print(t.vector_to_most_similar(t.model['26'],topn = 4))


    

    
    
    
    
    
    