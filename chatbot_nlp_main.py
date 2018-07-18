# -*- coding: utf-8 -*-

# TODO gestire in modo efficiente normalizzazione target, calcolo tf-idf
# TODO add logg class

import os.path
import time
import pickle
import pandas as pd
from numpy import array
from pprint import pprint
from modules import utils, word_normalizer as wnorm,  word_embedding as wemb, question_answering as qa, classification as classif

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.externals import joblib



def get_data(dataset_filepath, dump_file, load_dump, sheet_name, field_to_normalize, lang, langtag, treetagger_path, stop_words_flg, stop_words_src, stop_words_dataset):
    docs_df = pd.DataFrame()

    if(load_dump):
        file = open(dump_file, 'rb')
        docs_df = pickle.load(file)
    else:
        docs_df = utils.load_data_from_excel(dataset_filepath, sheet_name)
        # docs_df.set_index('ID', inplace=True)
        docs_df['NORMALIZED'] = docs_df.apply(lambda row: wnorm.normalize_text(row[field_to_normalize], lang, langtag, treetagger_path, stop_words_flg, stop_words_src, stop_words_dataset), axis=1)
        utils.pickle_result(docs_df, dump_file)

    return docs_df

def get_target(dataset_filepath, dump_file, load_dump, lang, langtag, treetagger_path, stop_words_flg, stop_words_src, stop_words_dataset, sheet_name='Dataset', field_to_normalize='DOMANDA'):
    return get_data(dataset_filepath, dump_file, load_dump, sheet_name, field_to_normalize, lang, langtag, treetagger_path, stop_words_flg, stop_words_src, stop_words_dataset)


def get_input(dataset_filepath, dump_file, load_dump, lang, langtag, treetagger_path, stop_words_flg, stop_words_src, stop_words_dataset, sheet_name='Test', field_to_normalize='RIFORMULAZIONE'):
    return get_data(dataset_filepath, dump_file, load_dump, sheet_name, field_to_normalize, lang, langtag, treetagger_path, stop_words_flg, stop_words_src, stop_words_dataset)


def question_classification(question_vect):
    scope_string = classif.in_out_scope(question_vect)
    return scope_string

def evaluate_single_question_telegram(question,w2v):
    #print('Start Evaluating single question...')
    print('Loading configuration file ...')
    cfg_dict = utils.load_configuration()
    #pprint(cfg_dict)

    # todo passa config list to func
    DATASET_PATH = cfg_dict['DATASET_PATH']
    DATASET_FILE = cfg_dict['DATASET_FILE']
    W2V_MODEL = cfg_dict['W2V_MODEL']
    TREETAGGER_PATH = cfg_dict['TREETAGGER_PATH']
    INPUT_DUMP_OBJ_SUFFIX = cfg_dict['INPUT_DUMP_OBJ_SUFFIX']
    TARGET_DUMP_OBJ_SUFFIX = cfg_dict['TARGET_DUMP_OBJ_SUFFIX']
    INPUT_LOAD_DUMP_FLAG = cfg_dict['INPUT_LOAD_DUMP_FLAG'] == 'True'
    TARGET_LOAD_DUMP_FLAG = cfg_dict['TARGET_LOAD_DUMP_FLAG'] == 'True'
    LANG = cfg_dict["LANG"]
    LANGTAG = cfg_dict["LANGTAG"]
    STOP_WORDS_FLAG = cfg_dict['STOP_WORDS_FLAG'] == 'True'
    STOPWORDS_SRC = cfg_dict["STOPWORDS_SRC"]
    STOPWORDS_FILE = cfg_dict["STOPWORDS_FILE"]
    DATASET_FILEPATH = DATASET_PATH + '/' + DATASET_FILE
    INPUT_DUMP_FILEPATH = DATASET_PATH + '/' + W2V_MODEL + INPUT_DUMP_OBJ_SUFFIX
    TARGET_DUMP_FILEPATH = DATASET_PATH + '/' + W2V_MODEL + TARGET_DUMP_OBJ_SUFFIX

    print('Loading target...')
    target_docs_df = get_target(DATASET_FILEPATH, TARGET_DUMP_FILEPATH, TARGET_LOAD_DUMP_FLAG, LANG, LANGTAG, TREETAGGER_PATH, STOP_WORDS_FLAG, STOPWORDS_SRC, STOPWORDS_FILE)
    #print('Number of target questions: {}'.format(len(target_docs_df)))

    #print('Loading w2v model...')
    #w2v = wemb.get_word_embedding_matrix(W2V_MODEL, LANG)
    #print('Number of words in vocabulary: {}'.format(len(w2v.wv.vocab)))
    
    print('Normalize the question...')
    question_to_list = []
    question_to_list.append(question)
    question_normalized_list = []
    question_normalized_list.append(wnorm.normalize_text(question, LANG, LANGTAG, TREETAGGER_PATH, STOP_WORDS_FLAG, STOPWORDS_SRC, STOPWORDS_FILE))
    

    print('Vectorization...')
    target_normalized_document_list = list(target_docs_df['NORMALIZED'])
    target_vect_file_name = 'target_vect.pkl'
    if os.path.isfile(target_vect_file_name):
        print('Load target_vect.pkl...')
        tfidf_w2v_target_doc_list = joblib.load(target_vect_file_name)
    else:
        tfidf_w2v_target_doc_list = qa.evaluate_docs_vectorization(target_normalized_document_list, target_normalized_document_list, w2v, W2V_MODEL)
        joblib.dump(tfidf_w2v_target_doc_list, target_vect_file_name)
  
    tfidf_w2v_question_list = qa.evaluate_docs_vectorization(question_normalized_list, target_normalized_document_list, w2v,  W2V_MODEL)


    print('Evaluating similarities...')
    t2v_target_list = [sum(tfidf_w2v_dict.values())  for tfidf_w2v_dict in tfidf_w2v_target_doc_list]
    
    t2v_question = []
    t2v_question.append(sum(tfidf_w2v_question_list[0].values()))

    
    print('Evaluating class of the question...')
    scope_string = question_classification(t2v_question)
    #print(scope_string)

    if scope_string == 'OUT':
        print('Question:')
        #print('The Question is out of scope')
        return 'Mi dispiace, sono un Bot specializzato in micronutrienti quindi non posso aiutarti :('

 
    similarity_vect_sum_doc_list = qa.evaluate_vector_similarity(t2v_target_list, t2v_question)
    
    predicted_index = similarity_vect_sum_doc_list.argmax()
    predicted_value = similarity_vect_sum_doc_list.max()
    predicted_str = target_docs_df.iloc[predicted_index]['DOMANDA']
    predicted_answer = target_docs_df.iloc[predicted_index]['RISPOSTA']

    #print('Question:')
    print(question)
    #print('Predicted Answer:')
    #print(predicted_answer)
    return predicted_answer


def evaluate_test(input_docs_df, w2v_model, dataset_dir, dataset_filepath, dump_file, load_dump, lang, langtag, treetagger_path, stop_words_flg, stop_words_src, stop_words_dataset):
    print('Loading w2v model...')
    w2v = wemb.get_word_embedding_matrix(w2v_model, lang)
    print('Number of words in vocabulary: {}'.format(len(w2v.wv.vocab)))
    print('Loading target...')
    target_docs_df = get_target(dataset_filepath, dump_file, load_dump, lang, langtag, treetagger_path, stop_words_flg, stop_words_src, stop_words_dataset)
    print('Number of target questions: {}'.format(len(target_docs_df)))

    print('Vectorization...')
#    input_normalized_document_list = [text[2].split() for text in word_lemmas_list]
#    target_normalized_document_set  = [text[3] for text in word_lemmas_list]
    input_normalized_document_list = list(input_docs_df['NORMALIZED'])
    target_normalized_document_set  = list(target_docs_df['NORMALIZED'])
    tfidf_w2v_input_doc_list = qa.evaluate_docs_vectorization(input_normalized_document_list, target_normalized_document_set, w2v, w2v_model)
    tfidf_w2v_output_doc_list = qa.evaluate_docs_vectorization(target_normalized_document_set, target_normalized_document_set, w2v, w2v_model)

    print('Evaluating similarities...')
    t2v_input_list = [sum(tfidf_w2v_dict.values())  for tfidf_w2v_dict in tfidf_w2v_input_doc_list]
    
    t2v_output_list = []
    for tfidf_w2v_dict in tfidf_w2v_output_doc_list:
        #pprint(tfidf_w2v_dict)    
        t2v_output_list.append(sum(tfidf_w2v_dict.values()))

    #print('saving...')
    #X = np.asarray(t2v_output_list)
    #np.save(dataset_dir + '/target_new_question_voliHotelPrestiti.npy', X)
 

    similarity_vect_sum_doc_list = qa.evaluate_vector_similarity(t2v_input_list, t2v_output_list)
    
    print('Reporting...')
    report_list_tfidf = {}
    nr_ok = 0
    nr_input_docs = len(input_docs_df)

    for index, row in input_docs_df.iterrows():
        similarity_doc_list = qa.evaluate_doc_similarity_2(tfidf_w2v_input_doc_list[index], tfidf_w2v_output_doc_list)
        input_str = row['RIFORMULAZIONE']
        input_normalized_str = row['NORMALIZED']
        predicted_index = array(similarity_doc_list).argmax()
        predicted_value = array(similarity_doc_list).max()
        predicted_str = target_docs_df.iloc[predicted_index]['DOMANDA']
        predicted_normalized_str = target_docs_df.iloc[predicted_index]['NORMALIZED']
        #print(target_docs_df['DOMANDA'])
        #print(row['DOMANDA ORIGINALE'])
        target_index = target_docs_df[target_docs_df['DOMANDA'] == row['DOMANDA ORIGINALE']].index.values.astype(int)[0]
        target_value = similarity_doc_list[target_index]
        target_str = target_docs_df.iloc[target_index]['DOMANDA']
        target_normalized_str = target_docs_df.iloc[target_index]['NORMALIZED']
        predicted_vs_target = 'OK' if predicted_index == target_index else 'KO'
        if predicted_vs_target == 'OK':
            nr_ok += 1
        report_list_tfidf[index] = {'Input id': index,
                              'Input': input_str,
                              'Input normalized': input_normalized_str,
                              'Target id': target_index,
                              'Target': target_str,
                              'Target normalized': target_normalized_str,
                              'Target sim': target_value,
                              'Predicted id': predicted_index,
                              'Predicted': predicted_str,
                              'Predicted normalized': predicted_normalized_str,
                              'Predicted sim': predicted_value,
                              'predicted_vs_target': predicted_vs_target}

    pprint(report_list_tfidf)
    print('Numero OK: {} di {} ({:.2%})'.format(nr_ok, nr_input_docs, nr_ok / nr_input_docs))

    print('Reporting...')
    report_list = {}
    nr_ok = 0
    nr_input_docs = len(input_docs_df)

    for index, row in input_docs_df.iterrows():
        input_str = row['RIFORMULAZIONE']
        input_normalized_str = row['NORMALIZED']
        predicted_index = similarity_vect_sum_doc_list[index].argmax()
        predicted_value = similarity_vect_sum_doc_list[index].max()
        predicted_str = target_docs_df.iloc[predicted_index]['DOMANDA']
        predicted_normalized_str = target_docs_df.iloc[predicted_index]['NORMALIZED']
        target_index = target_docs_df[target_docs_df['DOMANDA'] == row['DOMANDA ORIGINALE']].index.values.astype(int)[0]
        target_value = similarity_vect_sum_doc_list[index][target_index]
        target_str = target_docs_df.iloc[target_index]['DOMANDA']
        target_normalized_str = target_docs_df.iloc[target_index]['NORMALIZED']
        predicted_vs_target = 'OK' if predicted_index == target_index else 'KO'
        if predicted_vs_target == 'OK':
            nr_ok += 1
        report_list[index] = {'Input id': index,
                              'Input': input_str,
                              'Input normalized': input_normalized_str,
                              'Target id': target_index,
                              'Target': target_str,
                              'Target normalized': target_normalized_str,
                              'Target cosine-sim': target_value,
                              'Predicted id': predicted_index,
                              'Predicted': predicted_str,
                              'Predicted normalized': predicted_normalized_str,
                              'Predicted cosine-sim': predicted_value,
                              'predicted_vs_target': predicted_vs_target}

    pprint(report_list)
    print('Numero OK: {} di {} ({:.2%})'.format(nr_ok, nr_input_docs, nr_ok / nr_input_docs))

    utils.write_report_to_excel(dataset_dir, 'BeBot_Test_' + w2v_model + '- sum.xlsx', 'Test Accuracy', report_list)
    utils.write_report_to_excel(dataset_dir, 'BeBot_Test_' + w2v_model + '- tfidf.xlsx', 'Test Accuracy', report_list_tfidf)


def report_answers_correctness():
    pass




def main(argv=None):
    start_time = time.time()
    print('Starting...')
    print('Loading configuration file ...')
    cfg_dict = utils.load_configuration()
    pprint(cfg_dict)

    # todo passa config list to func
    DATASET_PATH = cfg_dict['DATASET_PATH']
    DATASET_FILE = cfg_dict['DATASET_FILE']
    W2V_MODEL = cfg_dict['W2V_MODEL']
    TREETAGGER_PATH = cfg_dict['TREETAGGER_PATH']
    INPUT_DUMP_OBJ_SUFFIX = cfg_dict['INPUT_DUMP_OBJ_SUFFIX']
    TARGET_DUMP_OBJ_SUFFIX = cfg_dict['TARGET_DUMP_OBJ_SUFFIX']
    INPUT_LOAD_DUMP_FLAG = cfg_dict['INPUT_LOAD_DUMP_FLAG'] == 'True'
    TARGET_LOAD_DUMP_FLAG = cfg_dict['TARGET_LOAD_DUMP_FLAG'] == 'True'
    LANG = cfg_dict["LANG"]
    LANGTAG = cfg_dict["LANGTAG"]
    STOP_WORDS_FLAG = cfg_dict['STOP_WORDS_FLAG'] == 'True'
    STOPWORDS_SRC = cfg_dict["STOPWORDS_SRC"]
    STOPWORDS_FILE = cfg_dict["STOPWORDS_FILE"]
    DATASET_FILEPATH = DATASET_PATH + '/' + DATASET_FILE
    INPUT_DUMP_FILEPATH = DATASET_PATH + '/' + W2V_MODEL + INPUT_DUMP_OBJ_SUFFIX
    TARGET_DUMP_FILEPATH = DATASET_PATH + '/' + W2V_MODEL + TARGET_DUMP_OBJ_SUFFIX

    print('Loading input...')
    input_docs_df = get_input(DATASET_FILEPATH, INPUT_DUMP_FILEPATH, INPUT_LOAD_DUMP_FLAG, LANG, LANGTAG, TREETAGGER_PATH, STOP_WORDS_FLAG, STOPWORDS_SRC, STOPWORDS_FILE)
    print('Number of input questions: {}'.format(len(input_docs_df)))

    evaluate_test(input_docs_df, W2V_MODEL, DATASET_PATH, DATASET_FILEPATH, TARGET_DUMP_FILEPATH, TARGET_LOAD_DUMP_FLAG, LANG, LANGTAG, TREETAGGER_PATH, STOP_WORDS_FLAG, STOPWORDS_SRC, STOPWORDS_FILE)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Execution time: {}'.format(elapsed_time))
    print('End')


if __name__ == '__main__':
    main()
