# %matplotlib
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import pickle
import os
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans, MeanShift, MiniBatchKMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.decomposition import PCA
import re
from gensim.models import Word2Vec, FastText
import umap
import hdbscan
from tqdm import tqdm
import unidecode
from scipy.stats import variation
import collections
# import cupy as cp
import gc

def strip_accents(s):
	#retira acentos de strings
	s = s.replace('`','').replace("'",'')
	return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def grava(coisa,filename,path):
	pkl_file = open(path + filename, 'wb')
	pickle.dump(coisa, pkl_file)
	pkl_file.close()

def grava_arquivo_grande(coisa,filename,path):
	pkl_file = open(path + filename, 'wb')
	pickle.dump(coisa, pkl_file, protocol=4)
	pkl_file.close()

def abre(filename,path):
	#formato do path:'/Users/Alexandre/'
	pkl_file = open(path + filename, 'rb')
	coisa = pickle.load(pkl_file)
	pkl_file.close()
	return coisa

def descobre_arquivos_na_pasta(pasta,tipo_do_arquivo='.xlsx'):
	#Descobre arquivos na pasta:
	arquivos = []
	for file in os.listdir(pasta):
		arquivos.append(os.fsdecode(file))
	arquivos = [arquivo for arquivo in arquivos if tipo_do_arquivo in arquivo] #seleciona soh arquivos com .xlsx
	return arquivos

def standardizacao(dict_of_dfs):
	std = {}
	for key in dict_of_dfs:
		scaler = StandardScaler()
		std[key] = DataFrame(scaler.fit_transform(dict_of_dfs[key]),index=dict_of_dfs[key].index,columns=dict_of_dfs[key].columns)
	return std

def doc2vec(model, doc, size, weights=None,remove_out_of_vocabulary_words=False): #faz media dos vetores word embeddings:
	# se passar weights, tem que retirar as palavras de fora do dicionario
	# remove out-of-vocabulary words #se for FastText nao precisa tirar.
	if remove_out_of_vocabulary_words:
		if weights:
			doc = [word for word in doc if (word in model.wv.vocab and word in weights)]
		else:
			doc = [word for word in doc if word in model.wv.vocab]
	if len(doc) == 0:
		return np.zeros(size,)
	else:
		if weights: #retorna a media do embbeding vector de todas as words do documento
			return np.average(model.wv[doc], axis=0,weights=[weights[word] for word in doc] )
		else: #se nao for passado weights, faz media simples:
			return np.mean(model.wv[doc], axis=0)

def find_all_matches_by_words(words,df,column):
	# encontra todos os registros que tem todas as palavras passadas:
	new_df = df
	for word in words:
		filter = new_df[column].str.contains('(^|\W)' + word + '(\W|$)',regex=True)
		filter = filter[filter==True].index
		new_df = df.loc[filter]
	cols = [column] + [c for c in df if 'grupo' in c]
	# results = df.loc[filter][[column,'grupo']]
	# print(len(df.loc[filter][[column,'grupo']]),'registros encontrados.')
	results = df.loc[filter][cols]
	print(len(df.loc[filter][cols]),'registros encontrados.')
	return results

def filtra_df_por_quantile(df,column,quantile_inferior,quantile_superior,dropna=True,cols_mostrar=None):
    preco = df[column].dropna()
    preco = preco[(preco > preco.quantile(quantile_inferior)) & (preco < preco.quantile(quantile_superior))]
    if cols_mostrar:
        return df.loc[preco.index][cols_mostrar]
    else:
        return df.loc[preco.index]

def calcula_cv_sentenca(df,doc_vectors):
	#comparacao do texto (vetor da sentenca):
	#soh entrar palavras treinadas (saem numeros)
	texto = doc_vectors.loc[df.index].mean(axis=1)
	texto_mean = texto.mean()
	if texto_mean == 0:
		return 1000
	else:
		return abs(texto.std() / texto.mean())

def calcula_cv_sentenca_cosine_distance(df,doc_vectors):
	media_do_grupo = doc_vectors.loc[df.index].mean(axis=0)
	distances = cosine_distances(doc_vectors.loc[df.index],media_do_grupo.values.reshape(1,-1))
	return (np.std(distances) / np.mean(distances)), distances

def calcula_cv_sentenca_v2(df,model):
    #comparacao do texto (vetor da sentenca):
    #agora entra tudo, numeros, etc:
    sentences = [sent.split() for sent in df['DS_ITEM_CLEAN']]
    #retira unidades:
    sentences = [[item for item in sentence if item not in unidades] for sentence in sentences]
    ultimo_registro = len(sentences)
    doc_vectors = {}
    for number, sent in enumerate(sentences[:ultimo_registro]):
        # doc_vectors[number] = doc2vec(model=model,doc=sent,size=qtd_dimensoes) #soh media dos vetores
        doc_vectors[number] = doc2vec(model=model,doc=sent,size=qtd_dimensoes,weights=None,remove_out_of_vocabulary_words=False)
    doc_vectors = DataFrame(doc_vectors).T
    texto = doc_vectors.mean(axis=1)
    return abs(texto.std() / texto.mean())

def calcula_cv_preco(df,column):
    preco = df[column].dropna()
    preco_mean = preco.mean()
    if preco_mean > 0:
        return abs(preco.std() / preco_mean)
    else:
        return 1000

def get_nome_representacao_do_grupo(df2):
	sentences = [sent for sent in df2['DS_ITEM_CLEAN']]
	sentences_set = set(sentences)
	if len(sentences_set) == 1: #ou seja, todos os itens sao iguais.
		representacao_grupo = [word for word in sentences_set][0].split()
	else:
		sentences2 = [sent.split() for sent in df2['DS_ITEM_CLEAN']]
		flat_sentences2 = [item for sublist in sentences2 for item in sublist]
		word_counter = collections.Counter(flat_sentences2)
		word_counter = Series(word_counter).sort_values(ascending=False)
		word_counter = word_counter.cumsum().pct_change()
		word_counter = word_counter.replace(np.nan,1)
		word_counter = word_counter[word_counter >= 0.15] #pra ignorar palavras inuteis.
		# word_counter = word_counter[word_counter >= 0.20] #pra ignorar palavras inuteis.
		# word_counter = word_counter[word_counter >= 0.05] #pra ignorar palavras inuteis.
		palavras_grupo = [word for word in word_counter.index]
		primeiras_palavras = [word for word in palavras_grupo if ((word not in unidades) and (not word.isdigit()))]
		if len(primeiras_palavras) > 1: #se tiver mais de uma palavra, inverte a ordem:
			primeira = primeiras_palavras[0]
			segunda = primeiras_palavras[1]
			if len(primeiras_palavras) > 2:
				primeiras_palavras = [segunda,primeira] + primeiras_palavras[2:]
			else:
				primeiras_palavras = [segunda,primeira]
		meio_palavras = [word for word in palavras_grupo if word.isdigit()]
		ultimas_palavras = [word for word in palavras_grupo if word in unidades]
		representacao_grupo = primeiras_palavras + meio_palavras + ultimas_palavras
		# representacao_grupo = []
		# for palavra in palavras_grupo:
		#     representacao_grupo.append(palavra)
		#     if palavra in unidades:
		#         break
	return representacao_grupo

def get_nome_representacao_do_grupo_v2(df2):
	sentences = [sent for sent in df2['DS_ITEM_CLEAN']]
	sentences_set = set(sentences)
	if len(sentences_set) == 1: #ou seja, todos os itens sao iguais.
		representacao_grupo = [word for word in sentences_set][0].split()
	else:
		sentences2 = [sent.split() for sent in df2['DS_ITEM_CLEAN']]
		flat_sentences2 = [item for sublist in sentences2 for item in sublist]
		word_counter = collections.Counter(flat_sentences2)
		word_counter = Series(word_counter).sort_values(ascending=False)
		word_counter = DataFrame(word_counter)
		word_counter.columns = ['word']
		word_counter['cumsum'] = word_counter.cumsum().pct_change().replace(np.nan,1)
		word_position = {}
		for word in word_counter.index:
			word_position[word] = [frase.index(word) for frase in [sentence for sentence in sentences2] if word in frase]
		word_position = Series(word_position)
		word_position = word_position.apply(lambda x: np.array(x)).apply(lambda x: np.mean(x))
		word_counter['position'] = word_position
		word_counter = word_counter.sort_values('position')

		word_counter = word_counter[word_counter['cumsum'] >= 0.15] #pra ignorar palavras inuteis.

		representacao_grupo = list(word_counter.index)

		# palavras_grupo = [word for word in word_counter.index]
		# primeiras_palavras = [word for word in palavras_grupo if ((word not in unidades) and (not word.isdigit()))]
		# if len(primeiras_palavras) > 1: #se tiver mais de uma palavra, inverte a ordem:
		# 	primeira = primeiras_palavras[0]
		# 	segunda = primeiras_palavras[1]
		# 	if len(primeiras_palavras) > 2:
		# 		primeiras_palavras = [segunda,primeira] + primeiras_palavras[2:]
		# 	else:
		# 		primeiras_palavras = [segunda,primeira]
		# meio_palavras = [word for word in palavras_grupo if word.isdigit()]
		# ultimas_palavras = [word for word in palavras_grupo if word in unidades]
		# representacao_grupo = primeiras_palavras + meio_palavras + ultimas_palavras
		
		# representacao_grupo = []
		# for palavra in palavras_grupo:
		#     representacao_grupo.append(palavra)
		#     if palavra in unidades:
		#         break
	return representacao_grupo

def get_nome_representacao_do_grupo_v3(df2,percentual_pra_considerar_com_numeros,qtd_primeiras_palavras,percentual_minimo_numeros_repetidos,qtd_palavras):
	#agora pega o item que mais se repetiu
	#se tem numeros, pros numeros entrarem na representacao tem que ter mais que 25%, senao pega soh as palavras ateh chegar nos numeros
	sentences = [sent for sent in df2['DS_ITEM_CLEAN']]
	sentences_set = set(sentences)
	if len(sentences_set) == 1: #ou seja, todos os itens sao iguais.
		representacao_grupo = [word for word in sentences_set][0].split()
	else:
		contagem = Series(sentences).value_counts()
		if (contagem.iloc[0] / contagem.sum()) > percentual_pra_considerar_com_numeros:
			representacao_grupo = contagem.index[0].split()[:qtd_primeiras_palavras]
		else:
			representacao_grupo = []
			numeros = [ [num for num in sent.split()[:qtd_palavras] if num.isdigit()] for sent in sentences]
			numeros_repetidos = Series(numeros).value_counts()
			for word in contagem.index[0].split()[:qtd_primeiras_palavras]:
				if word.isdigit():
					break
				else:
					representacao_grupo.append(word)
			
			if ( (len(numeros_repetidos.index[0]) > 0) and ((numeros_repetidos.iloc[0]/len(numeros)) > percentual_minimo_numeros_repetidos  ) ):
				for num in numeros_repetidos.index[0]:
					representacao_grupo.append(num)
	return representacao_grupo

def get_nome_representacao_do_grupo_v4(df2,qtd_palavras,percentual_pra_manter_palavra_na_representacao,unidades):
	#pega cada palavra e ve as que mais se repetem nas sentences
	#fica com aquelas que estao em mais do que X% das sentences
	sentences = [sent for sent in df2['DS_ITEM_CLEAN']]
	if len(set(sentences)) == 1: #ou seja, todos os itens sao iguais.
		representacao_grupo = [word for word in set(sentences)][0].split()
	else:
		palavras_series = df2['DS_ITEM_CLEAN'].str.split()
		palavras_series = palavras_series.apply(lambda x: x[:qtd_palavras])
		contagem_palavras_nas_sentences = {}
		# palavras = set([item for sublist in sentences for item in sublist.split()[:qtd_palavras]])
		palav = [item for sublist in palavras_series for item in sublist]
		palavras = sorted( set(palav), key=palav.index) #pra preservar a ordem em que as palavras aparecem (set puro coloca em ordem alfabetica)
		for palavra in palavras:
			contagem_palavras_nas_sentences[palavra] = (palavras_series.apply(lambda x: palavra in x)).sum()
		
		# contagem_palavras_nas_sentences = Series(contagem_palavras_nas_sentences).sort_values(ascending=False)
		contagem_palavras_nas_sentences = Series(contagem_palavras_nas_sentences)
		contagem_palavras_nas_sentences = contagem_palavras_nas_sentences / len(df2)
		contagem_palavras_nas_sentences = contagem_palavras_nas_sentences[contagem_palavras_nas_sentences > percentual_pra_manter_palavra_na_representacao]
		representacao_grupo = list(contagem_palavras_nas_sentences.index)
	#reordenacao:
	primeiras_palavras = [word for word in representacao_grupo if ((not word.isdigit()) and word not in unidades)]
	meio_palavras = [word for word in representacao_grupo if word.isdigit()]
	# ultimas_palavras = [word for word in representacao_grupo if ((word in unidades) and (word != 'x'))]
	ultimas_palavras = [word for word in representacao_grupo if word in unidades]
	# intercala numeros e unidades:
	if len(meio_palavras) == len(ultimas_palavras):
		result = [None]*(len(meio_palavras)+len(ultimas_palavras))
		result[::2] = meio_palavras
		result[1::2] = ultimas_palavras
		meio_palavras = result
		ultimas_palavras = []
	else:
		# if 'x' in representacao_grupo:
		if (('x' in ultimas_palavras) and (len(meio_palavras) > 0)):
			ultimas_palavras = [word for word in ultimas_palavras if word != 'x'] #retira o 'x', vai inserir abaixo:
			meio_palavras.insert(1,'x') #insere o 'x' apos o 1o numero
	if len(meio_palavras) == 0:
		representacao_grupo = primeiras_palavras #daih nao coloca unidades
	else:
		representacao_grupo = primeiras_palavras + meio_palavras + ultimas_palavras
	return representacao_grupo

def get_nome_representacao_do_grupo_v5(df2,qtd_palavras,percentual_pra_manter_palavra_na_representacao,unidades):
	#pega cada palavra e ve as que mais se repetem nas sentences
	#fica com aquelas que estao em mais do que X% das sentences
	sentences = [sent for sent in df2['DS_ITEM_CLEAN']]
	if len(set(sentences)) == 1: #ou seja, todos os itens sao iguais.
		representacao_grupo = [word for word in set(sentences)][0].split()
	else:
		palavras_series = df2['DS_ITEM_CLEAN'].str.split()
		palavras_series = palavras_series.apply(lambda x: x[:qtd_palavras])
		contagem_palavras_nas_sentences = {}
		# palavras = set([item for sublist in sentences for item in sublist.split()[:qtd_palavras]])
		palav = [item for sublist in palavras_series for item in sublist]
		palavras = sorted( set(palav), key=palav.index) #pra preservar a ordem em que as palavras aparecem (set puro coloca em ordem alfabetica)
		for palavra in palavras:
			contagem_palavras_nas_sentences[palavra] = (palavras_series.apply(lambda x: palavra in x)).sum()
		
		# contagem_palavras_nas_sentences = Series(contagem_palavras_nas_sentences).sort_values(ascending=False)
		contagem_palavras_nas_sentences = Series(contagem_palavras_nas_sentences)
		contagem_palavras_nas_sentences = contagem_palavras_nas_sentences / len(df2)
		contagem_palavras_nas_sentences = contagem_palavras_nas_sentences[contagem_palavras_nas_sentences > percentual_pra_manter_palavra_na_representacao]
		representacao_grupo = list(contagem_palavras_nas_sentences.index)
	#reordenacao:
	primeiras_palavras = [word for word in representacao_grupo if ((not word.isdigit()) and word not in unidades)]
	meio_palavras = [word for word in representacao_grupo if word.isdigit()]
	# ultimas_palavras = [word for word in representacao_grupo if ((word in unidades) and (word != 'x'))]
	ultimas_palavras = [word for word in representacao_grupo if word in unidades]
	# intercala numeros e unidades:
	if len(meio_palavras) == len(ultimas_palavras):
		result = [None]*(len(meio_palavras)+len(ultimas_palavras))
		result[::2] = meio_palavras
		result[1::2] = ultimas_palavras
		meio_palavras = result
		ultimas_palavras = []
	else:
		# if 'x' in representacao_grupo:
		if (('x' in ultimas_palavras) and (len(meio_palavras) > 0)):
			ultimas_palavras = [word for word in ultimas_palavras if word != 'x'] #retira o 'x', vai inserir abaixo:
			meio_palavras.insert(1,'x') #insere o 'x' apos o 1o numero
	if len(meio_palavras) == 0:
		representacao_grupo = primeiras_palavras #daih nao coloca unidades
	else:
		representacao_grupo = primeiras_palavras + meio_palavras + ultimas_palavras
	
	df2['DS_ITEM_CORTE'] = df2['DS_ITEM_CLEAN'].apply(lambda x: x.split())
	df2['DS_ITEM_CORTE'] = df2['DS_ITEM_CORTE'].apply(lambda x: x[:qtd_palavras])
	
	mais_repetido = df2['DS_ITEM_CORTE'].value_counts().index[0]
	
	return (representacao_grupo, mais_repetido)

def descobre_grupos_por_palavra_na_representacao(df,word,representacoes,grupo='grupo5'):
	df_result = Series({grupo:representacao for grupo, representacao in zip(representacoes.keys(), representacoes.values()) if word in representacao})
	df_result = DataFrame(df_result)
	df_result.columns = ['representacao']
	df_result['qtd sentencas'] = df[grupo].value_counts().loc[df_result.index]
	df_result = df_result.sort_values('qtd sentencas',ascending=False)
	return df_result

def descobre_grupos_por_palavras_na_representacao(df,words,representacoes,grupo='grupo5'):
	df_result = Series({grupo:representacao for grupo, representacao in zip(representacoes.keys(), representacoes.values()) if all([True if word in representacao else False for word in words]) })
	df_result = DataFrame(df_result)
	df_result.columns = ['representacao']
	df_result['qtd sentencas'] = df[grupo].value_counts().loc[df_result.index]
	df_result = df_result.sort_values('qtd sentencas',ascending=False)
	return df_result

def encontra_grupo_new_sentence(new_sentence,model,qtd_dimensoes,scaler,pca,umap_redux,grupos_finais_vectors_std_pca_umap,representacoes):
	#retira pontuacao, /, etc:
	new_sentence = new_sentence.translate(str.maketrans('', '', string.punctuation))
	#passa pra minusculo:
	new_sentence = new_sentence.lower()
	#insere espaco apos numero e letra (separa unidades de medida:) ex.: 500ml vs 100ml vs 500mg
	new_sentence = re.sub(r'(\d{1})(\D)',r'\1 \2',new_sentence)
	#insere espaco apos letra e numero ex.:c100 pc50
	new_sentence = re.sub(r'(\D{1})(\d)',r'\1 \2',new_sentence)
	#retira espacos duplicados
	new_sentence = re.sub(r' +',r' ',new_sentence)
	#limpa sentencas retirando stopwords (tem que ser minusculo) e pontuacao.
	new_sentence = [word for word in new_sentence.split() if word.lower() not in stopwords]
	#retira acentos:
	new_sentence = [unidecode.unidecode(word) for word in new_sentence]
	# #retira unidades:
	new_sentence = [item for item in new_sentence if item not in unidades]
	# #retira numeros:
	# new_sentence = [item for item in new_sentence if not item.isdigit()]
	new_sentence_vectors = doc2vec(model=model,doc=new_sentence,size=qtd_dimensoes)
	# new_sentence_vectors = doc2vec(model=model,doc=new_sentence,size=qtd_dimensoes,weights=weights,remove_out_of_vocabulary_words=True) #agora fazendo media ponderada pelos pesos do tfidf: media dos vetores.
	new_sentence_vectors = new_sentence_vectors.reshape(1,-1)
	new_sentence_vectors_std = scaler.transform(new_sentence_vectors)
	new_sentence_vectors_std_pca = pca.transform(new_sentence_vectors_std)
	new_sentence_vectors_std_pca_umap = umap_redux.transform(X=new_sentence_vectors_std_pca)
	grupo_mais_parecido = cosine_similarity(X=new_sentence_vectors_std_pca_umap.reshape(1,-1),Y=grupos_finais_vectors_std_pca_umap.values)
	grupo_mais_parecido = DataFrame(grupo_mais_parecido[0])
	grupo_mais_parecido.columns = ['cosine_similarity']
	grupo_mais_parecido['grupo'] = [ grupos_finais_vectors_std_pca_umap.index[i] for i in grupo_mais_parecido.index]
	grupo_mais_parecido = grupo_mais_parecido.sort_values('cosine_similarity',ascending=False)
	grupo_mais_parecido = grupo_mais_parecido.set_index('grupo')
	grupo_mais_parecido['representacao'] = Series(representacoes)

	return grupo_mais_parecido

def encontra_grupo_new_sentence_sem_dim_redux(new_sentence,model,qtd_dimensoes,grupos_finais_vectors_std_pca_umap,representacoes,stopwords,unidades):
	#retira pontuacao, /, etc:
	new_sentence = new_sentence.translate(str.maketrans('', '', string.punctuation))
	#passa pra minusculo:
	new_sentence = new_sentence.lower()
	#insere espaco apos numero e letra (separa unidades de medida:) ex.: 500ml vs 100ml vs 500mg
	new_sentence = re.sub(r'(\d{1})(\D)',r'\1 \2',new_sentence)
	#insere espaco apos letra e numero ex.:c100 pc50
	new_sentence = re.sub(r'(\D{1})(\d)',r'\1 \2',new_sentence)
	#retira espacos duplicados
	new_sentence = re.sub(r' +',r' ',new_sentence)
	#limpa sentencas retirando stopwords (tem que ser minusculo) e pontuacao.
	new_sentence = [word for word in new_sentence.split() if word.lower() not in stopwords]
	#retira acentos:
	new_sentence = [unidecode.unidecode(word) for word in new_sentence]
	# #retira unidades:
	new_sentence = [item for item in new_sentence if item not in unidades]
	# #retira numeros:
	# new_sentence = [item for item in new_sentence if not item.isdigit()]
	new_sentence_vectors = doc2vec(model=model,doc=new_sentence,size=qtd_dimensoes,remove_out_of_vocabulary_words=True)
	# new_sentence_vectors = doc2vec(model=model,doc=new_sentence,size=qtd_dimensoes,weights=weights,remove_out_of_vocabulary_words=True) #agora fazendo media ponderada pelos pesos do tfidf: media dos vetores.

	# new_sentence_vectors = new_sentence_vectors.reshape(1,-1)
	# new_sentence_vectors_std = scaler.transform(new_sentence_vectors)
	# new_sentence_vectors_std_pca = pca.transform(new_sentence_vectors_std)
	# new_sentence_vectors_std_pca_umap = umap_redux.transform(X=new_sentence_vectors_std_pca)

	grupo_mais_parecido = cosine_similarity(X=new_sentence_vectors.reshape(1,-1),Y=grupos_finais_vectors_std_pca_umap.values)
	grupo_mais_parecido = DataFrame(grupo_mais_parecido[0])
	grupo_mais_parecido.columns = ['cosine_similarity']
	grupo_mais_parecido['grupo'] = [ grupos_finais_vectors_std_pca_umap.index[i] for i in grupo_mais_parecido.index]
	grupo_mais_parecido = grupo_mais_parecido.sort_values('cosine_similarity',ascending=False)
	grupo_mais_parecido = grupo_mais_parecido.set_index('grupo')
	grupo_mais_parecido['representacao'] = Series(representacoes)

	return grupo_mais_parecido

def retorna_df_filtrado(df_reset,grupo,data_inicial=None,data_final=None,qtd_minima=None,unidade_medida=None):
    df2 = df_reset[df_reset['grupo10'] == grupo]
    if qtd_minima:
        df2 = df2[df2['QT_ITENS'] > qtd_minima]
    if data_inicial:
        df2 = df2[df2['DT_ABERTURA'] > data_inicial]
    if data_final:
        df2 = df2[df2['DT_ABERTURA'] < data_inicial]
    if unidade_medida:
        df2 = df2[df2['SG_UNIDADE_MEDIDA'] == unidade_medida]
    return df2

def encontra_unidades_medida(df2):
    return list(df2['SG_UNIDADE_MEDIDA'].unique())

def retorna_estatisticas(df2,cv_limite):
    stats = Series()
    stats['Qtd original itens'] = len(df2)
    stats['Media original'] = round(df2['ITEM_VL_UNITARIO_HOMOLOGADO'].mean(),4)
    stats['Mediana original'] = round(df2['ITEM_VL_UNITARIO_HOMOLOGADO'].median(),4)
    stats['Desvio padrao original'] = round(df2['ITEM_VL_UNITARIO_HOMOLOGADO'].std(),4)
    stats['CV original'] = round(calcula_cv_preco(df=df2,column='ITEM_VL_UNITARIO_HOMOLOGADO'),4)
    if stats['CV original'] <= cv_limite:
        return stats, df2
    else: #media saneada:
        preco_limpo = df2['ITEM_VL_UNITARIO_HOMOLOGADO'][df2['ITEM_VL_UNITARIO_HOMOLOGADO'] > 0] #soh considera preco positivo pra calcular o cv_preco, tambem tirar nans
        preco_limpo = preco_limpo.dropna()
        cv_preco = stats['CV original']

        quantile_superior = 0.95
        quantile_inferior = 0.05

        while cv_preco > cv_limite:
            q_sup = preco_limpo.quantile(quantile_superior)
            q_inf = preco_limpo.quantile(quantile_inferior)
            preco_limpo = preco_limpo[((preco_limpo > q_inf) & (preco_limpo < q_sup))]
            if len(preco_limpo) <= 2:
                break
            cv_preco = variation(preco_limpo, axis = 0)

        stats['-----'] = '-----'
        stats['Qtd saneada itens'] = len(preco_limpo)
        stats['Media saneada'] = round(preco_limpo.mean(),4)
        stats['Mediana saneada'] = round(preco_limpo.median(),4)
        stats['Desvio padrao saneada'] = round(preco_limpo.std(),4)
        stats['CV saneada'] = round(cv_preco,4)

        return stats, df2.loc[preco_limpo.index]

def vectorized_cosine_distance_cupy(matrix,norm_matrix,vector):
    'returns cosine distance between a vector and all elements of a matrix'

    # matrix = cp.array(matrix).astype(np.float32))
    # vector = cp.array(vector).astype(np.float32))

    dot_product = cp.dot(matrix,vector)

    norm_vector = cp.linalg.norm(vector,axis=0)

    # return cp.asnumpy(1 - (dot_product / (norm_matrix * norm_vector)))
    return 1 - (dot_product / (norm_matrix * norm_vector))

def cosine_distance_similarity(a, b, mode='distance'):
    """Takes 2 vectors a, b and returns the cosine similarity/distance according to the definition of the dot product
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if mode == 'similarity':
        return dot_product / (norm_a * norm_b)
    elif mode == 'distance':
        return 1 - (dot_product / (norm_a * norm_b))

def cosine_distance_similarity_cupy(a, b, mode='distance'):
    """Takes 2 vectors a, b and returns the cosine similarity/distance according to the definition of the dot product
    """
    dot_product = cp.dot(a, b)
    norm_a = cp.linalg.norm(a)
    norm_b = cp.linalg.norm(b)
    if mode == 'similarity':
        return dot_product / (norm_a * norm_b)
    elif mode == 'distance':
        return 1 - (dot_product / (norm_a * norm_b))

def vectorized_cosine_distance(matrix,vector):
    'returns cosine distance between a vector and all elements of a matrix'

    dot_product = np.dot(matrix,vector)

    norm_matrix = np.linalg.norm(matrix,axis=1)
    norm_vector = np.linalg.norm(vector,axis=0)

    return 1 - (dot_product / (norm_matrix * norm_vector))

def print_qtd_grupos_sentencas_uteis(df,grupos,grupox):
    print('% de sentencas uteis:', (df[grupox]>=0).sum() / len(df) )
    print('qtd de grupos:',len(grupos))

def print_exemplos_grupos(df,inicio,fim,grupos,grupox,cols,qtd_palavras,percentual_pra_manter_palavra_na_representacao,unidades):
    for grupo in grupos[inicio:fim]:
        df_mostrar = df[df[grupox] == grupo]
        print('\nGrupo:',grupo,'len:',len(df_mostrar))
        print(get_nome_representacao_do_grupo_v4(df2=df[df[grupox]==grupo],qtd_palavras=qtd_palavras,percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,unidades=unidades) )
        print(df_mostrar[cols])

def print_exemplos_grupos_v2_aleatorio(df,qtd_grupos_mostrar,grupos,grupox,cols,qtd_palavras,percentual_pra_manter_palavra_na_representacao,unidades):
	inicio = np.random.randint(len(grupos)-qtd_grupos_mostrar-1)
	fim = inicio + qtd_grupos_mostrar
	for grupo in grupos[inicio:fim]:
		df_mostrar = df[df[grupox] == grupo]
		print('\nGrupo:',grupo,'len:',len(df_mostrar))
		print(get_nome_representacao_do_grupo_v4(df2=df[df[grupox]==grupo],qtd_palavras=qtd_palavras,percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,unidades=unidades) )
		print(df_mostrar[cols])

def registro_aleatorio(df,stopwords,model,cols_mostrar,cv_limite,qtd_desvios,data_inicial,data_final,qtd_minima,unidade_medida,doc_vectors_grupos_finais,qtd_palavras,minimo_cosine_similarity):

    #Escolhe descricao aleatoria da base - filtra jah por unidade de medida do licitacon:
    random_number = np.random.randint(len(df))
    sentence_aleatoria = df['DS_ITEM_CLEAN'].iloc[random_number]
    unidade_aleatoria = df['SG_UNIDADE_MEDIDA'].iloc[random_number]
    preco_aleatorio = df['ITEM_VL_UNITARIO_HOMOLOGADO'].iloc[random_number]

    # print('\nDescricao:',sentence_aleatoria)
    # print('Unidade:',unidade_aleatoria)

    new_sentence = ' '.join([word for word in sentence_aleatoria.split() if word.lower() not in stopwords])
    #insere espaco apos / e -, pra no final nao ficar palavras assim: csolucao, ptexto (originais eram c/solucao, p-texto)
    new_sentence = re.sub(r'/|-',r' ',new_sentence)
    #retira pontuacao, /, etc:
    new_sentence = new_sentence.translate(str.maketrans('', '', string.punctuation))
    #passa pra minusculo:
    new_sentence = new_sentence.lower()
    #insere espaco apos numero e letra (separa unidades de medida:) ex.: 500ml vs 100ml vs 500mg
    new_sentence = re.sub(r'(\d{1})(\D)',r'\1 \2',new_sentence)
    #insere espaco apos letra e numero ex.:c100 pc50
    new_sentence = re.sub(r'(\D{1})(\d)',r'\1 \2',new_sentence)
    #retira espacos duplicados
    new_sentence = re.sub(r' +',r' ',new_sentence)
    #retira acentos:
    new_sentence = unidecode.unidecode(new_sentence)

    # media simples, sem pesos:
    # new_sentence_vector = np.array([model.wv[item] for item in new_sentence.split()]).mean(axis=0)

    new_sentence = new_sentence.split()[:qtd_palavras]

    # com pesos:
    if len(new_sentence) == 0:
        new_sentence_vector = np.zeros(qtd_dimensoes,)
    elif len(new_sentence) == 1:
        new_sentence_vector = model.wv[new_sentence[0]]
    elif len(new_sentence) > 1:
        pesos = np.array(range(len(new_sentence))[::]) + 1
        pesos = 1 / pesos # agora com pesos 1/x - tem que ser na ordem 1,2,..., os menores numeros dao maiores pesos - decai menos que exponencial, que eh muito brusca a queda.
        media = []
        divisao = 0
        counter = 0
        for word in new_sentence:
            if word.isdigit():
                media.append(model.wv[word] * ((pesos[0]+pesos[-1])*(1/4)) ) #nova abordagem: se eh digit, atribui peso NO 3/4 da faixa entre o peso da primeira e da ultima palavra. Mesmo peso pra todos os numeros, mais importante que palavras do fim, menos importante que palavras do inicio.
                divisao += ((pesos[0]+pesos[-1])*(1/4))
            else:
                media.append(model.wv[word] * pesos[counter])
                divisao += pesos[counter]
            counter += 1
        new_sentence_vector = np.array(media).sum(axis=0) / divisao #media de tudo

    grupo_mais_parecido = cosine_similarity(X=new_sentence_vector.reshape(1,-1),Y=doc_vectors_grupos_finais.values)
    grupo_mais_parecido = DataFrame(grupo_mais_parecido[0])
    grupo_mais_parecido.columns = ['cosine_similarity']
    grupo_mais_parecido['grupo'] = [ doc_vectors_grupos_finais.index[i] for i in grupo_mais_parecido.index]
    grupo_mais_parecido = grupo_mais_parecido.sort_values('cosine_similarity',ascending=False)
    grupo_mais_parecido = grupo_mais_parecido.set_index('grupo')

    if (len(grupo_mais_parecido[grupo_mais_parecido['cosine_similarity'] > minimo_cosine_similarity]) == 0):
        # print('\nNao encontrado grupo de produtos similar, especifique melhor a descricao do produto.')
        return False, None, None, None, sentence_aleatoria, unidade_aleatoria, preco_aleatorio
    else:
        grupo = grupo_mais_parecido[grupo_mais_parecido['cosine_similarity'] > minimo_cosine_similarity].index[0]
        
        df2 = df[df['grupo10'] == grupo]
        
        if qtd_minima:
            df2 = df2[df2['QT_ITENS'] > qtd_minima]
        if data_inicial:
            df2 = df2[df2['DT_ABERTURA'] > data_inicial]
        if data_final:
            df2 = df2[df2['DT_ABERTURA'] < data_inicial]
        if unidade_medida:
            df2 = df2[df2['SG_UNIDADE_MEDIDA'] == unidade_medida]

        if len(df2) == 0:
            print('\nNao encontrado registros com a mesma unidade de medida e/ou quantidade minima e/ou nas datas indicadas.')

        else:
            stats, registros = retorna_estatisticas(df2,cv_limite=cv_limite)
            # print(stats)
            if 'Media saneada' in stats:
                if preco_aleatorio > (stats['Media saneada'] + (qtd_desvios * stats['Desvio padrao saneada']) ):
                    mensagem = 'Preco da licitacao: R$ '+ str(preco_aleatorio) + '. ALERTA SOBREPRECO.'
                else:
                    mensagem = 'Preco da licitacao: R$ '+ str(preco_aleatorio) + '. Preco NORMAL, dentro de '+ str(qtd_desvios) + ' desvios padroes.'
            else:
                if preco_aleatorio > (stats['Media original'] + (qtd_desvios * stats['Desvio padrao original']) ):
                    mensagem = 'Preco da licitacao: R$ '+ str(preco_aleatorio) + '. ALERTA SOBREPRECO.'
                else:
                    mensagem = 'Preco da licitacao: R$ '+ str(preco_aleatorio) + '. Preco NORMAL, dentro de '+ str(qtd_desvios) + ' desvios padroes.'
            # print('\nRegistros:')
            # print(registros[cols_mostrar])
        return True, stats, registros[cols_mostrar], mensagem, sentence_aleatoria, unidade_aleatoria, preco_aleatorio