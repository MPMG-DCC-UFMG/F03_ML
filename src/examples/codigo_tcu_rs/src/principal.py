##################
#TREINO DO MODELO:
##################

# Se estiver usando o Ipython/Jupyter notebook:
%matplotlib
import os
folder = 'C:/banco-de-precos-licitacon/' # Inserir aqui a pasta local do repositorio:
os.chdir(folder)
from src.funcoes import * # Importa funcoes

cols_mostrar = ['DS_ITEM_CLEAN','SG_UNIDADE_MEDIDA','QT_ITENS','ITEM_VL_UNITARIO_HOMOLOGADO']

# PARAMETROS:
tamanho_minimo_pra_formar_grupo = 30
qtd_palavras = 16
qtd_dimensoes = 300
qtd_dimensoes_umap = 15
quantile_a_retirar_outliers_dbscan = 0.95
minimo_cosine_similarity = 0.9
qtd_ngram_FastText = 3
cv_maximo_pra_considerar_grupo_homogeneo_sentenca = 1.75
percentual_primeira_palavra_igual_pra_considerar_grupo_homogeneo = 0.50
percentual_pra_manter_palavra_na_representacao = 0.50
qtd_min_auditadas_para_formar_grupo = 10
quantile_a_retirar_numeros_diferentes_no_grupo = 0.95
quantile_a_retirar_quantidade_palavras_diferentes_no_grupo = 0.95

# Descobre arquivos de dados:
pasta = './data/'
arquivos = descobre_arquivos_na_pasta(pasta,tipo_do_arquivo='.xlsx')
arquivos.remove('Tabelas Auxiliares.xlsx')

# LEITURA INICIAL - demorado - le arquivos e concatena registros em um dataframe:
lista = []
for arquivo in tqdm(arquivos):
    df = pd.read_excel(pasta + arquivo,parse_dates=['DT_ABERTURA','DT_REF_VALOR_ESTIMADO','ANO_LICITACAO'],dtype={'NOME':str,'SERVICO_AUDITORIA':str,'NATUREZA':str,'NATUREZA_DESPESA':str,'DS_OBJETO':str,'DS_LOTE':str,'DS_ITEM':str,'SG_UNIDADE_MEDIDA':str,'DS_FONTE_REFERENCIA':str,'CD_FONTE_REFERENCIA':str,'TP_OBJETO':str,'DS_TP_OBJETO':str,'CD_TIPO_MODALIDADE':str,'DS_TIPO_MODALIDADE':str,'NR_LICITACAO':str,'NOME_MUNICIPIO':str,'ESFERA':str})
    lista.append(df)
for item in tqdm(range(len(lista))):
    lista[item]['DT_ABERTURA'] = pd.to_datetime(lista[item]['DT_ABERTURA'],dayfirst=True,errors='coerce')
    # https://stackoverflow.com/questions/32888124/pandas-out-of-bounds-nanosecond-timestamp-after-offset-rollforward-plus-adding-a
licitacon = pd.concat(lista,ignore_index=True)
licitacon = licitacon.set_index('ID_ITEM')
grava(licitacon,'licitacon_v8.pkl',pasta)

licitacon = abre('licitacon_v8.pkl',pasta)

#deixa somente colunas necessarias:
licitacon = licitacon[['ID_LICITACAO', 'NOME', 'SERVICO_AUDITORIA','NATUREZA_DESPESA', 'DS_ITEM', 'QT_ITENS', 'SG_UNIDADE_MEDIDA', 'ITEM_VL_UNITARIO_HOMOLOGADO', 'DT_ABERTURA', 'TP_OBJETO', 'DS_TP_OBJETO', 'CD_TIPO_MODALIDADE', 'DS_TIPO_MODALIDADE', 'NR_LICITACAO', 'ANO_LICITACAO', 'CD_MUNICIPIO_IBGE', 'NOME_MUNICIPIO', 'ESFERA']]

licitacon['NATUREZA_DESPESA'] = licitacon['NATUREZA_DESPESA'].astype(str)

#pra fazer download dos corpus/stopwords:
nltk.download()
nltk.download('punkt')

# stopwords sao somente punctuation. O resto DEIXO, tem palavras importantes pros produtos: com/sem/tem/nem, etc.
stopwords = set(list(punctuation))

unidades = ['x','mm','m','cm','ml','g','mg','kg','unidade','unidades','polegada','polegadas','grama','gramas','gb','mb','l','litro','litros','mts','un','mgml','w','hz','v','gr','lt','lts','lonas','cores','mcg']
primeira_palavra_generica = ['caixa','jogo','kit','conjunto','item','it','cjt','conj','conjt','jg','kt','de','para']

#limpa sentencas retirando stopwords (tem que ser minusculo) e pontuacao.
licitacon['DS_ITEM_CLEAN'] = [ ' '.join([word for word in item.split() if word.lower() not in stopwords]) for item in licitacon['DS_ITEM'].astype(str) ]
#insere espaco apos / e -, pra no final nao ficar palavras assim: csolucao, ptexto (originais eram c/solucao, p-texto)
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: re.sub(r'/|-',r' ',x))
#retira pontuacao, /, etc:
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
#passa pra minusculo:
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: x.lower())
#insere espaco apos numero e letra (separa unidades de medida:) ex.: 500ml vs 100ml vs 500mg
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: re.sub(r'(\d{1})(\D)',r'\1 \2',x))
#insere espaco apos letra e numero ex.:c100 pc50
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: re.sub(r'(\D{1})(\d)',r'\1 \2',x))
#retira espacos duplicados
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: re.sub(r' +',r' ',x))
#retira acentos:
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: unidecode.unidecode(x))
#se primeira palavra for numero, joga pro final (caso de numeros de referencia que colocam no inicio)
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: ' '.join(x.split()[1:] + [x.split()[0]]) if ((len(x) > 1) and (x.split()[0].isdigit()) ) else x)
# remove zeros a esquerda de numeros (02 litros, 05, etc.)
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: ' '.join([word.lstrip('0') for word in x.split()] ) )
# remove 'x', pra não diferenciar pneu 275 80 de 275 x 80:
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: ' '.join([word for word in x.split() if word is not 'x']))

#retira primeira palavra se estah em unidades ou primeira_palavra_generica:
# roda varias vezes pra tirar todas as primeiras palavras:
for _ in range(3):
    licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: ' '.join(x.split()[1:]) if (len(x) > 1 and (x.split()[0] in unidades or x.split()[0] in primeira_palavra_generica)) else x )

#mostra registros aleatorios:
# df['DS_ITEM_CLEAN'].iloc[np.random.randint(len(df),size=50)]

#a partir de agora filtra soh pra material de consumo OU Equipamentos e Material Permanente:
df = licitacon.copy()
df = df[df['NATUREZA_DESPESA'].apply(lambda x: x.endswith('30') or x.endswith('52'))]

# retira livros:
df = df[df['DS_ITEM_CLEAN'].apply(lambda x: 'livro' not in x)]

#limito a 16 palavras - pega 85% dos itens até 16 palavras, o resto é lixo, tem item com mais de 300 palavras.
sentences = [sent.split()[:qtd_palavras] for sent in df['DS_ITEM_CLEAN']]

# exploracao/descobre se tem palavra:
# guindaste = df['DS_ITEM_CLEAN'].str.contains('guindaste', regex=False)
# guindastes = df['DS_ITEM_CLEAN'].loc[guindaste[guindaste == True].index]
# guindastes.to_csv(folder+'guindastes.csv')

print('Treino word2vec/fastText word embeddings gensim:')
model = FastText(sentences,size=qtd_dimensoes, min_count=tamanho_minimo_pra_formar_grupo, workers=-1, min_n=qtd_ngram_FastText, max_n=qtd_ngram_FastText, iter=10)
# grava(coisa=model,filename='model_20190826.pkl',path=folder)
# grava_arquivo_grande(coisa=model,filename='model_20190728.pkl',path=folder)

print('Conversao word embeddings to sentence embedding, com pesos:')
doc_vectors = {}
for number, sent in enumerate(tqdm(sentences)):
    # dando peso maior pras primeiras palavras, peso decrescente ateh o final, numeros com mesmo peso da primeira palavra:
    if len(sent) == 0:
        doc_vectors[number] = np.zeros(qtd_dimensoes,)
    elif len(sent) == 1:
        doc_vectors[number] = model[sent[0]]
    elif len(sent) > 1:
        pesos = np.array(range(len(sent))[::]) + 1
        pesos = 1 / pesos # agora com pesos 1/x - tem que ser na ordem 1,2,..., os menores numeros dao maiores pesos - decai menos que exponencial, que eh muito brusca a queda.
        media = []
        divisao = 0
        counter = 0
        for word in sent:
            if word.isdigit():
                media.append(model.wv[word] * ((pesos[0]+pesos[-1])*(1/4)) ) #nova abordagem: se eh digit, atribui peso NO 3/4 da faixa entre o peso da primeira e da ultima palavra. Mesmo peso pra todos os numeros, mais importante que palavras do fim, menos importante que palavras do inicio.
                divisao += ((pesos[0]+pesos[-1])*(1/4))
            else:
                media.append(model.wv[word] * pesos[counter])
                divisao += pesos[counter]
            counter += 1
        doc_vectors[number] = np.array(media).sum(axis=0) / divisao #media de tudo

doc_vectors = DataFrame(doc_vectors).T
doc_vectors = doc_vectors.set_index(df.index) # coloca o ID_ITEM como index:

print('StandardScaler:')
scaler = StandardScaler()
doc_vectors_std_df = DataFrame(scaler.fit_transform(doc_vectors),index=doc_vectors.index,columns=doc_vectors.columns)

# descobrir quais as maiores quantidades de finais de natureza de despesa:
# df['final'] = df['NATUREZA_DESPESA'].apply(lambda x: x[-2:])
# df.groupby('final')['DS_ITEM_CLEAN'].count().sort_values(ascending=False)

# Reduz dims com UMAP - DEMORA - Reduz pra 15 dimns com UMAP direto das 300 dimns, sem PCA:
print('Reduz com UMAP pra', str(qtd_dimensoes_umap),'dimensoes.')
# agora com metric = 'cosine'. desempenho colocando init='random' piora, mas eh mais rapido.
umap_redux = umap.UMAP(n_components=qtd_dimensoes_umap, random_state=999, metric='cosine',verbose=True)
umap_redux.fit(doc_vectors_std_df) # soh material de consumo: 16h / soh nao material de consumo: 16h / material de consumo + equipamentos e material permantente (853 mil rows): 23h. / nova extracao 906 mil linhas (mat consumo, permanente, equipamentos): 31h, mas deu warning: WARNING: spectral initialisation failed! The eigenvector solver failed. This is likely due to too small an eigengap. Consider adding some noise or jitter to your data. Falling back to random initialisation! "WARNING: spectral initialisation failed! The eigenvector solver" / nova extracao v8 licitacon mat consumo/permanente/equipamentos (919 mil linhas): 38h.

# umap_redux = abre(filename='umap_redux_20190820.pkl', path=folder)
umap_redux = abre(filename='umap_redux_20190823.pkl', path=folder)
# grava(coisa=umap_redux,filename='umap_redux_20190823.pkl', path=folder)
# grava(coisa=umap_redux,filename='umap_redux_20190826.pkl', path=folder)

doc_vectors_std_df_umap = umap_redux.transform(X=doc_vectors_std_df)

# HDBSCAN clustering:
print('Clusterizando, tamanho minimo pra formar grupo:', str(tamanho_minimo_pra_formar_grupo))
min_samples = 1
clustering = hdbscan.HDBSCAN(min_cluster_size=tamanho_minimo_pra_formar_grupo,min_samples=min_samples,prediction_data=True,core_dist_n_jobs=-1)
clustering.fit(doc_vectors_std_df_umap)

df['grupo'] = clustering.labels_

# atribui -2 aos outliers pelo HDBSCAN:
threshold = pd.Series(clustering.outlier_scores_).quantile(quantile_a_retirar_outliers_dbscan)
outliers = np.where(clustering.outlier_scores_ > threshold)[0]
df.iloc[outliers,df.columns.get_loc('grupo')] = -2

grupos = np.unique(clustering.labels_)
grupos = [grupo for grupo in grupos if grupo >= 0]

# print_exemplos_grupos(df=df,inicio=1000,fim=1050,grupos=grupos,grupox='grupo',cols=cols_mostrar+['NOME'],qtd_palavras=qtd_palavras,percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,unidades=unidades)
print_qtd_grupos_sentencas_uteis(df=df,grupos=grupos,grupox='grupo')

#########################################
# EXCLUI sentences outliers pelo texto pelos exemplars - DEMORA
# FAZ cosine distance pra cada sentence dentro dos grupos
#########################################

print('EXCLUI sentences outliers pelo texto pelos exemplars:')

#pontos mais representativos de cada grupo (como hdbscan aceita formato aleatorio de cada cluster, nao tem como passar/nao existe um centroide, como no k-means, que assume grupos esfericos)
exemplars = []
for exemplar in tqdm(clustering.exemplars_):
    exemplars.append(np.mean(exemplar,axis=0))
exemplars_df = DataFrame(exemplars,index=range(len(grupos)))

map_grupos_exemplars = {}
df_temp = DataFrame(columns=['sims'])

for grupo in tqdm(grupos[:]):

    df2 = df[df['grupo'] == grupo]
    indexes = df2.index
    grupo_vectors = DataFrame(doc_vectors_std_df_umap,index=df.index).loc[indexes]
    
    grupo_do_exemplar = Series(cosine_similarity(grupo_vectors.mean(axis=0).values.reshape(1,-1),exemplars_df)[0]).sort_values(ascending=False).index[0]
    map_grupos_exemplars[grupo] = grupo_do_exemplar

    sims = cosine_similarity(grupo_vectors,exemplars[grupo_do_exemplar].reshape(1,-1))

    df2['sims'] = sims
    df_temp = df_temp.append(df2[['sims']])

#passa resultados pro df principal:
df['sims'] = df_temp
df['sims'] = df['sims'].replace(np.nan,-1)

#retira quem tem similaridade negativa - eh um bom parametro.
df['grupo2'] = np.where(df['sims'] < 0, -1, df['grupo'])

grupos = df['grupo2'].unique()
grupos = [grupo for grupo in grupos if grupo >= 0]

# print_exemplos_grupos(df=df,inicio=1500,fim=1600,grupos=grupos,grupox='grupo2',cols=cols_mostrar+['NOME'],qtd_palavras=qtd_palavras,percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,unidades=unidades)
print_qtd_grupos_sentencas_uteis(df=df,grupos=grupos,grupox='grupo2')

#######################
# Reduzi um passo aqui:
df['grupo3'] = df['grupo2']
#######################

##########################################
# Se a 1a palavra nao for a mesma em X% do grupo, exclui o grupo, eh muito heterogeneo: - eh RAPIDO
##########################################

print('Se a 1a palavra nao for a mesma em X% do grupo, exclui o grupo, eh muito heterogeneo:')

grupos = sorted(df['grupo3'].unique())
grupos = [grupo for grupo in grupos if grupo >=0] #tirar os -1, -2, etc.

grupos_homogeneos = []

for grupo in tqdm(grupos):
    df2 = df[df['grupo3'] == grupo]
    if len(df2) > 0:
        if ( df2['DS_ITEM_CLEAN'].apply(lambda x: x.split()[0] if (len(x.split()) > 0) else np.random.random()).value_counts().iloc[0] / len(df2) ) > percentual_primeira_palavra_igual_pra_considerar_grupo_homogeneo:
            grupos_homogeneos.append(grupo)

df['grupo4'] = df['grupo3'].isin(grupos_homogeneos)
df['grupo4'] = np.where(df['grupo4'], df['grupo3'], -1)

grupos = sorted(df['grupo4'].unique())
grupos = [grupo for grupo in grupos if grupo >=0] #tirar os -1, -2, etc.

# print_exemplos_grupos(df=df,inicio=1500,fim=1600,grupos=grupos,grupox='grupo4',cols=cols_mostrar+['NOME'],qtd_palavras=qtd_palavras,percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,unidades=unidades)
print_qtd_grupos_sentencas_uteis(df=df,grupos=grupos,grupox='grupo4')

############################################################################
# EXCLUI GRUPOS NAO HOMOGENEOS - pela contagem de palavras diferentes! - eh RAPIDO
############################################################################

df['qtd_palavras_diferentes'] = df['DS_ITEM_CLEAN'].apply(lambda x: len(set( [item for sublist in [sent.split()[:qtd_palavras] for sent in x] for item in sublist    ] ) ))
qtd_palavras_por_grupo = df.groupby('grupo4')['qtd_palavras_diferentes'].median()
qtd_palavras_por_grupo = qtd_palavras_por_grupo.sort_values()
qtd_max_palavras_diferentes_no_grupo = int(qtd_palavras_por_grupo.quantile(quantile_a_retirar_quantidade_palavras_diferentes_no_grupo))
print('Quantidade maxima de palavras diferentes aceita por grupo:', qtd_max_palavras_diferentes_no_grupo)

df['qtd_median_palavras_dif_grupo'] = df['grupo4'].map(qtd_palavras_por_grupo)

df['grupo5'] = np.where(df['qtd_median_palavras_dif_grupo'] > qtd_max_palavras_diferentes_no_grupo, -1, df['grupo4'])

grupos = df['grupo5'].unique()
grupos = [grupo for grupo in grupos if grupo >=0] #tirar os -1, -2, etc.

# print_exemplos_grupos(df=df,inicio=1500,fim=1600,grupos=grupos,grupox='grupo5',cols=cols_mostrar+['NOME'],qtd_palavras=qtd_palavras,percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,unidades=unidades)
print_qtd_grupos_sentencas_uteis(df=df,grupos=grupos,grupox='grupo5')

############################################################################
# EXCLUI GRUPOS que tem items SOMENTE DE UMA MESMA AUDITADA - sao escritos muito especificos, queremos coisas compradas por varias auditadas. - eh RAPIDO
############################################################################

qtd_auditadas_por_grupo = df.groupby(['grupo5'])['NOME'].apply(lambda x: len(np.unique(x)) )
df['qtd_auditadas_diferentes_do_grupo'] = df['grupo5'].map(qtd_auditadas_por_grupo)
df['grupo6'] = np.where(df['qtd_auditadas_diferentes_do_grupo'] > qtd_min_auditadas_para_formar_grupo, df['grupo5'], -1)

grupos = df['grupo6'].unique()
grupos = [grupo for grupo in grupos if grupo >=0] #tirar os -1, -2, etc.

# print_exemplos_grupos(df=df,inicio=1500,fim=1600,grupos=grupos,grupox='grupo6',cols=cols_mostrar+['NOME'],qtd_palavras=qtd_palavras,percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,unidades=unidades)
print_qtd_grupos_sentencas_uteis(df=df,grupos=grupos,grupox='grupo6')

############################################################################
# EXCLUI GRUPOS que tem MUITOS NUMEROS DIFERENTES NAS SENTENCES: - RAPIDO 
############################################################################

df['qtd_numeros_diferentes'] = df['DS_ITEM_CLEAN'].apply(lambda x: len(set( [item for sublist in [sent.split()[:qtd_palavras] for sent in x] for item in sublist if item.isdigit()   ] ) ))
df['qtd_numeros_diferentes'] = np.where(df['qtd_numeros_diferentes'] == 0, 0, df['qtd_numeros_diferentes']-1)
qtd_numeros_por_grupo = df.groupby('grupo4')['qtd_numeros_diferentes'].median()
qtd_numeros_por_grupo = qtd_numeros_por_grupo[qtd_numeros_por_grupo > 0]
qtd_numeros_por_grupo = qtd_numeros_por_grupo.sort_values()
qtd_max_numeros_diferentes_no_grupo = int(qtd_numeros_por_grupo.quantile(quantile_a_retirar_numeros_diferentes_no_grupo))
print('Quantidade maxima de numeros diferentes aceita por grupo:', qtd_max_numeros_diferentes_no_grupo)

df['qtd_median_numeros_dif_grupo'] = df['grupo4'].map(qtd_numeros_por_grupo)

df['grupo7'] = np.where(df['qtd_median_numeros_dif_grupo'] > qtd_max_numeros_diferentes_no_grupo, -1, df['grupo6'])

grupos = df['grupo7'].unique()
grupos = [grupo for grupo in grupos if grupo >=0] #tirar os -1, -2, etc.

# print_exemplos_grupos(df=df,inicio=1500,fim=1600,grupos=grupos,grupox='grupo7',cols=cols_mostrar+['NOME'],qtd_palavras=qtd_palavras,percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,unidades=unidades)
print_qtd_grupos_sentencas_uteis(df=df,grupos=grupos,grupox='grupo7')

###########################
# passo reduzido:
###########################

df['grupo8'] = df['grupo7']

###########################
print('Conversao word embeddings to sentence embedding, com pesos:')
###########################

model2 = Word2Vec(sentences,size=qtd_dimensoes, min_count=1,workers=-1)

doc_vectors2 = {}

for number, sent in enumerate(tqdm(sentences)):
    #agora dando peso maior pras primeiras palavras, peso decrescente ateh o final, numeros com mesmo peso da primeira palavra:
    if len(sent) == 0:
        doc_vectors2[number] = np.zeros(qtd_dimensoes,)
    elif len(sent) == 1:
        doc_vectors2[number] = model2.wv[sent[0]]
    elif len(sent) > 1:
        pesos = np.array(range(len(sent))[::]) + 1
        pesos = 1 / pesos # agora com pesos 1/x - tem que ser na ordem 1,2,..., os menores numeros dao maiores pesos - decai menos que exponencial, que eh muito brusca a queda.
        media = []
        divisao = 0
        counter = 0
        for word in sent:
            # media.append(model2.wv[word])
            # divisao += 1
            ######### AGORA O MODEL EH W2V E O PESO EH DOBRADO PRA DIGITS:
            if word.isdigit():
                media.append(model2.wv[word] * ((pesos[0]+pesos[-1])*(1/2)) )
                divisao += ((pesos[0]+pesos[-1])*(1/2))
            else:
                media.append(model2.wv[word] * pesos[counter])
                divisao += pesos[counter]
            counter += 1
        doc_vectors2[number] = np.array(media).sum(axis=0) / divisao #media de tudo

doc_vectors2 = DataFrame(doc_vectors2).T
doc_vectors2 = doc_vectors2.set_index(df.index)

doc_vectors_grupos = {}

for grupo in tqdm(grupos):
    indices = df[df['grupo8'] == grupo].index
    doc_vectors_grupos[grupo] = doc_vectors2.loc[indices]
    doc_vectors_grupos[grupo] = doc_vectors_grupos[grupo].mean(axis=0)

doc_vectors_grupos = DataFrame(doc_vectors_grupos).T

##########
#usa o scaler original:
doc_vectors_grupos_std = DataFrame(scaler.transform(doc_vectors_grupos),index=doc_vectors_grupos.index,columns=doc_vectors_grupos.columns)

grupos_similarities = cosine_similarity(doc_vectors_grupos_std)

grupos_similarities = DataFrame(grupos_similarities,index=doc_vectors_grupos.index,columns=doc_vectors_grupos.index)

similarity_minima_pra_juntar_grupos = 0.90

#junta os grupos:
grupos_similares = []
for grupo in tqdm(grupos_similarities):
    agrupar_df = grupos_similarities[grupo].sort_values(ascending=False)
    agrupar_df = agrupar_df[agrupar_df >= similarity_minima_pra_juntar_grupos]
    grupos_similares.append(list(agrupar_df.index))

novo_grupo = 0
mapeamento_grupos = {}
for mini_grupo in tqdm(grupos_similares):
    if len(mini_grupo) == 1:
        mapeamento_grupos[mini_grupo[0]] = novo_grupo
    else:
        for grupo in mini_grupo:
            if grupo not in mapeamento_grupos.keys():
                for mini_grupo2 in grupos_similares:
                    if grupo in mini_grupo2:
                        mapeamento_grupos[grupo] = novo_grupo
                        for grupo2 in mini_grupo2:
                            if grupo2 not in mapeamento_grupos.keys():
                                mapeamento_grupos[grupo2] = novo_grupo
    novo_grupo += 1

df['grupo9'] = df['grupo8'].map(mapeamento_grupos)
df['grupo9'] = df['grupo9'].fillna(-1)
df['grupo9'] = df['grupo9'].astype(int)

grupos = df['grupo9'].unique()
grupos = [grupo for grupo in grupos if grupo >=0] #tirar os -1, -2, etc.

# print_exemplos_grupos(df=df,inicio=1500,fim=1600,grupos=grupos,grupox='grupo9',cols=cols_mostrar+['NOME'],qtd_palavras=qtd_palavras,percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,unidades=unidades)
print_exemplos_grupos_v2_aleatorio(df=df,qtd_grupos_mostrar=10,grupos=grupos,grupox='grupo9',cols=cols_mostrar+['NOME'],qtd_palavras=qtd_palavras,percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,unidades=unidades)
print_qtd_grupos_sentencas_uteis(df=df,grupos=grupos,grupox='grupo9')

###############################################
# Agora repassar as sentences do grupo -1 e tentar encaixá-las nos grupos formados, só com grande certeza. - Demora 5 min.
###############################################

excluidos_index =  df[df['grupo9'] == -1].index
incluidos_index =  df[df['grupo9'] >= 0].index

doc_vectors_excluidos = doc_vectors.loc[excluidos_index]
doc_vectors_incluidos = doc_vectors.loc[incluidos_index]

doc_vectors_grupos_finais = {}
for grupo in tqdm(grupos):
    df2 = df[df['grupo9'] == grupo]
    # doc_vectors_grupos_finais[grupo] = doc_vectors_std.loc[df2.index].mean()
    # doc_vectors_grupos_finais[grupo] = DataFrame(doc_vectors_std_pca_umap,index=doc_vectors_std.index).loc[df2.index].mean()
    doc_vectors_grupos_finais[grupo] = doc_vectors_incluidos.loc[df2.index]
    doc_vectors_grupos_finais[grupo] = doc_vectors_grupos_finais[grupo].values.mean(axis=0)
# index aqui jah eh o numero certo dos grupos:
doc_vectors_grupos_finais = DataFrame(doc_vectors_grupos_finais).T

compara = cosine_similarity(doc_vectors_excluidos.loc[excluidos_index],doc_vectors_grupos_finais.values)
compara = DataFrame(compara,index=excluidos_index, columns=grupos)
similarity_do_grupo_mais_parecido = compara.max(axis=1)
grupo_mais_parecido = compara.idxmax(axis=1)

similarity_minima_pra_encaixar_itens_excluidos_no_final = 0.95

encaixar_excluidos = Series( np.where(similarity_do_grupo_mais_parecido >= similarity_minima_pra_encaixar_itens_excluidos_no_final, grupo_mais_parecido, -1), index= similarity_do_grupo_mais_parecido.index)
df['grupo10'] = encaixar_excluidos
df['grupo10'] = df['grupo10'].fillna(-1)
df['grupo10'] = np.where(df['grupo10'] == -1, df['grupo9'], df['grupo10'])

grupos = df['grupo10'].unique()
grupos = [grupo for grupo in grupos if grupo >=0] #tirar os -1, -2, etc.

# print_exemplos_grupos(df=df,inicio=1500,fim=1600,grupos=grupos,grupox='grupo10',cols=cols_mostrar+['NOME'],qtd_palavras=qtd_palavras,percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,unidades=unidades)
print_exemplos_grupos_v2_aleatorio(df=df,qtd_grupos_mostrar=10,grupos=grupos,grupox='grupo10',cols=cols_mostrar+['NOME'],qtd_palavras=qtd_palavras,percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,unidades=unidades)
print_qtd_grupos_sentencas_uteis(df=df,grupos=grupos,grupox='grupo10')

grava(coisa=doc_vectors_grupos_finais,filename='doc_vectors_grupos_finais_20190823v3.pkl',path=folder)

###############################################
# Agora repassa todas os registros do LICITACON que nao foram usados (outros que nao natureza_despesa de material de consumo) e tenta encaixa-los
###############################################

#jah foram tentados todos do df, os que foram classificados e os que nao deu, ficaram -1 mesmo.
registros_a_tentar_classificar = [i for i in licitacon.index if i not in df.index]

df2 = licitacon.loc[registros_a_tentar_classificar]

sentences2 = [sent.split()[:qtd_palavras] for sent in df2['DS_ITEM_CLEAN']]

print('Conversao word embeddings to sentence embedding, com pesos:')
doc_vectors2 = {}
#Here the magic happens:
for number, sent in enumerate(tqdm(sentences2)):
    #agora dando peso maior pras primeiras palavras, peso decrescente ateh o final, numeros com mesmo peso da primeira palavra:
    if len(sent) == 0:
        doc_vectors2[number] = np.zeros(qtd_dimensoes,)
    elif len(sent) == 1:
        doc_vectors2[number] = model.wv[sent[0]]
    elif len(sent) > 1:
        pesos = np.array(range(len(sent))[::]) + 1
        pesos = 1 / pesos # agora com pesos 1/x - tem que ser na ordem 1,2,..., os menores numeros dao maiores pesos - decai menos que exponencial, que eh muito brusca a queda.
        media = []
        divisao = 0
        counter = 0
        for word in sent:
            if word.isdigit():
                media.append(model.wv[word] * ((pesos[0]+pesos[-1])*(1/4)) ) #nova abordagem: se eh digit, atribui peso NO 3/4 da faixa entre o peso da primeira e da ultima palavra. Mesmo peso pra todos os numeros, mais importante que palavras do fim, menos importante que palavras do inicio.
                divisao += ((pesos[0]+pesos[-1])*(1/4))
            else:
                media.append(model.wv[word] * pesos[counter])
                divisao += pesos[counter]
            counter += 1
        doc_vectors2[number] = np.array(media).sum(axis=0) / divisao #media de tudo

doc_vectors2 = DataFrame(doc_vectors2).T
doc_vectors2 = doc_vectors2.set_index(df2.index) # coloca o ID_ITEM como index:

# faz por partes pra nao dar memory error:
qtd_chunks = 10
similarity_do_grupo_mais_parecido_final = Series()
grupo_mais_parecido_final = Series()

for df_splited in tqdm(np.array_split(doc_vectors2, qtd_chunks)):

    compara = cosine_similarity(df_splited.values,doc_vectors_grupos_finais.values)
    compara = DataFrame(compara,index=df_splited.index, columns=doc_vectors_grupos_finais.index)

    similarity_do_grupo_mais_parecido = compara.max(axis=1)
    grupo_mais_parecido = compara.idxmax(axis=1)

    similarity_do_grupo_mais_parecido_final = similarity_do_grupo_mais_parecido_final.append(similarity_do_grupo_mais_parecido)
    grupo_mais_parecido_final = grupo_mais_parecido_final.append(grupo_mais_parecido)

# -2 sao os que nao foram encaixados, mas vao ser treinados no proximo umap:
# -1 continua sendo os que passaram pelo processo mas nao foram classificados como material de consumo
encaixar_excluidos = Series( np.where(similarity_do_grupo_mais_parecido_final >= similarity_minima_pra_encaixar_itens_excluidos_no_final, grupo_mais_parecido_final, -2), index= similarity_do_grupo_mais_parecido_final.index)

#encaixa todos os results do material de expediente no licitacon:
classificacoes_final = encaixar_excluidos.append(df['grupo10'])
licitacon['grupo10'] = classificacoes_final

grupos = licitacon['grupo10'].unique()
grupos = [grupo for grupo in grupos if grupo >=0] #tirar os -1, -2, etc.

print_exemplos_grupos_v2_aleatorio(df=licitacon,qtd_grupos_mostrar=10,grupos=grupos,grupox='grupo10',cols=cols_mostrar+['NOME'],qtd_palavras=qtd_palavras,percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,unidades=unidades)
print_qtd_grupos_sentencas_uteis(df=licitacon,grupos=grupos,grupox='grupo10')

# grava(coisa=licitacon,filename='licitacon_20190826.pkl',path=folder)
grava(coisa=licitacon,filename='licitacon_20190823v3.pkl',path=folder)