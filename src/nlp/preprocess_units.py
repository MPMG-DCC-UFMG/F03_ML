from nlp.preprocessing_portuguese import TextPreProcessing as tpp
from nlp.preprocessing import PreprocessingText
from gensim.parsing.preprocessing import strip_multiple_whitespaces


def isfloat(value):
    value_ = value.replace(',','.')
    try:
        float(value_)
        return True
    except ValueError:
        return False


def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]


def remove_special_characters(text):
    lista = '-#@%?º°ª:/;~^`[{]}\\|!$"\'&*()=+><\t\r\n…'
    result = text
    for i in range(0, len(lista)):
        result = result.replace(lista[i], ' ')
    return result


def remove_dots_commas(text):
    lista = '.,'
    result = text
    for i in range(0, len(lista)):
        result = result.replace(lista[i], '')
    return result


def lemmatize_unit_metric(text, canonical_unit_metric):
    if text in canonical_unit_metric:
        return canonical_unit_metric[text]
    return text


def process_un_medida(un_medida, stopwords, preprocessing, canonical_unit_metric):

    un_medida_process = str(un_medida).lower()

    if len(un_medida_process) == 50:
        un_medida_process = un_medida_process.lstrip('0')
    un_medida_process = remove_prefix(un_medida_process, "1 ")
    un_medida_process = tpp.remove_accents(un_medida_process)

    un_medida_process = remove_special_characters(un_medida_process)
    un_medida_process = remove_dots_commas(un_medida_process)

    un_medida_process = ' '+un_medida_process+' '
    un_medida_process = re.sub(r' (\d+)([a-z]+) ', r' \1 \2 ', un_medida_process, flags=re.I)
    un_medida_process = un_medida_process.strip()

    items = un_medida_process.split(' ')
    new_items = []
    for item in items:
        if item not in stopwords:
            new_items.append(item)
    items = new_items

    items = preprocessing.lemmatization_document(items)
    new_items = []
    for item in items:
        new_items.append(lemmatize_unit_metric(item, canonical_unit_metric))

    un_medida_process = ' '.join(new_items)

    un_medida_process = strip_multiple_whitespaces(un_medida_process)

    return un_medida_process


def group_dsc_unidade_medida(items_df):
    '''
        Group some of the unit metrics ('dsc_unidade_medida') in the dataframe
        (replacing some of the values):
        'cx' -> 'caixa', ['unitario', 'unid', 'und'] -> 'unidade'.
    '''

    with open('../data/palavras/unit_metric_canonical.json', 'r') as data:
        unit_metric_canonical = json.load(data)

    canonical_unit_metric = {}

    for canonic, unit_list in unit_metric_canonical.items():
        for unit in unit_list:
            canonical_unit_metric[unit] = canonic

    preprocessing = PreprocessingText(spellcheck=False)
    stopwords = preprocessing.stopwords

    items_df['dsc_unidade_medida'] = items_df['dsc_unidade_medida'].apply(lambda x: process_un_medida(x, stopwords, preprocessing, canonical_unit_metric))
