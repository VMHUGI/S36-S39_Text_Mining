import pandas as pd
import string
from unidecode import unidecode
import spacy
# python -m spacy_spanish_lemmatizer download wiki
from spacy_spanish_lemmatizer import SpacyCustomLemmatizer
# python -m spacy download es_core_news_sm
import es_core_news_sm
# python -m spacy download es_core_news_md
import es_core_news_md
# python -m spacy download es_core_news_lg
import es_core_news_lg
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#db_prueba=pd.read_excel(r'D:\Proyectos\Sistema STD\BD Consolidada Final.xlsx')

def estandarizacion_palabras (palabra):
    '''
    Toma la columna de un dataframe como input y transforma el contenido de minúsuclas a mayúsculas,
    así como retira los acentos agudos (ÁÉÍÓÚ), acentos graves (ÀÈÌÒÙ) o acentos circunflejos (ÂÊÎÔÛ)
          
    Inputs:
    palabra: dataframe con una columna definida
   
    Return: 
    Columnas inicial del dataframe en mayúsculas y sin acentos agudos o graves
    
    '''
    palabra = palabra.str.upper()
    palabra = palabra.str.translate(str.maketrans("ÁÉÍÓÚ", "AEIOU"))
    palabra = palabra.str.translate(str.maketrans("ÀÈÌÒÙ", "AEIOU"))
    palabra = palabra.str.translate(str.maketrans("ÂÊÎÔÛ", "AEIOU"))
    palabra = palabra.str.split().str.join(' ')
    
    return palabra


def eliminacion_stopwords (palabra):
    '''
    Toma la columna de un dataframe como input y elimina todas aquellas palabras incluidas
    dentro de la lista de stopwords, tales como preposiciones, artículos, conjunciones,
    adjetivos demostrativos, adjetivos posesivos, entre otros.
          
    Inputs:
    palabra: dataframe con una columna definida
   
    Return: 
    Columnas inicial del dataframe sin los stopwords definidos
    
    '''
    stopwords_spanish = stopwords.words('spanish')
    stopwords_spanish = estandarizacion_palabras(pd.DataFrame(stopwords_spanish)[0]).tolist()   
    palabra = palabra.str.split().apply(lambda x: [i for i in x if i not in stopwords_spanish]).str.join(' ')
    
    return palabra


def eliminacion_numeros_puntuacion (palabra):
    '''
    Toma la columna de un dataframe como input y elimina los números y los signos de puntuación
    string.punctuation -> !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
    adicionales -> ¡¿°º-–•“”‘’´ª¨
          
    Inputs:
    palabra: dataframe con una columna definida
   
    Return: 
    Columnas inicial del dataframe sin incluir números ni signos de puntuación
    
    '''
    palabra = palabra.str.replace(r'\d+','')
    palabra = palabra.str.translate(str.maketrans('','',string.punctuation + '¡¿°º-–•“”‘’´ª¨'))
    palabra = palabra.str.split().str.join(' ')
    
    return palabra


def lematizacion (palabra):
    '''   
    Toma la columna de un dataframe como input y realiza la lematización de cada una de las palabras
    incluidas en dicha columna.
        
    Args:
    palabra: dataframe con una columna definida
        
    Returns:
    Columnas inicial del dataframe tokenizada

    '''    
    nlp = es_core_news_sm.load()
    lemmatizer = SpacyCustomLemmatizer()
    nlp.add_pipe(lemmatizer, name="lemmatizer", after="tagger")

    palabra = palabra.apply(nlp)
    palabra = palabra.apply(lambda x: ' '.join([t.lemma_.upper() for t in x]))
        
    return palabra


def diccionario_palabras (palabra):
    '''
    Toma la columna de un dataframe, estandariza las palabras en mayúsculas y sin tilde, luego remueve
    la puntuación y números, posteriormente retira los stopwords y finalmente lematiza las palabras.
    Una vez ello, se separa cada palabra y se realiza el conteo con lo que finalmente se obtiene un
    diccionario con el listado de palabras y la frecuencia con la que aparece cada una de ellas.
    
    Inputs:
    palabra: dataframe con una columna definida
   
    Return: 
    Diccionario de palabras con su frecuencia respectiva
    
    '''
    palabra = estandarizacion_palabras(palabra)
    palabra = eliminacion_numeros_puntuacion(palabra)
    palabra = eliminacion_stopwords(palabra)
    palabra = lematizacion(palabra)
    
    
    db_aux = palabra.str.cat(sep=' ')
    words = nltk.tokenize.word_tokenize(db_aux)
    word_dist = nltk.FreqDist(words)

    diccionario = pd.DataFrame.from_dict(word_dist, orient='index')
    diccionario.columns = ['FRECUENCIA']
    diccionario.index.name = 'PALABRA'
    diccionario = diccionario.reset_index().sort_values(['PALABRA'], ascending = True)

    return diccionario

