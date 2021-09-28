# imports

import os
from dotenv import load_dotenv, dotenv_values
from pyhive import hive
import pandas as pd
import io
import math
import multiprocessing

nome_trilha = 'F03_PRECIFICACAO_ITEM_LICITACAO'
database='default'
auth='CUSTOM'

def get_database_parameters():

    load_dotenv()
    username = os.environ.get('DB_USER_NAME')
    password = os.environ.get('DB_PASSWORD')
    host = os.environ.get('DB_HOST')
    port = int(os.environ.get('DB_PORT'))
    schema = os.environ.get('DB_SCHEMA')

    return (username, password, host, port, schema)


def hive_table_to_dataframe(table, table_schema=None, query=None):
    '''
        It returns the content of a Hive table in a data frame.

        table (str): Hive table name.
    '''

    username, password, host, port, schema = get_database_parameters()
    if table_schema is not None:
        schema = table_schema
    hive_connection = hive.Connection(
                host=host,
                port=port,
                username=username,
                password=password,
                database=database,
                auth=auth)

    table_from = schema+'.'+table
    if query is not None:
        dataframe = pd.read_sql(query + table_from, hive_connection)
    else:
        dataframe = pd.read_sql("SELECT * FROM "+table_from, hive_connection)
    hive_connection.close()
    column_list = dataframe.columns
    for column in column_list:
        dataframe=dataframe.rename(columns={column: column.split(".")[1]})
    return dataframe


def _insert_hive_table(dataframe, table, version, batch_size, begin, end,
                       process_cont):
    '''
        It inserts the content of a data frame in a Hive table.

        dataframe (dataframe): Dataframe to save in a Hive table.
        table (str): Hive table name.
        version (str): Execution version.
        batch_size (int): Batch size.
        begin (int): Begining of the interval of the dataframe.
        end (int): Ending of the interval of the dataframe.
        process_cont (int): Current process.
    '''

    username, password, host, port, schema = get_database_parameters()
    cont = 1

    batch_rows_wr = io.StringIO()
    for index, row in dataframe.iloc[begin:end].iterrows():

        columns_value_wr = io.StringIO()
        row_values = row.values
        for i in range(len(row_values)):
            if ((row_values[i] == None) or ((type(row_values[i]) == float) and (math.isnan(row_values[i])) )):
                columns_value_wr.write("null")
            elif (type(row_values[i]) == str):
                str_value = row_values[i].replace("\\", "\\\\")
                str_value = str_value.replace("'", "\\\'")
                str_value = "\'"+str_value+"\'"
                columns_value_wr.write(str_value)
            else:
                columns_value_wr.write(str(row_values[i]))
            columns_value_wr.write(", ")
        columns_value_wr.write(str(version)+", ")
        columns_value_wr.write("CURRENT_TIMESTAMP, ")
        columns_value_wr.write("\""+nome_trilha+"\"")
        batch_rows_wr.write("( "+columns_value_wr.getvalue()+" ),")
        if (((cont % batch_size) == 0) or (begin+cont == end)) :
            print('Insert Hive process', process_cont, begin+cont, end)

            batch_rows = batch_rows_wr.getvalue()[:-1]
            hive_connection = hive.Connection(
                host=host,
                port=port,
                username=username,
                password=password,
                database=database,
                auth=auth)
            cursor = hive_connection.cursor()
            cursor.execute("INSERT INTO "+schema+"."+table+" VALUES "+batch_rows)
            cursor.close()
            hive_connection.close()
            batch_rows_wr = io.StringIO()
        cont+=1


def _calc_interval(total_size, num_process):
    '''
        It calculates the interval of the dataframe that each process must handle.

        total_size (int): Total size of the dataframe.
        num_process (int): Number of process.
    '''

    process_load = int(total_size / num_process)
    begin = 0
    end = 0
    intervals = []
    while (begin <= total_size) :
        if ((begin+process_load) <= total_size):
            end += (process_load)
            intervals.append([begin,end])
            begin = (end)
        else:
            end = (total_size)
            intervals.append([begin,end])
            break

    return intervals


def dataframe_to_hive_table(dataframe, table, version, batch_size=1000,
                            num_process=20):
    '''
        It saves the content of a data frame in a Hive table, using batch and multi-process.
        It drops then recreates the table.

        dataframe (dataframe): Dataframe to save in a Hive table.
        table (str): Hive table name.
        version (str): Execution version.
        batch_size (int): Batch size.
        num_process (int): Number of process.
    '''

    username, password, host, port, schema = get_database_parameters()
    hive_connection = hive.Connection(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        auth=auth)
    cursor = hive_connection.cursor()
    cursor.execute("DROP TABLE "+schema+"."+table)

    columns_name_wr = io.StringIO()
    column_list = dataframe.columns
    for i in range(len(column_list)):
        columns_name_wr.write(column_list[i] + " STRING, ")
    columns_name_wr.write("metadata_trilha_versao STRING, ")
    columns_name_wr.write("metadata_trilha_data_execucao STRING, ")
    columns_name_wr.write("metadata_nome_trilha STRING")

    cursor.execute("CREATE TABLE "+schema+"."+table+" ( "+columns_name_wr.getvalue()+" ) ")
    cursor.close()
    hive_connection.close()

    manager = multiprocessing.Manager()
    jobs = []
    begin_end_tuples = _calc_interval(dataframe.shape[0],num_process)
    for x in range(len(begin_end_tuples)):
        begin, end = begin_end_tuples[x]
        p = multiprocessing.Process(target=_insert_hive_table,
                args=[dataframe, table, version, batch_size, begin, end, x])
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
        proc.close()
