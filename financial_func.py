import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv
import requests
import datetime
import json
import queue

# self-defined
from common import get_stock_id
from finance import embedding_fin_query
from writelog import write_error, write_info
from stockinfo import embedding_stockinfo_query
from chat_openai import query_aoai_embedding
from app import symbol_data,process_stock_id
from cachetools import cached, TTLCache

# Cache data for 1 hour with a maximum of 100 items stored
cache = TTLCache(maxsize=100, ttl=3600)

load_dotenv()
# connection_string = "postgres://postgres:systex6214.@database-3.c1yid0fost4w.ap-northeast-1.rds.amazonaws.com/2024-GPT-E"
connection_string = os.getenv("FDC_POSTGRES_URI")

@cached(cache)
def load_json_file(filepath='/Users/fernandodoro/Desktop/修成/symbol.json'):
        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)
        
data = load_json_file()


def connect_db():
    """連接到 PostgreSQL 數據庫"""
    global connection_string
    try:
        write_info(f"fdc connection_string: {connection_string}")
        conn = psycopg2.connect(connection_string)
        return conn
    except Exception as e:
        write_error(f"資料庫連接失敗: {e}")
        return None


def call_fin_report(query):
    conn = connect_db()
    cursor = conn.cursor()
    """
    select stock_symbol,quarter,content,total_chunk,ymdon,date_modified,latest,type
    from fin_report
    where latest = 1 and type = 'A';
    """

    # 執行 select_newstypename_sql 與 select_term_sql
    cursor.execute(query)
    data_newstypename = cursor.fetchall()
    with conn:
        df = pd.read_sql(query, conn)

    # Close the connection
    # cursor = conn.cursor()
    conn.close()
    return df.to_json(orient='records')

def financial_statement(year:int, quarter:int, symbol:int, type='綜合損益彙總表')->dict:
    try:
        original = year 
        if year >= 1000:
            year -= 1911

        if type == '綜合損益彙總表':
            url = 'https://mops.twse.com.tw/mops/web/ajax_t163sb04'
        elif type == '資產負債彙總表':
            url = 'https://mops.twse.com.tw/mops/web/ajax_t163sb05'
        elif type == '營益分析彙總表':
            url = 'https://mops.twse.com.tw/mops/web/ajax_t163sb06'
        else:
            print('type does not match')

        r = requests.post(url, {
            'encodeURIComponent':1,
            'step':1,
            'firstin':1,
            'off':1,
            'TYPEK':'sii',
            'year':str(year),
            'season':str(quarter),
        })

        r.encoding = 'utf8'
        dfs = pd.read_html(r.text, header=None)
        dfs = pd.concat(dfs[1:], axis=0, sort=False)

        from io import StringIO
        csv_buffer = StringIO()
        dfs = dfs[dfs['公司 代號'] == symbol].dropna(axis=1, how='all')
        dfs['year'] = original
        dfs['quarter'] = quarter
        df_json = dfs.to_json(orient='records')
        parsed = json.loads(df_json)
        return parsed[0] # as dict
    except:
        return None

from fuzzywuzzy import process
def verify_id_by_name(query):
    # Load JSON data from a file
    # Find the best match for a company name using fuzzy matching
    def find_best_match(data, query):
        all_names = {name: key for key, names in data.items() for name in names}
        best_match, score = process.extractOne(query, all_names.keys())
        company_id = all_names[best_match]
        return company_id, best_match, score
    company_id, best_match, score = find_best_match(data, query)
    write_info('company info')
    write_info(query)
    write_info(company_id)

    return int(company_id)

def find_company_id(query:str)->dict:
    out = get_stock_id(query,0)
    stock_id_json = json.loads(out)
    name = stock_id_json['stock'][0]['company_name']
    id = verify_id_by_name(name)
    return {"symbol":id}

if __name__=="__main__":
    search_result = []

    query = "我要台積電的最新行情"
    Low_temperature = 0.1
    stock_id_queue = queue.Queue()

    embedded_query = query_aoai_embedding(query)
    process_stock_id(query, Low_temperature, symbol_data, stock_id_queue)

    data = stock_id_queue.get()
    stock_list = data['stock']
    # Access the first item in the list
    stock = stock_list[0]

    # Access individual elements
    stock_id = stock['symbol']
    stock_name = stock['company_name']

    q_result_list = embedding_fin_query(stock_id, stock_name, embedded_query, search_result)


    for i in q_result_list:
        write_info(i)
        write_info('\n')

