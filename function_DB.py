
# 可以拿來標注的變數類型
# string
# number
# boolean
# null/empty
# object
# array
# Enum
# anyOf

import json
from datetime import datetime

# from financial_func import call_fin_report
from financial_func import financial_statement,find_company_id
from start_model import initial_model

# Get the current date
current_date = datetime.now()
# Extract the year
current_year = current_date.year

symbol_file_path = 'symbol.json'
with open(symbol_file_path, "r", encoding="utf-8") as f:
    symbol_data = json.load(f)
symbol_list = list(map(int,list(symbol_data.keys())))

##### 這裡開始為self-defined functions，近量以json的格式回傳，以利於LM閱讀
# 乘法
def multiply(a,b):
    return {"output":a*b}


tools = [
        
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "multiply two digits",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "The city name, e.g. San Francisco",
                        },
                        "b": {
                            "type": "number",
                            "description": "The city name, e.g. San Francisco",
                        }
                    },
                    "required": ["a","b"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "financial_property_data",
                "description": "it is a tool to find the corresponding company's financial report",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "year": {
                            "type": "number",
                            "description": "year",
                            # "enum": list(range(1901,current_year+1)) ＃可以指定範圍，以list的形式
                        },
                        "quarter": {
                            "type": "number",
                            "description": "quarter",
                            "enum": [1,2,3,4]
                        },
                        "symbol": {
                            "type": "number",
                            "description": "company's id or symbol",
                            # "enum": symbol_list,
                        }
                    },
                    "required": ["year","quarter","symbol"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_company_id",
                "description": "find company's id in stock market from user's text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "text which contains the information of company's name ",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            }
        },
    ]
# financial_statement
##### 到這裡為self-defined functions

# 在使用dict儲存
available_functions = {
    "multiply" : multiply,
    "financial_property_data" : financial_statement,
    "find_company_id" : find_company_id
}

# 寫注意事項，譬如function的使用順序
careful_note = """
    1. if there is need to call financial_property_data , you must call find_company_name_id to check correct company's id
    """

if __name__ == '__main__':
    aoai = initial_model()
    query = """What's the current time in San Francisco"""
    query = """what is the ratio between a company's last stock price and aa  company's first stock price"""
    query = '請從我的postgre資料庫找出aa最新的股價和aa最舊的股票，並計算最新跟最舊的比例'
    query = '寶成公司的app怎麼下載'

    response = aoai.chat.completions.create(
        model="gpt-4o-mini",
        messages= [
        {
        "role": "system",
        "content": "you are a powerful chatbot , you could use tools based on user's query to find the answer"
        }
        ,
        {
        "role": "user",
        "content": query
        }],

        tools=tools,
        tool_choice="auto",  
    )

    # 這邊的輸出是LM使用過的function
    response_message = response.choices[0].message 
    tool_calls = response_message.tool_calls

    # 在用for迴圈把過程跑一遍
    response_message = response.choices[0].message 
    tool_calls = response_message.tool_calls
    messages = []
    if tool_calls:
        # Step 3: Extending conversation with a function reply        
        messages.append(response_message)

        # Step 4: Sending each function's response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                **function_args
            )
            print(function_response)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",    
                    "name": function_name,
                    "content": function_response
                }
            )

    # 把產出的資料給LM在做一次問題答案的關聯分析，然後再產出答案
    response = aoai.chat.completions.create(
        model="gpt-4o-mini",
        messages= [
        {
        "role": "system",
        "content": """你是個答覆機器人，會根據推理[處來的資料]和原先的問題進行分析，再進一步的去做邏輯推理，
        如果你覺得出來的答案和問題不相符，請再做出更近一步的解析，
        並且輸出透過得出的答案，在一步步的說明給使用者
        [處來的資料]:
        {message}
        
        """.format(message = messages)
        }
        ,
        {
        "role": "user",
        "content": query
        }],
    )


    response_message = response.choices[0].message 
    for i in response_message.content.split('\n'):
        print(i)



