import queue
import datetime
import json
import os 
from writelog import write_info
from dotenv import load_dotenv
import pandas as pd

# self-defined
from financial_func import call_fin_report
from finance import embedding_fin_query
from stockinfo import embedding_stockinfo_query
from chat_openai import query_aoai_embedding
from app import symbol_data,process_stock_id
from start_model import initial_model
from function_DB import tools , available_functions,careful_note

load_dotenv()
aoai = initial_model()
tools_instruc = []
for tool in tools :
    # tools_instruc.append({'name': tool['function']['name'],'description' : tool['function']['description']})
    tools_instruc.append(f"{tool['function']['name']} - description: {tool['function']['description']}\n parameters : {tools[0]['function']['parameters']['properties']}")

# to confirm if response is acceptable
def satisfaction_check(query,tool_out):
    response = aoai.chat.completions.create(
            model="gpt-4o-mini",
            messages= [
                {
                "role": "system",
                "content": """
                [query] : {query}
                [response] : {response}
                [time] : {date}

                as an user who requests the question, are you satisfied with question : [query] to answer : [response] corresponding to the time : [time]
                the [response] could be in any format, please judge it yourself
                please must answer in english and start with yes or no,
                later generate the response or explanation

                assistant : 
                """.format(response = tool_out,query = query,date = datetime.date.today().strftime("%Y-%m-%d"))
                }
                ],

            )

    return response.choices[0].message.content.lower() 


# check if it is json data
def is_json_format(variable):
    if isinstance(variable, pd.DataFrame):
        variable = variable.to_dict(orient="records")
    if isinstance(variable,dict):
        variable = json.dumps(variable)
    if isinstance(variable,list):
        variable =  " ".join(map(str, variable))
    return variable

system_prompt = """
        [Today]
        {date}

        [tools]
        {tools_instruc}
        Here are some instructions for tools:
        {careful_note}

        [previous output]
        {sql_info}

        {info}

        You are an agent who is capable of applying multiple [tools] to get to correct information when [previous output] is not enough to answer the question

        you should follow instrucitons below : 

        following are instruciton of this process : 
        1. check time and company name and company number are identified correctly or no , especailly if time is mentioned to be latest then you should match year and quarter between [Today] and [previous output] as priority
        2. confirm  whether [previous output] could answer the query. If it is enough to answer the question then stop the search , otherwise initiate a search to correct the data.
        3. [previous output] contains tools used and its output, be careful with same output from same function.
        4. if the information you try to retrieve is not existing, you will change the quarter and year to fetch another information
        5. before you use API searches information, explain to the user how you will fulfill their requests with the functions provided
        6. please respond in traditional chinese !!!
        """
class BreakNestedLoop1(Exception):
    pass
class BreakNestedLoop2(Exception):
    pass

def out(query,content=None):
    cnt = 1
    no_cnt = 0
    tool_out = []
    THRESHOLD = 3
    # if prev output is wrong , then it will go for this loop to use function call
    record = dict()
    tool_out.append({"role": "system","content" : system_prompt.format(tools_instruc = tools_instruc, careful_note= careful_note, sql_info=content, info = tool_out,date = datetime.date.today().strftime("%Y-%m-%d"))})
    tool_out.append({"role": "user","content" :query})
    while(True):
        response = aoai.chat.completions.create(
            model="gpt-4o-mini",
            messages= tool_out,
            tools=tools,
            tool_choice="auto",  
            )

        # 從輸出中找到，有什麼工具被使用了
        response_message = response.choices[0].message 
        write_info(f'time {cnt}')
        write_info('out')
        write_info(response_message.content)
        write_info('tools')
        write_info(response_message.tool_calls)
        write_info('why stop')
        write_info(response.choices[0].finish_reason)
        write_info('')
        tool_calls = response_message.tool_calls
        output = ""
        try : 
            if response.choices[0].finish_reason != 'stop' or response.choices[0].finish_reason == 'tool_calls' :
                # 必須先有tool_calls
                tool_out.append(response.choices[0].message)
                print('message')
                print(tool_out[-1])
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call( # 請以json format 回傳 function 的 output
                        **function_args # parameters 不拘數量 
                    )

                    if function_response:
                        function_args.update(function_response) # 融合args 跟 func 輸出
                    
                    # 才能有tools
                    tool_out.append(
                        {
                            "role": "tool",  
                            "content": json.dumps(function_args),
                            "tool_call_id": tool_call.id,  
                        }
                    )
                    function_response = is_json_format(function_response)
                    if function_response not in record :
                        record[function_response] =1 
                    record[function_response] += 1
                    if record[function_response] >= THRESHOLD :
                        raise BreakNestedLoop1
            elif response.choices[0].finish_reason == 'stop':
                output = response_message.content
                cnt += 1
                yes_or_no = satisfaction_check(query,output)
                print('yes_or_no')
                print(yes_or_no)
                if 'yes' in yes_or_no :
                    break
                if 'no' in yes_or_no :
                    tool_out.append({"role": "assistant","content" :yes_or_no})
                    no_cnt += 1
                    if no_cnt == 2:
                        raise BreakNestedLoop2
        except BreakNestedLoop1:
            write_info('more than 3 time repetition, finish the loop')
            output = f"問題: {query} 無法被處理，請確認資訊正確再輸入一次"
            break
        except BreakNestedLoop2:
            write_info('endless loop')
            break
        cnt += 1
    return output 
        


if __name__ == "__main__":
    search_result = []

    # query = "我要台積電2025年第1季的行情"
    # query = "我要長榮2024年第1季的財報"
    query = "我要台積電2022年第4季的行情，然後再跟我說1+1等於多少"
    # query = "然後再跟我說1+1等於多少，我要台積電2022年第4季的行情"
    # query = "3+5-7*6"

    Low_temperature = 0.1
    stock_id_queue = queue.Queue()

    try : 
        embedded_query = query_aoai_embedding(query) # 把文字轉成vector
        process_stock_id(query, Low_temperature, symbol_data, stock_id_queue) # 抓去stock id 跟名稱
        data = stock_id_queue.get() #把資料從queue抓出來
        stock_list = data['stock']
        # Access the first item in the list
        stock = stock_list[0]

        # Access individual elements
        stock_id = stock['symbol']
        stock_name = stock['company_name']

        content = embedding_fin_query(stock_id, stock_name, embedded_query, search_result) #從SQL抓出資料
    except : 
        pass
    
    print(out(query,content))