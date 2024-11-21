from financial_func import financial_statement,find_company_id
from langchain_core.messages import HumanMessage,SystemMessage
from start_model import initial_model, model_starter
from langchain_core.tools import tool
from typing import Optional, Type, Union
from writelog import write_info
import datetime


tools = list() #拿來儲存 function，並且為給LM

@tool
def get_todays_date():
    "it is a tool to find latest date"
    today = datetime.date.today()
    return today

tools.append(get_todays_date)

@tool
def financial_property_data(year:int, quarter:int, symbol:int) -> object:
    """it is a tool to find the corresponding company's financial report.

    Args:
        year: year
        quarter: quarter
        symbol: company's id or symbol
    """
    return financial_statement(year,quarter,symbol)

tools.append(financial_property_data)


@tool
def find_company_id_data(query:str) -> int:
    """it is a tool to find the corresponding company's id in financial market. 

    Args:
        query: a sentence asked by user about search of company's id or symbom
    """
    return find_company_id(query)

tools.append(find_company_id_data)

# 這個dict 是拿來執行function
available_functions = {
    "get_todays_date": get_todays_date,
    "financial_property_data":financial_property_data,
    "find_company_id_data": find_company_id_data
}


def run_agent_with_memory(agent, query, memory):
    # memory.chat_memory.add_user_message(HumanMessage(query)) # langchain 的歷史紀錄api，也許日後可以使用

    # covert to langchain class

    # ai_msg = agent.invoke(memory.chat_memory.messages)
    ai_msg = agent.invoke(memory) # 跑LM並得到回覆

    # memory.chat_memory.add_ai_message(ai_msg)
    memory.append(ai_msg) # 儲存於 chat_history

    return ai_msg

# execute function
def execute_function(agent,ai_msg,query,memory):
    info = list()
    # memory.chat_memory.add_message(AIMessage(ai_msg.additional_kwargs)) # tool_calls first
    while(ai_msg.response_metadata['finish_reason'] == 'tool_calls'): #如果結束理由為tool_calls，代表還能繼續
        for tool_call in ai_msg.tool_calls:
            tool = available_functions[tool_call["name"]]
            tool_msg = tool.invoke(tool_call)
            # memory.chat_memory.add_message(tool_msg)
            memory.append(tool_msg) # 儲存做的每個動作
            info.append(tool_msg)
        ai_msg = run_agent_with_memory(agent, query, memory)
    return ai_msg

# Generate a response
def response_llm(llm,info,query,memory):
    system_prompt = "請用拿到的資訊針對問題給出回覆，並且以繁體中文給使用者進行回覆: {out}"
    response = llm.invoke([SystemMessage(content=system_prompt.format(out = info)),
                            HumanMessage(content=query)])
    # memory.chat_memory.add_message(response)
    memory.append(response)

    return response


if __name__=="__main__":
    system_prompt = """
    [Today]
    {date}
    You are an agent who is capable of applying multiple [tools] to get to correct information when [previous output] is not enough to answer the question

    you should follow instrucitons below : 

    following are instruciton of this process : 
    1. check time and company name and company number are identified correctly or no , especailly if time is mentioned to be latest then you should match year and quarter between [Today] and [previous output] as priority
    2. confirm  whether [previous output] could answer the query. If it is enough to answer the question then stop the search , otherwise initiate a search to correct the data.
    3. [previous output] contains tools used and its output, be careful with same output from same function.
    4. if the information you try to retrieve is not existing, you will change the quarter and year to fetch another information
    5. before you use API searches information, explain to the user how you will fulfill their requests with the functions provided
    6. please respond in traditional chinese !!!""".format(date = datetime.date.today().strftime("%Y-%m-%d"))

    llm = model_starter()
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = list()
    # memory.chat_memory.add_message(SystemMessage(system_prompt)) # system prompt
    memory.append(SystemMessage(system_prompt)) # system prompt

    llm_with_tools = llm.bind_tools(tools)

    query = '我要精誠的最新財報'
    query = "我想要精誠，台積電跟鴻海的的最新財報"
    # query = "我想要精誠，台積電跟鴻海的的最新財報"

    memory.append(HumanMessage(query))
    ai_msg = run_agent_with_memory(llm_with_tools,query, memory)
    info = execute_function(llm_with_tools,ai_msg,query,memory)
    # response = response_llm(llm,info,query,memory)
    for i in info.content.split('\n'):
        write_info(i)