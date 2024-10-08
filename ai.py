import os
import time
from openai import OpenAI
import json
import tools
from typing import *
from dotenv import load_dotenv
load_dotenv()
aikey = os.getenv("aikey")
url = os.getenv("aiurl")
model = os.getenv("model")
client = OpenAI(
    api_key = aikey, # 在这里将 MOONSHOT_API_KEY 替换为你从 Kimi 开放平台申请的 API Key
    base_url = url,
)
tool = [
    {
        "type": "function",
        "function": {
            "name": "time",
            "description": "get now time (like be 2024-09-15 12:34:56)",
            "parameters": { # 使用 parameters 字段来定义函数接收的参数
				"type": "object", # 固定使用 type: object 来使 Kimi 大模型生成一个 JSON Object 参数
				"required": ["query"], # 使用 required 字段告诉 Kimi 大模型哪些参数是必填项
				"properties": { # properties 中是具体的参数定义，你可以定义多个参数
					"nothing": { # 在这里，key 是参数名称，value 是参数的具体定义
						"type": "string", # 使用 type 定义参数类型
						"description": """
							just anything(JUST A TIME)
						""" # 使用 description 描述参数以便 Kimi 大模型更好地生成参数
					}
				}
			}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": """查询天气 返回文字(注：不成功或查询失败为500)""",
            "parameters": {
                "type": "object",
                "required": ["adm1","adm2"],
                "properties": {
                    "adm1":{
                        "type": "string",
                        "description":"地点（如：南山 江门 等）（不要加'区'字）"
                    },
                    "adm2":{
                        "type": "string",
                        "description":"省（如重庆市 深圳 等）"
                    }
                }
            }
        }
    },
#    {
#         "type": "function",
#         "function": {
#             "name": "draw",
#             "description": """调用stable diffusers绘图 模型为动漫风 返回为一个http(仅内网访问)处理速度慢 请务必将“需要插入图片的地方”写为"...(((./wdads.png)))..." 如"小喵的自拍来啦喵~ (((./result.png))) 希望主人喜欢小喵的照片喵~ 咱会一直陪在主人身边的喵~" ...代表你输出的其他内容 提示词使用英文 每个消息只能使用一次""",
#             "parameters": {
#                 "type": "object",
#                 "required": ["pro","negpro"],
#                 "properties": {
#                     "pro":{
#                         "type": "string",
#                         "description":"""提示词 英文  """
#                     },
#                     "negpro":{
#                         "type": "string",
#                         "description":"负面提示词 英文"
#                     }
#                 }
#             }
#         }
#     },
]




def weather(arguments: Dict[str, Any]) -> Any:
    adm1 = arguments["adm1"]
    adm2 = arguments["adm2"]
    print(adm1 +"+"+ adm2)
    result = tools.wea(adm1,adm2)
    return {"result": result}

def gtime(arguments: Dict[str, Any]) :
    print(arguments.__str__())
    current_time = time.time()
    local_time = time.localtime(current_time)
    formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    return formatted_time


tool_map = {
    "time" : gtime,
    "weather" : weather
}

def chat(messages,input):
    messages.append({
		"role": "user",
		"content": input,	
	})
    finish_reason = None
    while finish_reason is None or finish_reason == "tool_calls":
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            tools=tool, 
        )
        choice = completion.choices[0]
        finish_reason = choice.finish_reason
        if finish_reason == "tool_calls": 
            messages.append(choice.message) 
            for tool_call in choice.message.tool_calls: 
                tool_call_name = tool_call.function.name
                tool_call_arguments = json.loads(tool_call.function.arguments) 
                tool_function = tool_map[tool_call_name] 
                tool_result = tool_function(tool_call_arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call_name,
                    "content": json.dumps(tool_result), 
                })
    
    assistant_message = completion.choices[0].message
    messages.append(assistant_message)
    return (choice.message.content,messages)