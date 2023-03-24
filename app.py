# [LangChainを使ってチャットボットとお話しするアプリを作ってみる - INOUE-KOBO.COM](https://www.inoue-kobo.com/ai_ml/langchain/)
# [LangChain の Googleカスタム検索 連携を試す｜npaka｜note](https://note.com/npaka/n/nd9a4a26a8932)
# [LangChainを使ってOSINTを自動化する](https://zenn.dev/tatsui/articles/c4b4f796a85395)

# https://github.com/hwchase17/langchain/blob/master/langchain/agents/conversational_chat/base.py
# https://github.com/hwchase17/langchain/blob/master/langchain/agents/conversational_chat/prompt.py

import ptvsd
import gradio as gr
import os, json, sys, re
from typing import Any, List, Optional, Sequence, Tuple
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentExecutor
from langchain.agents.conversational_chat.base import ConversationalChatAgent, AgentOutputParser
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    PromptValue,
    SystemMessage,
)

ptvsd.enable_attach()

llm_gpt = ChatOpenAI(
    temperature=0.95,
    model_name="gpt-3.5-turbo",
)
search = GoogleSearchAPIWrapper()
search_news = GoogleSearchAPIWrapper(google_cse_id=os.environ["GOOGLE_CSE_ID_NEWS"], k=1)
tools = [
    Tool(
        name = "News search",
        func=search_news.run,
        description="読売新聞のサイトを検索することができます。最新の話題について答える場合に優先的に利用することができます。入力は検索内容です。検索内容は日本語で入力します。"
    ),
    Tool(
        name = "Google Search",
        func=search.run,
        description="知らないことを調べるときに利用することができます。また、今日の日付や為替レートなど現在の状況についても確認することができます。また、アニメや漫画の質問に答える必要がある場合に役立ちます。入力は検索内容です。検索内容は日本語で入力します。"
    ),
]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

PREFIX = """アシスタントは、OpenAIによって訓練された大型言語モデルです。
アシスタントは、単純な質問に答えるだけでなく、幅広いトピックについて詳しい説明や議論を提供することができるよう設計されています。言語モデルとして、アシスタントは受け取った入力に基づいて人間らしいテキストを生成することができ、自然な会話を行い、話題に関連する意味のある回答を提供することができます。
アシスタントは常に学習して改善しており、その能力は常に進化しています。多くのテキストを処理し理解することができ、これらの知識を利用して幅広い質問に対して正確で情報量の多い回答を提供することができます。さらに、アシスタントは、受け取った入力に基づいて自分自身のテキストを生成することができ、幅広いトピックに関して説明や説明を行い、議論を展開することができます。
全体として、アシスタントは幅広いタスクに対応することができ、幅広いトピックについて貴重な情報を提供することができる強力なシステムです。特定の質問についてのヘルプが必要な場合や、特定のトピックについての会話をしたい場合には、アシスタントがお手伝いします。"""

FORMAT_INSTRUCTIONS = """RESPONSE FORMAT INSTRUCTIONS
----------------------------
When responding to me please, please output a response in one of two formats:
**Option 1:**
Use this if you want the human to use a tool.
Markdown code snippet formatted in the following schema:
```json
{{{{
    "action": string \\ The action to take. Must be one of {tool_names}
    "action_input": string \\ The input to the action
}}}}
```
**Option #2:**
Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:
```json
{{{{
    "action": "Final Answer",
    "action_input": string \\ You should put what you want to return to use here. Plaese follow character role guideline.
}}}}
```"""

SUFFIX_TEMPLATE = """TOOLS
------
Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:

{{tools}}

{format_instructions}

CHAT HISTORY
--------------------
Here is the chat history.
%s

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):
{{{{input}}}}"""

class MyAgentOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Any:
        if '"action"' in text and '"action_input"' in text:
            text = re.sub('\n', '', text)
            text = '{' + text.split("{", 1)[1]
            print(text)
        return super().parse(text)

def chat(prefix_messages, user_input):
    prompt = ConversationalChatAgent.create_prompt(tools=tools, system_message=PREFIX)

    memory.clear()
    input = ""
    output = ""
    system_messages = []
    for message in json.loads(prefix_messages):
        if message['role'] == 'system':
            system_messages.append(SystemMessage(content=message['content']))
        if message['role'] == 'assistant':
            output = message['content']
            memory.save_context({"input": input}, {"ouput": output})
        if message['role'] == 'user':
            input = message['content']
    prompt.messages[0:0] = system_messages
    print(memory)

    llm_chain = LLMChain(llm=llm_gpt, prompt=prompt)
    print(llm_chain)
    agent = ConversationalChatAgent(llm_chain=llm_chain, allowed_tools=[tool.name for tool in tools], output_parser=MyAgentOutputParser())
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

    with get_openai_callback() as cb:
        for i in range(3):
            try:
                response = agent_executor.run(input=user_input)
                print("total_tokens: {0}".format(cb.total_tokens))
                return "Success", response, cb.total_tokens

            except ValueError as e:
                print(e)
                err = "{0}".format(e)
                if err.startswith("Could not parse LLM output: "):
                    print("total_tokens: {0}".format(cb.total_tokens))
                    return "Warning", re.sub('^Could not parse LLM output: (.*)$', r'\1', err), cb.total_tokens
                else:
                    continue
            except Exception as e:
                print(e)
                if i == 2:
                    print("total_tokens: {0}".format(cb.total_tokens))
                    return "Failure", e, cb.total_tokens
                else:
                    continue



chatbot = gr.Chatbot().style()

app = gr.Blocks()
with app:
    with gr.Column():
        input1 = gr.TextArea(label=f"prefix_messages JSON")
        input2 = gr.TextArea(label=f"Text")
        submit = gr.Button("送信", variant="primary")
        output1 = gr.Textbox(label="Output Message")
        output2 = gr.Textbox(label="response")
        output3 = gr.Textbox(label="total_tokens")
        submit.click(
            chat,
            inputs=[input1, input2],
            outputs=[output1, output2, output3],
            api_name="chat"
        )
if __name__ == '__main__':
    app.launch(server_name = "0.0.0.0")