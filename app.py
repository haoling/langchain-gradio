# [LangChainを使ってチャットボットとお話しするアプリを作ってみる - INOUE-KOBO.COM](https://www.inoue-kobo.com/ai_ml/langchain/)
# [LangChain の Googleカスタム検索 連携を試す｜npaka｜note](https://note.com/npaka/n/nd9a4a26a8932)
# [LangChainを使ってOSINTを自動化する](https://zenn.dev/tatsui/articles/c4b4f796a85395)

import gradio as gr
import os, json, sys, re
from langchain import LLMChain, ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import ConversationalAgent, Tool, AgentExecutor
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

def chat(prefix_messages, user_input):
    llm_gpt = ChatOpenAI(
        temperature=0.95,
        model_name="gpt-3.5-turbo",
    )
    search = GoogleSearchAPIWrapper()
    search_news = GoogleSearchAPIWrapper(google_cse_id=os.environ["GOOGLE_CSE_ID_NEWS"])
    tools = [
        Tool(
            name = "News search",
            func=search_news.run,
            description="最新の話題について答える場合に利用することができます。入力は検索内容です。検索内容は日本語で入力します。"
        ),
        Tool(
            name = "Intermediate Answer",
            func=search.run,
            description="知らないことを調べるときに利用することができます。また、今日の日付や為替レートなど現在の状況についても確認することができます。また、アニメや漫画の質問に答える必要がある場合に役立ちます。入力は検索内容です。検索内容は日本語で入力します。"
        ),
    ]
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    system_message = []
    input = ""
    output = ""
    for message in json.loads(prefix_messages):
        if message['role'] == 'system':
            system_message.append(message['content'])
        if message['role'] == 'assistant':
            output = message['content']
            memory.save_context({"input": input}, {"ouput": output})
        if message['role'] == 'user':
            input = message['content']

    prompt = ConversationalAgent.create_prompt(
        tools=tools,
        prefix="""AssistantはOpenAIによって訓練された大規模な言語モデルです。
Assistantは、簡単な質問に答えるだけでなく、幅広いトピックについて詳細な説明や議論を行うことができるように設計されています。言語モデルとして、Assistantは入力に基づいて人間らしいテキストを生成できるため、自然な会話をすることができ、話題に沿った意味のある回答を提供することができます。
Assistantは常に学習して改善し、その能力は常に進化しています。Assistantは大量のテキストを処理し理解することができ、この知識を活用して、幅広い質問に正確かつ有益な回答を提供できます。また、Assistantは受け取った入力に基づいて独自のテキストを生成することができ、幅広いトピックについて説明や説明を行い、議論に参加することができます。
全体的に言えば、Assistantは幅広いタスクに役立ち、多くのトピックに対して貴重な洞察や情報を提供することができる強力なツールです。特定の質問に対する支援が必要な場合や、特定のトピックについての会話をしたい場合、Assistantがサポートしてくれます。

TOOLS:
------

Assistantは下記のツールにアクセスできます:""",
        format_instructions="""ツールを使うには、下記のフォーマットに従ってください:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: アクションへの入力です。
Observation: Actionの結果です。
```
Humanに結果を返信する必要があるとき、またはツールを使う必要が無いときに、必ず下記のフォーマットに従う必要があります:
```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```""",
        suffix="""When you have a response to say to the Human, you MUST role-play the character according to the Character role guidelines:
%s

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}""" % ("\n".join(system_message)),
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    print(prompt)
    llm_chain = LLMChain(llm=llm_gpt, prompt=prompt)
    agent = ConversationalAgent(llm_chain=llm_chain, allowed_tools=[tool.name for tool in tools])
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
                    return "Warning", re.sub('^Could not parse LLM output: `(.*)`$', r'\1', err), cb.total_tokens
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