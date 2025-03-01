import os
import json
from autogen.agentchat import ConversableAgent
from dotenv import load_dotenv


load_dotenv()


class AgentExecutor:
    def __init__(self, temperature=0.7):
        self.llm_config = {
            "model": os.getenv("MODEL", "gpt-4"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": temperature
        }

    def single_agent(self, name="SingleAgent", human_input_mode="NEVER", agent_params={}):
        agent = ConversableAgent(
            name=name,
            llm_config=self.llm_config,
            function_map = None,
            human_input_mode=human_input_mode,
            **agent_params
        )
        return agent, None
    
    def two_agents(self, agent_json_path="", more_agent_params={}):
        agents = {}
        if len(agent_json_path) == 0:
            agent_json_path = input("Enter agent json path: ")
        with open(agent_json_path, 'r') as f:
            agent_json = json.load(f)
        for agent_params in agent_json["agents"]:
            agents[f'{agent_params["name"]}'] = ConversableAgent(llm_config=self.llm_config, **agent_params, **more_agent_params)
        return agents, agent_json["interactions"]
    

def start():
    agent_selector = input("Enter agent type: ")
    agent_executor = AgentExecutor()
    agents, interactions = eval(f"agent_executor.{agent_selector}()")
    print(agents, interactions)
    agent_action = input("Enter agent action: ")
    if agent_action == "generate_reply":
        agent_reply = agents.generate_reply(messages=[{"content": input("Enter user input: "), "role": "user"}])
        return agent_reply
    elif agent_action == "initiate_chat":
        initiator = input("Enter initiator: ")
        responder = input("Enter responder: ")
        response = agents[initiator].initiate_chat(
            agents[responder], 
            **interactions
        )
        return response


if __name__ == "__main__":
    print(start())

