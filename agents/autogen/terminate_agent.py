import os
import autogen
from conversational_agent import AgentExecutor


class AgentTerminationTechniques:
    def __init__(self, temperature=0.7):
        self.llm_config = {
            "model": os.getenv("MODEL", "gpt-4"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": temperature
        }
        self.agent_executor = AgentExecutor(temperature=temperature)
        

    def terminate_on_counts(self, count_dict):
        agents, interactions = self.agent_executor.two_agents()
        initiator = input("Enter initiator: ")
        responder = input("Enter responder: ")
        response = agents[initiator].initiate_chat(
            agents[responder], 
            **interactions,
            **count_dict
        )
        return response
    
    def terminate_on_messages(self, termination_msg=lambda message: "got you" in message["content"].lower()):
        bramhi_agent, _ =self.agent_executor.single_agent(name="Bramhi", agent_params={
            "system_message": "Your name is bramhi and you are a part of a duo of comedians."
        })
        kishore_agent, _ =self.agent_executor.single_agent(name="Kishore", agent_params={
            "is_termination_msg": termination_msg,
            "system_message": "Your name is kishore and you are a part of a duo of comedians."
        })
        response = kishore_agent.initiate_chat(
            bramhi_agent, 
            message="Bramhi, tell me a joke and then say Got You"
        )
        return response
    

if __name__ == "__main__":
    agent_termination_techniques = AgentTerminationTechniques()
    print(agent_termination_techniques.terminate_on_counts({
        "max_turns": 3
    }))
    print(agent_termination_techniques.terminate_on_counts({
        "max_consecutive_auto_reply": 0
    }))
    print(agent_termination_techniques.terminate_on_messages())
