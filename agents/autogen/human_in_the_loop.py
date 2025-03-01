from autogen import ConversableAgent
from conversational_agent import AgentExecutor


def agent_talks_to_human():
    agent_executor = AgentExecutor()
    agent_with_number, _ = agent_executor.single_agent(
        name="agent_with_number",
        human_input_mode="TERMINATE",
        agent_params={
            "system_message": "You are playing a game of guess-my-number. You have the number 53 in your mind, and I will try to guess it. If I guess too high, say 'too high', if I guess too low, say 'too low'. ",
            "is_termination_msg": lambda msg: "53" in msg["content"]
        }
    )
    agent_guess_number, _ = agent_executor.single_agent(
        name="agent_guess_number",
        human_input_mode="NEVER",
        agent_params={
            "system_message": "I have a number in my mind, and you will try to guess it. "
            "If I say 'too high', you should guess a lower number. If I say 'too low', "
            "you should guess a higher number. ",
            "is_termination_msg": lambda msg: "53" in msg["content"]
        }
    )
    human_proxy = ConversableAgent(
        "human_proxy",
        llm_config=False,  # no LLM used for human proxy
        human_input_mode="ALWAYS"
    )

    # Start a chat with the agent with number with an initial guess.
    human_proxy.initiate_chat(
        agent_with_number,  # this is the same agent with the number as before
        message="10",
    )

    agent_with_number.initiate_chat(
        agent_guess_number,
        message="I have a number between 1 and 100. Guess it!",
    )






if __name__ == "__main__":
    print(agent_talks_to_human())

