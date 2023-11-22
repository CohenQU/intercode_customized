import re

from typing import Tuple

from .utils import ACTION_PARSER_MAP, PROMPT_MAP, CompletionGPT, ChatGPT, PalmChat, PalmCompletion, HFChat

import requests
import os, json, sys, time, re, math, random, datetime, argparse, requests
from typing import List, Dict, Any, Union

from fastchat.model.model_adapter import get_conversation_template

# import TimeoutException
from requests.exceptions import Timeout, ConnectionError

class Agent:
    def __init__(self, **configs) -> None:
        self.name = configs.pop("name", None)
        self.src = configs.pop("src", None)
        pass

    # def create_session(self) -> Session:
    #     return Session(self.inference)

    def inference(self, history: List[dict]) -> str:
        raise NotImplementedError
    
class Prompter:
    @staticmethod
    def get_prompter(prompter_name: Union[str, None]):
        # check if prompter_name is a method and its variable
        if not prompter_name:
            return None
        if hasattr(Prompter, prompter_name) and callable(getattr(Prompter, prompter_name)):
            return getattr(Prompter, prompter_name)
    
    @staticmethod
    def claude(messages: List[Dict[str, str]]):
        prompt = ""
        role_dict = {
            "user": "Human",
            "agent": "Assistant",
        }
        for item in messages:
            prompt += f"{role_dict[item['role']]}: {item['content']}\n\n"
        prompt += "Assistant:"
        return {"prompt": prompt}

    @staticmethod
    def openchat_v3_1(messages: List[Dict[str, str]]):
        prompt = "Assistant is GPT4<|end_of_turn|>"
        role_dict = {
            "user": "User: {content}<|end_of_turn|>",
            "agent": "Assistant: {content}<|end_of_turn|>",
        }
        for item in messages:
            prompt += role_dict[item['role']].format(content=item['content'])
        prompt += "Assistant:"
        return {"prompt": prompt}
    
    @staticmethod
    def openchat_v3_2(messages: List[Dict[str, str]]):
        prompt = ""
        role_dict = {
            "user": "GPT4 User: {content}<|end_of_turn|>",
            "agent": "GPT4 Assistant: {content}<|end_of_turn|>",
        }
        for item in messages:
            prompt += role_dict[item['role']].format(content=item['content'])
        prompt += "GPT4 Assistant:"
        return {"prompt": prompt}

class FastChatAgent(Agent):
    def __init__(self, model_name, controller_address=None, worker_address=None, temperature=0, max_new_tokens=32, top_p=0, prompter=None, args=None, **kwargs) -> None:
        if controller_address is None and worker_address is None:
            raise ValueError("Either controller_address or worker_address must be specified.")
        self.controller_address = controller_address
        self.worker_address = worker_address
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.prompter = Prompter.get_prompter(prompter)
        self.args = args or {}
        super().__init__(**kwargs)

    def inference(self, history: List[dict]) -> str:
        if self.worker_address:
            worker_addr = self.worker_address
        else:
            controller_addr = self.controller_address
            worker_addr = controller_addr
        if worker_addr == "":
            return
        gen_params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "echo": False,
            "top_p": self.top_p,
            **self.args
        }
        if self.prompter:
            prompt = self.prompter(history)
            gen_params.update(prompt)
        else:
            conv = get_conversation_template(self.model_name)

            for history_item in history:
                role = history_item["role"]
                content = history_item["content"]
                if role == "user" or role == "system":
                    conv.append_message(conv.roles[0], content)
                elif role == "agent":
                    conv.append_message(conv.roles[1], content)
                else:
                    raise ValueError(f"Unknown role: {role}")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            gen_params.update({
                "prompt": prompt,
                "stop": conv.stop_str,
                "stop_token_ids": conv.stop_token_ids,
            })
        headers = {"User-Agent": "FastChat Client"}
        for _ in range(3):
            try:
                response = requests.post(
                    controller_addr + "/worker_generate_stream",
                    headers=headers,
                    json=gen_params,
                    stream=True,
                    timeout=120,
                )
                text = ""
                for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if line:
                        text = json.loads(line)["text"]
                return text
            # if timeout or connection error, retry
            except Timeout: 
                print("Timeout, retrying...")
            except ConnectionError:
                print("Connection error, retrying...")
            time.sleep(5)
        else:
            raise Exception("Timeout after 3 retries.")

class BasePolicy:
    def __init__(self):
        pass

    def forward(query, observation, available_actions):
        raise NotImplementedError
    
class HumanPolicy(BasePolicy):
    def __init__(self):
        super().__init__()
    
    def forward(self, query, observation, available_actions):
        action = input('> ')
        return action
    
class CompletionGPTPolicy(BasePolicy):
    def __init__(self, language: str, setting: str, template: str, dialogue_limit: int = None, model: str = "text-davinci-003", response_limit: int = 500):
        super().__init__()
        self.language = language.upper()
        self.setting = setting
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.response_limit = response_limit
        self.model = model
        self.handicap = ""
        self.dialogue_limit = dialogue_limit
    
    def reset(self):
        self.prompt = None
        self.dialogue = {}

    def add_to_dialogue(self, handicap: str):
        self.handicap = handicap

    def forward(self, query, observation, reward, available_actions) -> Tuple[str, bool]:
        if self.prompt is None:
            # First Turn
            prompt = self.prompt = self.template.get_init_msg() + self.handicap + self.template.get_query_msg(query)
        else:
            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > self.response_limit:
                observation = observation[:self.response_limit]
            elif isinstance(observation, list) and len(observation) > 50:
                observation = observation[:50]
            if isinstance(observation, str) and observation == "" or isinstance(observation, list) and len(observation) == 0:
                observation = "No output"
            self.dialogue['reward'] = reward
            self.dialogue['observations'] = observation
            # N-th Turn
            prompt = self.prompt + self.template.get_obs_msg(self.dialogue)

        # Retrieve Completion GPT
        actions = CompletionGPT(prompt, model=self.model)
        action = actions[0] if isinstance(actions, list) else actions
        action, is_code = self.action_parser(action)
        self.dialogue['actions'] = action
        return action, is_code

class ChatGPTPolicy(BasePolicy):
    def __init__(self, language: str, setting: str, template: str, dialogue_limit: int = None, model: str = "gpt-3.5-turbo", response_limit: int = 1000, num_samples: int = 1):
        super().__init__()
        self.language = language.upper()
        self.setting = setting
        self.dialogue_limit = dialogue_limit
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.model = model
        self.response_limit = response_limit
        self.num_samples = num_samples
        print(f"Teacher Model is {self.model}")
    
    def reset(self):
        self.dialogue = [{"role": "system", "content": self.template.get_init_msg()}]

    def add_to_dialogue(self, handicap: str):
        self.dialogue.append({"role": "system", "content": handicap})

    def forward(self, query, observation, reward, available_actions) -> Tuple[str, bool]:
        print("Teacher Policy Forward")
        # Append response to dialogue
        self.template.get_query_msg(query)
        if self.dialogue[-1]["role"] == "system":
            # First Turn
            self.dialogue.append({"role": "user", "content": self.template.get_query_msg(query)})
        else:
            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > self.response_limit:
                observation = observation[:self.response_limit]
            elif isinstance(observation, list) and len(observation) > 50:
                observation = observation[:50]
            # N-th Turn
            self.dialogue.append({"role": "user", "content": self.template.get_obs_msg(observation, reward)})
            # Only keep {self.dialogue_limit} most recent messages
            if self.dialogue_limit and len(self.dialogue) - 2 > self.dialogue_limit:
                self.dialogue = self.dialogue[:2] + self.dialogue[-self.dialogue_limit:]

        # Retrieve Action from ChatGPT
        raw_actions = ChatGPT(self.dialogue, model=self.model, num_samples=self.num_samples)
        actions = []
        is_codes = []
        for action in raw_actions:
            action, is_code = self.action_parser(action)
            if action not in actions:
                actions.append(action)
                is_codes.append(is_code)
                
        return actions, is_codes
    
class FastChatPolicy(BasePolicy):
    def __init__(self, language: str, setting: str, template: str, dialogue_limit: int = None, model: str = "", response_limit: int = 1000, num_samples: int = 1):
        super().__init__()
        self.language = language.upper()
        self.setting = setting
        self.dialogue_limit = dialogue_limit
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.model = model
        self.response_limit = response_limit
        self.num_samples = num_samples
        self.agent = FastChatAgent(model_name=model, controller_address="http://localhost:21002", worker_address=None, temperature=0.7, max_new_tokens=response_limit, top_p=1.0, prompter=None, args=None, name="FastChatAgent")
        
    def reset(self):
        self.dialogue = [{"role": "system", "content": self.template.get_init_msg()}]

    def add_to_dialogue(self, handicap: str):
        self.dialogue[-1]["content"] += handicap
        # self.dialogue.append({"role": "system", "content": handicap})

    def forward(self, query, observation, reward, available_actions) -> Tuple[str, bool]:
        print("Student Policy Forward")
        # Append response to dialogue
        if self.dialogue[-1]["role"] == "system":
            # First Turn
            self.dialogue[-1]["role"] = "user"
            self.dialogue[-1]["content"] += self.template.get_query_msg(query)
            # self.dialogue.append({"role": "user", "content": self.template.get_query_msg(query)})
        else:
            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > self.response_limit:
                observation = observation[:self.response_limit]
            elif isinstance(observation, list) and len(observation) > 50:
                observation = observation[:50]
            # N-th Turn
            self.dialogue.append({"role": "user", "content": self.template.get_obs_msg(observation, reward)})
            # Only keep {self.dialogue_limit} most recent messages
            if self.dialogue_limit and len(self.dialogue) - 2 > self.dialogue_limit:
                self.dialogue = self.dialogue[:2] + self.dialogue[-self.dialogue_limit:]

        # Retrieve Action from ChatGPT
        actions = []
        is_codes = []
        for i in range(self.num_samples):
            raw_actions = self.agent.inference(self.dialogue)
            action = raw_actions[0] if isinstance(raw_actions, list) else raw_actions
            action, is_code = self.action_parser(action)
            if action not in actions:
                actions.append(action)
                is_codes.append(is_code)

        return actions, is_codes
    
class FastChatPolicy_Single(BasePolicy):
    def __init__(self, language: str, setting: str, template: str, dialogue_limit: int = None, model: str = "", response_limit: int = 1000, num_of_actions: int = 1):
        super().__init__()
        self.language = language.upper()
        self.setting = setting
        self.dialogue_limit = dialogue_limit
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.model = model
        self.response_limit = response_limit


        self.agent = FastChatAgent(model_name=model, controller_address="http://localhost:21002", worker_address=None, temperature=0.7, max_new_tokens=response_limit, top_p=1, prompter=None, args=None, name="FastChatAgent")
        
    def reset(self):
        self.dialogue = [{"role": "system", "content": self.template.get_init_msg()}]

    def add_to_dialogue(self, handicap: str):
        self.dialogue[-1]["content"] += handicap
        # self.dialogue.append({"role": "system", "content": handicap})

    def forward(self, query, observation, reward, available_actions) -> Tuple[str, bool]:
        print("Student Policy Forward")
        # Append response to dialogue
        if self.dialogue[-1]["role"] == "system":
            # First Turn
            self.dialogue[-1]["role"] == "user"
            self.dialogue[-1]["content"] += self.template.get_query_msg(query)
            # self.dialogue.append({"role": "user", "content": self.template.get_query_msg(query)})
        else:
            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > self.response_limit:
                observation = observation[:self.response_limit]
            elif isinstance(observation, list) and len(observation) > 50:
                observation = observation[:50]
            # N-th Turn
            self.dialogue.append({"role": "user", "content": self.template.get_obs_msg(observation, reward)})
            # Only keep {self.dialogue_limit} most recent messages
            if self.dialogue_limit and len(self.dialogue) - 2 > self.dialogue_limit:
                self.dialogue = self.dialogue[:2] + self.dialogue[-self.dialogue_limit:]

        # Retrieve Action from ChatGPT
        actions = self.agent.inference(self.dialogue)
        action = actions[0] if isinstance(actions, list) else actions
        action, is_code = self.action_parser(action)
        self.dialogue.append({"role": "agent", "content": action})
        return action, is_code
    
class PalmChatPolicy(BasePolicy):
    def __init__(self, language: str, setting: str, template: str, dialogue_limit: int = None, model: str = "models/chat-bison-001", response_limit: int = 1000):
        super().__init__()
        self.language = language.upper()
        self.setting = setting
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.response_limit = response_limit
        self.model = model
        self.handicap = ""
    
    def reset(self):
        self.chatbot = PalmChat(self.model)
        self.dialogue = []

    def add_to_dialogue(self, handicap: str):
        self.handicap = handicap

    def forward(self, query, observation, reward, available_actions) -> Tuple[str, bool]:
        if len(self.dialogue) == 0:
            # First Turn
            self.dialogue = [{"author": "0", "content": self.template.get_init_msg() + self.handicap + self.template.get_query_msg(query)}]
            self.chatbot.init_chat(init_message=self.dialogue)
        else:
            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > self.response_limit:
                observation = observation[:self.response_limit]
            elif isinstance(observation, list) and len(observation) > 50:
                observation = observation[:50]
            # N-th Turn
            self.chatbot.reply(self.template.get_obs_msg(observation, reward))

        # Retrieve Action from PalmChat
        action = self.chatbot.get_response()
        action, is_code = self.action_parser(action)
        return action, is_code
    

class PalmCompletionPolicy(BasePolicy):
    def __init__(self, language: str, setting: str, template: str, dialogue_limit: int = None, model: str = "models/text-bison-001", response_limit: int = 1000):
        super().__init__()
        self.language = language.upper()
        self.setting = setting
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.response_limit = response_limit
        self.model = model
        self.handicap = ""
    
    def reset(self):
        self.prompt = None
        self.dialogue = {}

    def add_to_dialogue(self, handicap: str):
        self.handicap = handicap

    def forward(self, query, observation, reward, available_actions) -> Tuple[str, bool]:
        if self.prompt is None:
            # First Turn
            prompt = self.prompt = self.template.get_init_msg() + self.handicap + self.template.get_query_msg(query)
        else:
            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > self.response_limit:
                observation = observation[:self.response_limit]
            elif isinstance(observation, list) and len(observation) > 50:
                observation = observation[:50]
            if isinstance(observation, str) and observation == "" or isinstance(observation, list) and len(observation) == 0:
                observation = "No output"
            self.dialogue['reward'] = reward
            self.dialogue['observations'] = observation
            # N-th Turn
            prompt = self.prompt + self.template.get_obs_msg(self.dialogue)

        # Retrieve Action from PalmChat
        actions = PalmCompletion(prompt, model=self.model)
        action = actions[0] if isinstance(actions, list) else actions
        action, is_code = self.action_parser(action)
        self.dialogue['actions'] = action
        return action, is_code

class HFChatPolicy(BasePolicy):
    def __init__(self, language: str, setting: str, template: str, dialogue_limit: int = None, model: str = "gpt-3.5-turbo", response_limit: int = 1000):
        super().__init__()
        self.language = language.upper()
        self.setting = setting
        self.dialogue_limit = dialogue_limit
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.model = model
        self.response_limit = response_limit
    
    def reset(self):
        self.dialogue = []

    def add_to_dialogue(self, handicap: str):
        self.dialogue.append({"role": "system", "content": handicap})

    def forward(self, query, observation, reward, available_actions) -> Tuple[str, bool]:
        # Append response to dialogue
        if len(self.dialogue) == 0:
            # First Turn
            self.dialogue = [{"role": "<|system|>", "content": "You are an expert in Bash."}] 
            self.dialogue.append({"role": "<|user|>", "content": self.template.get_init_msg() + self.template.get_query_msg(query)})
        else:
            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > self.response_limit:
                observation = observation[:self.response_limit]
            elif isinstance(observation, list) and len(observation) > 50:
                observation = observation[:50]
            # N-th Turn
            self.dialogue.append({"role": "<|user|>", "content": self.template.get_obs_msg(observation, reward)})
            # Only keep {self.dialogue_limit} most recent messages
            if self.dialogue_limit and len(self.dialogue) - 2 > self.dialogue_limit:
                self.dialogue = self.dialogue[:2] + self.dialogue[-self.dialogue_limit:]
        
        chat = ""
        for d in self.dialogue:
            chat += f"{d['role']} {d['content']} <|end|>\n"
        actions = HFChat(chat + "<|assistant|>")
        action = actions[0] if isinstance(actions, list) else actions
        action, is_code = self.action_parser(action)
        self.dialogue.append({"role": "<|assistant|>", "content": action})
        return action, is_code