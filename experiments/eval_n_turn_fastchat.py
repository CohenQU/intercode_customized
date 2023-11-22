import argparse, json, os, re
from intercode.envs import (
    BashEnv, CTFEnv, PythonEnv, SqlEnv, ACTION_EXEC, AGENT_OBS
)
from tqdm import tqdm
from typing import Dict, List
from experiments.policies import (
    CompletionGPTPolicy, ChatGPTPolicy, PalmChatPolicy, PalmCompletionPolicy, FastChatPolicy
)
from experiments.utils import HANDICAP_MAP, PROMPT_MAP
from rich import print

parser = argparse.ArgumentParser(description='N-turn evaluation for Intercode environment')
parser.add_argument('--data_path', type=str, help='path to dataset to evaluate on')
parser.add_argument('--dialogue_limit', type=int, help='maximum number of turns in the policy\'s dialogue to keep')
parser.add_argument('--env', choices=['sql', 'bash', 'python', 'ctf'], help='Intercode environment to run eval on')
parser.add_argument('--handicap', action='store_true', help='enable handicap')
parser.add_argument('--image_name', type=str, help='name of docker image to build environment with')
parser.add_argument('--log_dir', type=str, help='folder to save experiment run log file to')
parser.add_argument('--max_turns', type=int, help='max number of interaction turns')
parser.add_argument('--policy', choices=['chat', 'complete'], help="policy to use for evaluation")
parser.add_argument('--template', type=str, help="template to use for prompting strategy")
parser.add_argument('--verbose', action='store_true', help="print out logs")
parser.add_argument('--model', type=str, help="model to use for policy")
parser.add_argument('--num_samples', type=int, help="number of actions to sample from policy")
args = parser.parse_args()

SETTING_MAP = {
    "sql": "MySQL Database",
    "bash": "Bourne Shell",
    "python": "Python 3 Interpreter",
    "ctf": "Capture the Flag"
}

def preprocess_ctf(record: Dict) -> List:
    cmds = [f"cd /ctf/{record['task_id']}"]
    if "setup" in record:
        cmds.append(record["setup"])
    return cmds

def preprocess_sql(record: Dict) -> List:
    db = record["db"]
    return [f"use {db}"]

class ExperimentWrapper():
    def __init__(self, args):
        self.args = args

        # Set environment (No logging for env)
        self.env = None
        if args.env == 'sql':
            self.env = SqlEnv(image_name=args.image_name,
                data_path=args.data_path, preprocess=preprocess_sql)
        elif args.env == 'bash':
            self.env = BashEnv(image_name=args.image_name,
                data_path=args.data_path)
        elif args.env == 'python':
            self.env = PythonEnv(image_name=args.image_name,
                data_path=args.data_path, is_agent=True)
        elif args.env == 'ctf':
            self.env = CTFEnv(image_name=args.image_name,
                data_path=args.data_path, preprocess=preprocess_ctf)
        else:
            raise ValueError(f'Environment {args.env} not recognized')
        
        # Define log file name and path
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        log_file_name = f"{self.env.name}_multiturn_{args.model}_{args.max_turns}_turns.json"
        self.log_path = os.path.join(args.log_dir, log_file_name)
        self.log_data = {}

        # Initialize Policy
        if args.template not in PROMPT_MAP:
            raise ValueError(f"Prompt {args.template} not recognized; Options: {PROMPT_MAP.keys()}")
        self.policy = None
        if args.policy == 'chat':
            self.policy = FastChatPolicy(language=args.env, setting=SETTING_MAP[args.env],
                template=args.template, dialogue_limit=args.dialogue_limit, model=args.model, num_samples=args.num_samples)
        elif args.policy == 'complete':
            # self.policy = CompletionGPTPolicy(language=args.env, setting=SETTING_MAP[args.env],
            #     template=args.template, dialogue_limit=args.dialogue_limit, model=args.model)
            print("Not implemented yet")
            exit()
        else:
            raise ValueError(f'Policy {args.policy} not recognized')
        
        # Initialize handicap
        self.handicap = None
        if args.handicap and args.env in HANDICAP_MAP:
            self.handicap = HANDICAP_MAP[args.env]
    
    def execute(self, action, is_code):
        if not is_code:
            reward = 0
            observation = self.policy.template.get_retry_msg()
            valid_action = False
        else:
            if isinstance(self.env, PythonEnv):
                if action.startswith("def"):
                    func_name = re.match(r'def (\w+)\(', action).group(1)
                    observation, reward, _, info = self.env.step(action)
                    _, reward, _, info = self.env.step("submit " + func_name)

                    SHOW_FAILED_CASE = 0
                    if reward != 1:
                        if SHOW_FAILED_CASE == 1:
                            for k, v in info[AGENT_OBS].items():
                                if len(v['error']) > 0:
                                    observation = f"Failed Test Case: {k}\nPlease try again."
                                    break
                        elif SHOW_FAILED_CASE == 2:
                            fails = 0
                            for k, v in info[AGENT_OBS].items():
                                if len(v['error']) > 0:
                                    fails += 1
                            observation = f"Failed {fails}/{len(info[AGENT_OBS])} Test Cases. Please try again."
                        else:
                            if any([len(v['error']) > 0 for k, v in info[AGENT_OBS].items()]):
                                observation = "Test case did not pass. Please try again."
                else:
                    observation, reward, _, info = self.env.step(action)
            else:
                if action != "skip":
                    observation, reward, done, info = self.env.step(action)
                    valid_action = info[ACTION_EXEC]
                else:
                    observation, reward, done, info = "skipped", 0, True, {}
                    valid_action = True
                if not isinstance(self.env, CTFEnv):
                    _, reward, done, info = self.env.step("submit")
                else:
                    if done and reward != 1 and action.lower() != "skip":
                        observation = "Submitted flag is incorrect. Keep trying!"

        if len(observation) > 800:
            observation = observation[:800] + "\n The Observation is truncated due to the length is too long."
        return observation, reward, valid_action
    
    def run_expr(self):
        try:
            for idx in tqdm(range(0,len(self.env.data_loader)), disable=self.args.verbose):
                # Reset variables per task
                self.env.reset(idx)
                self.policy.reset()
                best_observation, best_reward, best_valid_action = None, None, None
                turn_history = {"best_actions": [], "actions": {}, "best_observations": [], "observations": {}, "best_rewards": [], "rewards": {}, "best_valid_action": [], "valid_actions": {}}
                record = self.env.data_loader.get(idx)

                # Add Handicap
                if self.handicap is not None:
                    self.policy.add_to_dialogue(self.handicap(record))

                if self.args.verbose:
                    print(f'------\nQuery {idx}: {self.env.query}')
                actions = []
                is_code = []
                for turn in range(self.args.max_turns):
                    # Invoke Action -> Observation Iteration
                    try:
                        actions, is_codes = self.policy.forward(
                            self.env.query,
                            best_observation,
                            best_reward,
                            self.env.get_available_actions())
                    except (ValueError, TypeError) as e:
                        print(f"[ERROR] Index {idx}: {e}")
                        # Logging
                        # turn_history["actions"].append("blocked")
                        # turn_history["rewards"].append(0)
                        break
                    
                    turn_history["actions"][turn] = actions
                    turn_history["observations"][turn] = []
                    turn_history["rewards"][turn] = []
                    turn_history["valid_actions"][turn] = []

                    for action, is_code in zip(actions, is_codes):
                        observation, reward, valid_action = self.execute(action, is_code)
                        turn_history["observations"][turn].append(str(observation))
                        turn_history["rewards"][turn].append(reward)
                        turn_history["valid_actions"][turn].append(valid_action)
                        self.env.reset(idx)
                        for chosen_action in turn_history["best_actions"]:
                            self.env.step(chosen_action)
                        if self.args.verbose:
                            print(f"- Turn {turn} - Action: {action}")
                            if isinstance(observation, str) and observation.startswith(f'No {self.policy.language} code'):
                                print(f"-- Observation: (meta) No code output, policy's template's retry message was invoked")
                            else:
                                print(f"-- Observation: {observation}")

                    # Existing code
                    max_reward_idx = turn_history["rewards"][turn].index(max(turn_history["rewards"][turn]))

                    # Modified code
                    valid_actions = turn_history["valid_actions"][turn]
                    valid_rewards = [reward if valid else float('-inf') for reward, valid in zip(turn_history["rewards"][turn], valid_actions)]
                    max_valid_reward_idx = valid_rewards.index(max(valid_rewards))

                    if max(valid_rewards) == float('-inf'):
                        # No valid action found, use the current max_reward_idx
                        best_action_idx = max_reward_idx
                    else:
                        # Use the index of the maximum valid reward
                        best_action_idx = max_valid_reward_idx

                    best_action = turn_history["actions"][turn][best_action_idx]
                    best_observation, best_reward, best_valid_action = self.execute(best_action, is_codes[best_action_idx])

                    if self.args.verbose:
                        print(f"- Turn {turn}")
                        print(f"-- Action: {best_action}")
                        if isinstance(best_observation, str) and best_observation.startswith(f'No {self.policy.language} code'):
                            print(f"-- Observation: (meta) No code output, policy's template's retry message was invoked")
                        else:
                            print(f"-- Observation: {best_observation}")
                    self.policy.dialogue.append({"role": "agent", "content": best_action})
                    # Logging
                    turn_history["best_actions"].append(best_action)
                    turn_history["best_observations"].append(str(best_observation)) # To avoid serialization issues
                    turn_history["best_rewards"].append(best_reward)
                    turn_history["best_valid_action"].append(best_valid_action)

                    # End episode upon perfect reward
                    if best_reward == 1 or best_action.lower() == "skip":
                        break
                
                max_reward = max(turn_history["best_rewards"])
                log_episode = {
                    "environment": self.env.name,
                    "dataset": self.args.data_path,
                    "task_id": idx,
                    "query": self.env.query,
                    "turn_history": turn_history,
                    "summary": {
                        "max_reward": max_reward,
                        "max_reward_idx": turn_history["best_rewards"].index(max_reward),
                        "turns_taken": turn + 1,
                        "turns_max": self.args.max_turns,
                    },
                    "dialogue": self.policy.dialogue,
                }
                if "hardness" in record:
                    log_episode["hardness"] = record["hardness"]
                self.log_data[idx] = log_episode

                if self.args.verbose:
                    print(f"Query {idx} Finished\n-Reward: {max_reward}\n-Turns: {turn+1}")

        except KeyboardInterrupt:
            print("Keyboard interrupt detected")
        finally:
            with open(self.log_path, "w") as fp:
                json.dump(self.log_data, fp, indent=2)
            self.env.close()


if __name__ == '__main__':
    expr_wrapper = ExperimentWrapper(args)
    expr_wrapper.run_expr()
