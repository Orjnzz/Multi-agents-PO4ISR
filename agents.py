import math
import random
import re
import json
from opt.request import Request
from opt.config import init_config
from opt.utils import extract_item_json_list, extract_wrapped_json, ndcg, extract_edit_prompt
from agno.agent import Agent

config = init_config()
req = Request(config)

class EvalAgent(Agent):
    def __init__(self,model,name='EvalAgent'):
        super().__init__(
            model=model,
            description="Using initial prompt to rerank candidate set based on user session interactions.",
            name=name
        )
    def evaluate(self,prompt,session_data):
        input_str = session_data['input']
        json_req = (
                    "Output Requirements: Provide JUST ONLY a clear and concise ranking of the entire candidate set(NOT EXPLAIN). "
                    "Ensure that all items in the candidate set are assigned a rank, and that the ranking is based solely on the items within the candidate set. "
                    "Importantly, The output format for the ranking must be in a JSON structure as follow: "
                    '{ "1": "<item_1>", "2": "<item_2>", ... "20": "<item_20>" } '
                    "and the name of candidate item in response MUSTN'T CHANGE!"
                )
        full_prompt = f"{prompt}\n{json_req}\n{input_str}"
        response = req.request(user=full_prompt,json_mode=True)
        return response

class ErrorCheckingAgent(Agent):
    def __init__(self,model,name='ErrorCheckingAgent'):
        super().__init__(
            model=model,
            description="Validates and corrects LLM responses.",
            name=name
        )
    def check_error(self,response,session_data):
        try: 
            data = json.loads(response)
            return response
        except json.JSONDecodeError:
            candidate_items = re.findall(r'\d+\."([^"]+)"', session_data['input'].split('\nCandidate Set: ')[1])
            correct_prompt = (
                f"The following response is invalid: {response}. Please correct it to be ONLY as a JSON structure as follow: "
                f'{{"1": "<item_1>", "2": "<item_2>", ... "20": "<item_20>"}} '
                f"from the candidate set {candidate_items}. Wrap the corrected JSON with <START> and <END> tags."
            )
            # print("Correct prompt: ", correct_prompt)
            corrected_response = self.run(correct_prompt).content
            # print("corrected_response:", corrected_response)
            wrapped_json = extract_wrapped_json(corrected_response)
            if wrapped_json:
                return wrapped_json
            return None


class DetectErrrorAgent:
    def detect_error(self, response, target):
        result_list = extract_item_json_list(response, target)
        # print("Detect error processing: ")
        # print("- Response: ", response)
        # print("- Target: ", target)
        # print("- Result: ", result_list)
        if not result_list:
            return True
        threshold = 6
        rank = int(result_list[-1])
        return rank >= threshold

class InferReasonAgent(Agent):
    def __init__(self,model,name='InferReasonAgent'):
        super().__init__(
            model=model,
            description="Infers reasons for prompt failures.",
            name=name
        )
    def infer(self,error_input,prompt,num_feedbacks):
        reason_prompt = (
            "I'm trying to write a zero-shot recommender prompt.\n"\
            "My current prompt is \"$prompt$\"\n"\
            "But this prompt gets the following example wrong: $error_case$ "\
            "give $num_feedbacks$ reasons why the prompt could have gotten this example wrong.\n"\
            "Wrap each reason with <START> and <END>"
        ).replace("$prompt$", prompt).replace("$error_case$", error_input).replace("$num_feedbacks$", str(num_feedbacks))
        reasons = self.run(reason_prompt).content
        extract_reasons = extract_edit_prompt(reasons)
        # print("Len extract reasons: ",len(extract_reasons))
        if len(extract_reasons) == 0:
            reasons = reasons
        else:
            reasons = extract_reasons
        # print("- Reasons: ", reasons)
        return reasons

class RefinePromptAgent(Agent):
    def __init__(self,model,name='RefinePromptAgent'):
        super().__init__(
            model=model,
            description="Refines prompts based on inferred reasons.",
            name=name
        )
    def refine(self,prompt,error_input,reasons):
        refine_prompt = (
            "I'm trying to write a zero-shot recommender prompt.\n"\
            "My current prompt is \"$prompt$\"\n"\
            "But this prompt gets the following example wrong: $error_case$\n"\
            "Based on these example the problem with this prompt is that $reasons$.\n"\
            "Based on the above information, please wrote one improved prompt.\n"\
            "The prompt is wrapped with <START> and <END>.\n"\
            "The new prompt is:"
        ).replace("$prompt$", prompt).replace("$error_case$", error_input).replace("$reasons$", '\n'.join(reasons))
        refined_prompt = self.run(refine_prompt).content
        extract_refined_prompts = extract_edit_prompt(refined_prompt)
        if extract_refined_prompts:
            refined_prompt = extract_refined_prompts[0]  
            # print("- Refined prompt 1 : ", refined_prompt)
            return refined_prompt
        else:
            # print("- Refined prompt 2 : ", refined_prompt)
            return refined_prompt

class AugmentAgent(Agent):
    def __init__(self,model,name='AugmentAgent'): 
        super().__init__(
            model=model,
            description="Generates variations of refined prompts.",
            name=name
        )
    def augment(self,refined_prompt,additional_sample):
        augment_prompt = (
            "Generate a variation of the following instruction while keeping the semantic meaning.\n"\
            "Input: $refined_prompt$\n"\
            "Output:"
        ).replace("$refined_prompt$", refined_prompt)
        augmented_prompts = [self.run(augment_prompt).content for _ in range(additional_sample)]
        # print("- augmented_prompts: ", augmented_prompts)
        return augmented_prompts

class SelectionAgent(Agent):
    def __init__(self,model,name='SelectionAgent'):
        super().__init__(
            model=model,
            description='Selects the best prompts using UCB bandit algorithm.',
            name=name
        )
        self.eval_agent = EvalAgent(model)
        self.error_check_agent = ErrorCheckingAgent(model)

    def select(self, prompt_list, train_data, time_steps=16, explore_param=2, sample_num=32,beam_width=5):
        # print("- Prompt list", prompt_list)
        if not prompt_list:
            return []
        selections = [0] * len(prompt_list)
        rewards = [0] * len(prompt_list)

        for t in range(1, time_steps + 1):
            sample_data = random.sample(train_data, min(sample_num, len(train_data)))
            ucb_values = [
                (rewards[i] / selections[i] + explore_param * math.sqrt(math.log(t) / selections[i])) if selections[i] > 0 else float('inf')
                for i in range(len(prompt_list))
            ]
            # print("- UCB values: ", ucb_values)
            selected_idx = ucb_values.index(max(ucb_values))
            selected_prompt = prompt_list[selected_idx]
            # print("-select prompt: ", selected_prompt)
            reward = 0
            for data in sample_data:
                # json_req = (
                #     "Output Requirements: Provide JUST ONLY a clear and concise ranking of the entire candidate set(NOT EXPLAIN). "
                #     "Ensure that all items in the candidate set are assigned a rank, and that the ranking is based solely on the items within the candidate set. "
                #     "Importantly, The output format for the ranking must be in a JSON structure as follow: "
                #     '{ "1": "<item_1>", "2": "<item_2>", ... "20": "<item_20>" } '
                #     "and the name of candidate item in response MUSTN'T CHANGE!"
                # )
                # sel_prompt = f"{selected_prompt}\n{json_req}\n"
                prediction = self.eval_agent.evaluate(selected_prompt, data)
                corrected_response = self.error_check_agent.check_error(prediction, data)
                if corrected_response:
                    result_list = extract_item_json_list(corrected_response, data['target'])
                    # print("- result_list: ", result_list)
                    # print("- target: ", data['target'])
                else:
                    result_list = []
                if result_list:
                    target_idx =  result_list[0]
                    # print("- target_idx: ", target_idx)
                    reward += ndcg(target_idx)
                    # print(f"{t} Reward {reward}")
            
            selections[selected_idx] += len(sample_data)
            rewards[selected_idx] += reward
        
        prompt_reward_pairs = list(zip(rewards, prompt_list))
        prompt_reward_pairs.sort(reverse=True, key=lambda x: x[0])
        return [pair[1] for pair in prompt_reward_pairs[:beam_width]]