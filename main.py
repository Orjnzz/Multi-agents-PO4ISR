import json
import os
import random
from dotenv import load_dotenv
from agno.models.openai.like import OpenAILike
from agents import EvalAgent, ErrorCheckingAgent, DetectErrrorAgent, InferReasonAgent, RefinePromptAgent, AugmentAgent, SelectionAgent
from opt.utils import extract_item_json_list, ndcg, extract_wrapped_json

load_dotenv('your_path.env')
DEEPINFRA_API_KEY = os.getenv('DEEPINFRA_API_KEY')

llm_model = OpenAILike(
    id="meta-llama/Llama-3.3-70B-Instruct",
    api_key=DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
)

data_path = "./dataset/"
train_file = f"{data_path}train_150s_processed.json"
val_file = f"{data_path}valid_98s_processed.json"

with open(train_file, 'r') as f:
    train_data = json.load(f)
with open(val_file, 'r') as f:
    val_data = json.load(f)

config = {
    'search_depth': 2,
    'beam_width': 5,
    'batch_size': 16,
    'num_feedbacks': 2,
    'addition_sample': 2,
    'num_candidates': 8,
    'time_steps': 16,
    'explore_param': 2,
    'sample_num': 32,
}

initial_prompt = (
       """
       Based on the user's current session interactions, you need to answer the following subtasks step by step:
       1. Discover combinations of items within the session, where the number of combinations can be one or more.
       2. Based on the items within each combination, infer the user's interactive intent within each combination.
       3. Select the intent from the inferred ones that best represents the user's current preferences.
       4. Based on the selected intent, please rerank the 20 items in the candidate set according to the possibility of potential user interactions and show me your ranking results with item index.
       Note that the order of all items in the candidate set must be given, and the items for ranking must be within the candidate set.
       """
)

eval_agent = EvalAgent(model=llm_model)
error_check_agent = ErrorCheckingAgent(model=llm_model)
detect_error_agent = DetectErrrorAgent()
infer_reason_agent = InferReasonAgent(model=llm_model)
refine_agent = RefinePromptAgent(model=llm_model)
augment_agent = AugmentAgent(model=llm_model)
selection_agent = SelectionAgent(model=llm_model)

beam = [initial_prompt]
for depth in range(config['search_depth']):
    print(f'Depth {depth + 1} / {config["search_depth"]}')
    candidate_prompts = []
    for prompt in beam:
        batch = random.sample(train_data, min(config['batch_size'], len(train_data)))
        error_cases = []
        # error = 0
        for data in batch:
            response = eval_agent.evaluate(prompt,data)
            # print("-Initail response: ", response)
            corrected_response = error_check_agent.check_error(response,data)
            if corrected_response:
                is_error = detect_error_agent.detect_error(corrected_response,data['target'])
                if is_error:
                    # error += 1
                    # print("- Error: ", error)
                    error_cases.append({'input': data['input'], 'output': corrected_response})
        for error in error_cases[:config['num_feedbacks']]:
            reasons = infer_reason_agent.infer(error['input'], prompt, config['num_feedbacks'])
            refined_prompt = refine_agent.refine(prompt, error['input'], reasons)
            augmented_prompts = augment_agent.augment(refined_prompt, config['addition_sample'])
            candidate_prompts.append(refined_prompt)
            candidate_prompts.extend(augmented_prompts)
            # print("Candidate prompts: ", candidate_prompts)
    beam = selection_agent.select(candidate_prompts, train_data, config['time_steps'], config['explore_param'], config['sample_num'], config['beam_width'])
    # print("BEAM: ", beam)

def compute_reward(prompt, data):
    reward = 0
    for d in data:
        prediction = eval_agent.evaluate(prompt, d)
        check_prediction = error_check_agent.check_error(prediction,d)
        # print("- compute_reward process: ",prediction)
        # print("- target: ", d['target'])
        if check_prediction:
            result_list = extract_item_json_list(check_prediction, d['target'])
        else:
            result_list = [] 
        # print('- result_list: ', result_list)
        if result_list:
            target_idx = result_list[0]
            # print('- target_idx: ', target_idx)
            reward += ndcg(target_idx)
            # print(f"- Reward {reward}")
    return reward / len(data)

best_prompt = max(beam, key=lambda p: compute_reward(p, val_data))
print("Best prompt:", best_prompt)

