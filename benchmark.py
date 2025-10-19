from datasets import load_dataset, DatasetDict
import pydantic
import typing
import json
import tqdm
import time
import logging
import argparse
import os
import yaml
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from judge import LightCPVerifierJudge, SupportedLanguage, ProblemNotFoundError
from util import extract_longest_cpp_code

def fix_cpp_headers(code: str) -> str:
    """
    Fix missing C++ headers by adding common competitive programming headers.
    """
    if not code:
        return code
    
    lines = code.split('\n')
    
    # Check if bits/stdc++.h is already included
    has_bits_stdcpp = any('#include <bits/stdc++.h>' in line for line in lines)
    
    # If bits/stdc++.h is present, return as is
    if has_bits_stdcpp:
        return code
    
    # Find the position of the first #include
    first_include_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('#include'):
            first_include_idx = i
            break
    
    # If no includes found, insert at the beginning
    if first_include_idx == -1:
        return '#include <bits/stdc++.h>\nusing namespace std;\n\n' + code
    
    # Replace the first include with bits/stdc++.h
    lines[first_include_idx] = '#include <bits/stdc++.h>'
    
    # Remove duplicate includes that are now redundant
    redundant_includes = [
        '#include <iostream>',
        '#include <vector>',
        '#include <string>',
        '#include <map>',
        '#include <set>',
        '#include <algorithm>',
        '#include <cmath>',
        '#include <tuple>',
    ]
    
    filtered_lines = []
    for line in lines:
        is_redundant = False
        for redundant in redundant_includes:
            if redundant in line:
                is_redundant = True
                break
        if not is_redundant:
            filtered_lines.append(line)
    
    # Ensure 'using namespace std;' is present
    has_using_namespace = any('using namespace std' in line for line in filtered_lines)
    if not has_using_namespace:
        # Find where to insert 'using namespace std;' (after includes)
        insert_idx = 0
        for i, line in enumerate(filtered_lines):
            if line.strip().startswith('#include'):
                insert_idx = i + 1
            elif not line.strip().startswith('#'):
                break
        
        filtered_lines.insert(insert_idx, 'using namespace std;')
    
    return '\n'.join(filtered_lines)

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv(override=True)

def get_dynamic_api_config(model_name):
    config_path = os.path.join(os.path.dirname(__file__), "models.yaml")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"models.yaml not found at {config_path}")
    
    models = config.get('models', [])
    
    for model_config in models:
        if model_config['name'] == model_name:
            base_url = model_config.get('base_url')
            api_key_template = model_config.get('api_key', '')
            
            if api_key_template.startswith('${') and api_key_template.endswith('}'):
                env_var = api_key_template[2:-1]
                api_key = os.getenv(env_var)
                if not api_key:
                    print(f"⚠️ Missing API key for '{model_name}': {env_var}")
            else:
                api_key = api_key_template
            
            return api_key, base_url, model_config
    
    raise ValueError(f"Model '{model_name}' not found in models.yaml")


class OpenAILLM:
    def __init__(self, model_name: str):
        api_key, base_url, model_config = get_dynamic_api_config(model_name)
        self.model = model_name
        self.model_config = model_config
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.prompt = """
        You are a competitive programmer. You will be given a problem statement, please implement solution in C++. The execution time and memory limit are also stated in the statement so be aware of the complexity of the program. Please wrap the code in ```cpp and ``` so that it is properly formatted.

        IMPORTANT: Always include all necessary headers. For competitive programming, it's recommended to use:
        #include <bits/stdc++.h>
        using namespace std;

        This includes all standard library headers and prevents compilation errors due to missing includes.
        """

    def generate_solution(self, problem_statement: str):
        user_prompt = self.prompt + problem_statement
        
        completion_params = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        
        if "temperature" in self.model_config:
            completion_params["temperature"] = self.model_config["temperature"]
        
        if "extra_body" in self.model_config:
            completion_params.update(self.model_config["extra_body"])
        
        completion = self.client.chat.completions.create(**completion_params)
        response_content = completion.choices[0].message.content
        prompt_tokens = completion.usage.prompt_tokens
        response_tokens = completion.usage.completion_tokens
        total_tokens = prompt_tokens + response_tokens
        return response_content, str(completion), total_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="LiveCodeBench Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Model name to use for LLM API")
    parser.add_argument("--worker", type=int, default=16, help="Number of workers for LightCPVerifier")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of problems to process (for testing)")
    parser.add_argument("--parallel", type=int, default=10, help="Number of parallel API calls (default: 10)")
    parser.add_argument("-r","--resume", type=str, default=None, help="Path to JSON file to resume from")
    parser.add_argument("--split",type=str,default="quater_2025_1_3",help="Dataset split")
    parser.add_argument("--problem_ids", type=str, default=None, help="Comma-separated list of problem IDs to test (e.g., '2059E2,2060G,2062E1')")
    return parser.parse_args()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkResult(pydantic.BaseModel):
    problem_id: str
    problem_title: str
    difficulty: str
    platform: str
    text_response: str | None = None
    code: str | None
    judge_result: str
    submission_id: int | None = None
    response_meta: typing.Any

class ProblemTestState(pydantic.BaseModel):
    problem_id: str
    problem_title: str
    difficulty: str
    platform: str
    problem_statement: str
    text_response: str | None = None
    code: str | None = None
    submission_id: int | None = None
    judge_result: str = "Judging"
    response_meta: typing.Any = None
    token_count: int | None = None


def generate_solution_for_problem(llm_instance: OpenAILLM, problem: ProblemTestState):
    response, meta, token_count = llm_instance.generate_solution(problem.problem_statement)
    code = extract_longest_cpp_code(response)
    if code:
        code = fix_cpp_headers(code)
    return problem.problem_id, response, code, meta, token_count


def check_result_until_done(judge: LightCPVerifierJudge, problem: ProblemTestState, max_wait: int = 300):
    start_time = time.time()
    while True:
        if time.time() - start_time > max_wait:
            return "Judge Timeout"
        result = judge.get_result(problem.submission_id)
        if result != "Judging":
            return result
        time.sleep(1)

def evaluate_problem(judge: LightCPVerifierJudge, problem: ProblemTestState, problem_set: dict, result_file: str):
    if not problem.code:
        problem.judge_result = "No Code"
        save_results(problem_set, result_file)
        return
    
    max_retries = 3
    for retry in range(max_retries):
        try:
            submission_id = judge.submit(problem.problem_id, SupportedLanguage.CPP, problem.code)
            if submission_id:
                problem.submission_id = submission_id
                logger.info(f"Successfully submitted {problem.problem_id} with ID: {submission_id}")
                save_results(problem_set, result_file)
                break
        except ProblemNotFoundError:
            logger.warning(f"Problem {problem.problem_id} not found in judge dataset.")
            problem.judge_result = "Problem Not Found"
            save_results(problem_set, result_file)
            return
        except Exception as e:
            logger.warning(f"Submit attempt {retry + 1}/{max_retries} failed for {problem.problem_id}: {e}")
            if retry < max_retries - 1:
                time.sleep(2)
    
    if not problem.submission_id:
        logger.error(f"All submission attempts failed for {problem.problem_id}")
        problem.judge_result = "Submit Failed"
        save_results(problem_set, result_file)
        return
    
    try:
        result = check_result_until_done(judge, problem)
        problem.judge_result = result
        logger.info(f"Result for {problem.problem_id}: {result}")
        save_results(problem_set, result_file)
    except Exception as e:
        logger.error(f"Error fetching result for {problem.problem_id}: {e}")
        problem.judge_result = "Judge Failed"
        save_results(problem_set, result_file)


ABNORMAL_JUDGE_RESULTS = {
    "Judging", "Judge Failed", "Submit Failed", "No Code", "Problem Not Found",
    "Judge Failed: Network error", "Judge Failed: Compilation Error",
    "Judge Failed: Error: Unknown system error",
    "Judge Failed: Error: Unknown system error -35",
    "Judge Timeout"
}


def retry_generate_solution(llm_instance, problem: ProblemTestState):
    try:
        pid, response, code, meta, token_count = generate_solution_for_problem(llm_instance, problem)
        problem.text_response = response
        problem.code = code
        problem.response_meta = meta
        problem.token_count = token_count
        logger.info(f"Retry generation successful for {problem.problem_id}")
        return True
    except Exception as e:
        logger.error(f"Retry generation failed for {problem.problem_id}: {e}")
        return False


def check_and_retry(problem_set: dict[str, ProblemTestState], llm_instance, judge: LightCPVerifierJudge, result_file: str, max_retries: int = 2):
    for retry_round in range(max_retries):
        needs_regeneration = []
        needs_reevaluation = []
        
        for problem in problem_set.values():
            if not problem.code:
                needs_regeneration.append(problem)
            elif problem.judge_result in ABNORMAL_JUDGE_RESULTS:
                needs_reevaluation.append(problem)
        
        if not needs_regeneration and not needs_reevaluation:
            logger.info(f"Round {retry_round + 1}: All problems are healthy, no retry needed.")
            break
        
        logger.info(f"Round {retry_round + 1}: Found {len(needs_regeneration)} problems needing regeneration, "
                   f"{len(needs_reevaluation)} problems needing reevaluation")
        
        if needs_regeneration:
            logger.info(f"Retrying generation for: {[p.problem_id for p in needs_regeneration]}")
            with ThreadPoolExecutor(max_workers=min(len(needs_regeneration), 5)) as executor:
                futures = {
                    executor.submit(retry_generate_solution, llm_instance, problem): problem
                    for problem in needs_regeneration
                }
                for future in as_completed(futures):
                    problem = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Retry gen error for {problem.problem_id}: {e}")
            
            regenerated = [p for p in needs_regeneration if p.code]
            logger.info(f"Successfully regenerated {len(regenerated)}/{len(needs_regeneration)} problems")
        
        if needs_reevaluation:
            problem_ids = [p.problem_id for p in needs_reevaluation]
            logger.info(f"Retrying evaluation for: {problem_ids}")
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(evaluate_problem, judge, problem, problem_set, result_file): problem
                    for problem in needs_reevaluation
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Retry eval error: {e}")
            
            reevaluated = [p for p in needs_reevaluation if p.judge_result not in ABNORMAL_JUDGE_RESULTS]
            logger.info(f"Successfully reevaluated {len(reevaluated)}/{len(needs_reevaluation)} problems")
        
        save_results(problem_set, result_file)
    
    final_abnormal = [p for p in problem_set.values() if not p.code or p.judge_result in ABNORMAL_JUDGE_RESULTS]
    if final_abnormal:
        logger.warning(f"Final abnormal problems: {[p.problem_id for p in final_abnormal]}")
    else:
        logger.info("All problems are healthy after retry checks!")

def get_problem_set(dataset) -> dict[str, ProblemTestState]:
    problem_set = {}
    splits = dataset.values() if hasattr(dataset, 'values') else [dataset]
    for split in splits:
        for row in split:
            if row["problem_id"] not in problem_set:
                problem_set[row["problem_id"]] = ProblemTestState(**row)
    return problem_set

def print_stats(dataset, problem_set: dict[str, ProblemTestState]):
    print("=" * 80)
    print("BENCHMARK STATISTICS")
    print("=" * 80)

    split_difficulty_stats = {}

    splits = dataset.values() if hasattr(dataset, 'values') else {"default": dataset}
    for split_name, split in splits.items():
        split_difficulty_stats[split_name] = {}
        
        for row in split:
            problem_id = row["problem_id"]
            difficulty = row.get("difficulty", "unknown")

            if problem_id in problem_set:
                judge_result = problem_set[problem_id].judge_result
            else:
                judge_result = "Not Tested"

            if difficulty not in split_difficulty_stats[split_name]:
                split_difficulty_stats[split_name][difficulty] = {
                    "total": 0, 
                    "accepted": 0, 
                    "judge_results": {}
                }
            
            split_difficulty_stats[split_name][difficulty]["total"] += 1
            if judge_result == "Accepted":
                split_difficulty_stats[split_name][difficulty]["accepted"] += 1

            if judge_result not in split_difficulty_stats[split_name][difficulty]["judge_results"]:
                split_difficulty_stats[split_name][difficulty]["judge_results"][judge_result] = []
            split_difficulty_stats[split_name][difficulty]["judge_results"][judge_result].append(problem_id)

    for split_name in split_difficulty_stats:
        print(f"\n[SPLIT: {split_name.upper()}]")
        print("-" * 60)
        
        total_problems_in_split = 0
        total_accepted_in_split = 0
        
        for difficulty, stats in sorted(split_difficulty_stats[split_name].items()):
            total = stats["total"]
            accepted = stats["accepted"]
            accuracy = (accepted / total * 100) if total > 0 else 0.0
            
            print(f"\n{difficulty.upper()} Difficulty: {accepted}/{total} ({accuracy:.1f}%)")

            for judge_result, problem_ids in sorted(stats["judge_results"].items()):
                count = len(problem_ids)
                percentage = (count / total * 100) if total > 0 else 0.0
                print(f"  {judge_result:20s}: {count:3d} ({percentage:5.1f}%) - {', '.join(sorted(problem_ids))}")
            
            total_problems_in_split += total
            total_accepted_in_split += accepted

        overall_accuracy = (total_accepted_in_split / total_problems_in_split * 100) if total_problems_in_split > 0 else 0.0
        print(f"\nOVERALL for {split_name}: {total_accepted_in_split}/{total_problems_in_split} ({overall_accuracy:.1f}%)")
    
    print("\n" + "=" * 80)


def calculate_statistics(problem_set: dict[str, ProblemTestState]) -> dict:
    difficulty_stats = {}
    
    for problem in problem_set.values():
        difficulty = problem.difficulty.lower()
        judge_result = problem.judge_result
        
        if difficulty not in difficulty_stats:
            difficulty_stats[difficulty] = {
                "total": 0,
                "accepted": 0,
                "accuracy": 0.0
            }
        
        difficulty_stats[difficulty]["total"] += 1
        if judge_result == "Accepted":
            difficulty_stats[difficulty]["accepted"] += 1
    
    for difficulty in difficulty_stats:
        total = difficulty_stats[difficulty]["total"]
        accepted = difficulty_stats[difficulty]["accepted"]
        difficulty_stats[difficulty]["accuracy"] = round((accepted / total * 100), 2) if total > 0 else 0.0
    
    total_problems = sum(stats["total"] for stats in difficulty_stats.values())
    total_accepted = sum(stats["accepted"] for stats in difficulty_stats.values())
    average_accuracy = round((total_accepted / total_problems * 100), 2) if total_problems > 0 else 0.0
    
    statistics = {
        "easy": difficulty_stats.get("easy", {"total": 0, "accepted": 0, "accuracy": 0.0}),
        "medium": difficulty_stats.get("medium", {"total": 0, "accepted": 0, "accuracy": 0.0}),
        "hard": difficulty_stats.get("hard", {"total": 0, "accepted": 0, "accuracy": 0.0}),
        "average": {
            "total": total_problems,
            "accepted": total_accepted,
            "accuracy": average_accuracy
        }
    }
    
    return statistics


def save_results(problem_set: dict[str, ProblemTestState], filename: str = "result.json"):
    results = []
    for problem in problem_set.values():
        result_data = {
            "problem_id": problem.problem_id,
            "problem_title": problem.problem_title,
            "difficulty": problem.difficulty,
            "platform": problem.platform,
            "text_response": problem.text_response,
            "code": problem.code,
            "judge_result": problem.judge_result,
            "submission_id": problem.submission_id,
            "response_meta": problem.response_meta,
            "token_count": problem.token_count,
        }
        results.append(result_data)
    
    statistics = calculate_statistics(problem_set)
    
    output_data = {
        "statistics": statistics,
        "results": results
    }
    
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=4)


def load_saved_results(filename: str) -> dict[str, dict] | None:
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "results" in data:
            saved_data = data["results"]
        elif isinstance(data, list):
            saved_data = data
        else:
            logger.warning(f"Invalid JSON format in {filename}")
            return None
        
        problem_set = {}
        for item in saved_data:
            pid = item["problem_id"]
            problem_set[pid] = {
                "text_response": item.get("text_response"),
                "code": item.get("code"),
                "submission_id": item.get("submission_id"),
                "judge_result": item.get("judge_result"),
                "response_meta": item.get("response_meta"),
                "token_count": item.get("token_count"),
            }
        return problem_set
    except Exception as e:
        logger.warning(f"Failed to load saved results: {e}")
        return None


def merge_results(problem_set: dict[str, ProblemTestState], saved_set: dict[str, dict]):
    resumed_count = 0
    for problem_id, saved_data in saved_set.items():
        if problem_id in problem_set:
            if saved_data["judge_result"] != "Judging":
                problem_set[problem_id].text_response = saved_data["text_response"]
                problem_set[problem_id].code = saved_data["code"]
                problem_set[problem_id].submission_id = saved_data["submission_id"]
                problem_set[problem_id].judge_result = saved_data["judge_result"]
                problem_set[problem_id].response_meta = saved_data["response_meta"]
                problem_set[problem_id].token_count = saved_data.get("token_count")
                resumed_count += 1
    return resumed_count


if __name__ == "__main__":
    args = parse_args()
    llm_instance = OpenAILLM(args.model)
    worker = args.worker
    parallel = args.parallel
    
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    if args.resume:
        result_file = args.resume
        print(f"Resuming from: {result_file}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"result/LiveCodeBench_{args.split}/{args.model.replace('/', '_')}/{timestamp}.json"
        print(f"Starting new benchmark, results will be saved to: {result_file}")
    
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    dataset = load_dataset("QAQAQAQAQ/LiveCodeBench-Pro", split=args.split)
    problem_set = get_problem_set(dataset)
    
    if args.problem_ids:
        problem_ids = [pid.strip() for pid in args.problem_ids.split(",")]
        problem_set = {pid: problem_set[pid] for pid in problem_ids if pid in problem_set}
        print(f"Testing specific problems: {list(problem_set.keys())}")
    elif args.limit:
        problem_ids = list(problem_set.keys())[:args.limit]
        problem_set = {pid: problem_set[pid] for pid in problem_ids}
        print(f"Limited to first {args.limit} problems for testing")
    
    saved_set = load_saved_results(result_file)
    if saved_set:
        resumed = merge_results(problem_set, saved_set)
        print(f"Resumed {resumed} completed problems from {result_file}")
    
    with LightCPVerifierJudge(worker=worker) as judge:
        problems_to_process = [
            p for p in problem_set.values() 
            if p.judge_result == "Judging"
        ]
        if problems_to_process:
            print(f"Processing {len(problems_to_process)} remaining problems (parallel={parallel})")
        else:
            print("All problems already completed!")
        
        generation_worker = parallel
        eval_worker = min(parallel, 10)
        
        with ThreadPoolExecutor(max_workers=generation_worker) as gen_executor:
            futures = {
                gen_executor.submit(generate_solution_for_problem, llm_instance, problem): problem 
                for problem in problems_to_process
            }
            
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Generating solutions"):
                problem = futures[future]
                try:
                    pid, response, code, meta, token_count = future.result()
                    problem.text_response = response
                    problem.code = code
                    problem.response_meta = meta
                    problem.token_count = token_count
                    save_results(problem_set, result_file)
                except Exception as e:
                    logger.error(f"Error generating solution for {problem.problem_id}: {e}")
        
        problems_with_code = [p for p in problems_to_process if p.code]
        print(f"Generated {len(problems_with_code)} solutions, starting evaluation (parallel={eval_worker})")
        
        with ThreadPoolExecutor(max_workers=eval_worker) as eval_executor:
            futures = {
                eval_executor.submit(evaluate_problem, judge, problem, problem_set, result_file): problem 
                for problem in problems_with_code
            }
            
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Evaluating solutions"):
                problem = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error evaluating problem {problem.problem_id}: {e}")
        
        check_and_retry(problem_set, llm_instance, judge, result_file, max_retries=2)
    
    print_stats(dataset, problem_set)
