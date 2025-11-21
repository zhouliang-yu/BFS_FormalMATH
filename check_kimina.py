import json
import heapq
import os
import time
import re
import sys
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm, trange
from typing import List, Dict, Any, Optional, Tuple
import pdb
# Import kimina client
# Try to import from various possible locations
try:
    from client import Lean4Client
except ImportError:
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../test_compute'))
        from client import Lean4Client
    except ImportError:
        # If client.py is not found, you need to create it or adjust the import path
        # The client should implement Lean4Client with a verify() method
        print("ERROR: Could not import Lean4Client. Please ensure client.py is in your Python path.")
        print("The client.py should contain a Lean4Client class with a verify() method.")
        sys.exit(1)

import openai
import tiktoken
import transformers
import vllm

openai.api_key = ""  # Fill openai API key

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Initialize kimina client
client = Lean4Client(
    base_url=os.environ.get("LEAN4_API_URL"),
    api_key=os.environ.get("LEAN4_API_KEY"),
)


def lean_response_is_success(result_json: dict, accept_sorry=False, lean4_proof: str = None):
    """Check if Lean verification was successful."""
    if "results" in result_json:
        results = result_json.get("results", [])
        if not results:
            return 0.0
        
        for result in results:
            if result.get("error") is not None:
                return 0.0
            
            response = result.get("response", {})
            messages = response.get("messages", [])
            for msg in messages:
                if msg.get("severity") == "error":
                    return 0.0
                if not accept_sorry and msg.get("severity") == "warning":
                    if "declaration uses 'sorry'" in msg.get("data", ""):
                        return 0.0
    else:
        if result_json.get("error") is not None:
            return 0.0
        
        messages = result_json.get("response", {}).get("messages", [])
        for msg in messages:
            if msg.get("severity") == "error":
                return 0.0
            if not accept_sorry and msg.get("severity") == "warning":
                if "declaration uses 'sorry'" in msg.get("data", ""):
                    return 0.0
    
    if lean4_proof is not None:
        if 'theorem' not in lean4_proof:
            return 0.0
    return 1.0


def verify_proof_with_kimina(full_proof: str, imports: str = "") -> Tuple[bool, Dict]:
    """
    Verify a complete proof using kimina.
    Returns (success, result_dict)
    """
    try:
        # Construct full Lean code
        header = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"
        
        # If imports are provided and different from default, use them
        if imports and "import Mathlib" not in imports:
            # Use custom imports
            if "import" in imports:
                header = imports + "\n\n"
            else:
                header = imports + "\n" + header
        
        # Combine header and proof
        if 'import Mathlib' not in full_proof:
            full_lean_code = header + full_proof
        else:
            full_lean_code = full_proof
        
        # Verify using kimina
        proof = [{"proof": full_lean_code, "custom_id": random.randint(0, 2**16)}]
        result = client.verify(proof, timeout=60)
        
        success = lean_response_is_success(result, lean4_proof=full_lean_code, accept_sorry=False)
        return success > 0, result
        
    except Exception as e:
        print(f"Kimina verification error: {e}")
        return False, {"error": str(e)}


def _load_data(dataset_name, dataset_path):
    """Load dataset from JSONL file."""
    if 'minif2f' in dataset_name:
        data = []
        with open(dataset_path, encoding='utf-8') as f:
            for line in f.readlines():
                data_ = json.loads(line)
                data.append(data_)

        if 'valid' in dataset_name:
            data = [x for x in data if x['split'] == 'valid']
        else:
            data = [x for x in data if x['split'] == 'test']
            for example in data:
                example["srcContext"] = "import MiniF2F.Minif2fImport\n  open BigOperators Real Nat Topology\n"
    else:
        data = []
        with open(dataset_path, encoding='utf-8') as f:
            for line in f.readlines():
                data_ = json.loads(line)
                data.append(data_)

    return data


def truncate_middle(text, tokenizer, max_tokens=8000):
    """Truncate text from middle if too long."""
    tokens = tokenizer.encode(text) if hasattr(tokenizer, 'encode') else tokenizer(text)['input_ids']
    if len(tokens) <= max_tokens:
        return text

    keep_tokens = max_tokens // 2
    head_tokens = tokens[:keep_tokens]
    tail_tokens = tokens[-keep_tokens:]

    if hasattr(tokenizer, 'decode'):
        truncated_text = tokenizer.decode(head_tokens) + "..." + tokenizer.decode(tail_tokens)
    else:
        truncated_text = tokenizer.decode(head_tokens) + "..." + tokenizer.decode(tail_tokens)
    return truncated_text


def discard_after_marker(text, marker='/- BEGIN EXERCISES -/\n'):
    """Discard text after a marker."""
    marker_index = text.find(marker)
    if marker_index != -1:
        return text[:marker_index] + marker
    return text


def _load_jsonl(filepath):
    """Load JSONL file, return empty list if file doesn't exist."""
    if os.path.exists(filepath):
        data = []
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        return data
    else:
        return []


def get_premises(example, premise_path):
    """Get premises for an example."""
    if not os.path.exists(premise_path):
        raise FileNotFoundError(f"The folder '{premise_path}' does not exist.")
    
    project_premises = []
    for module in example["dependencyMetadata"]["importedModules"]:
        file = module + ".jsonl"
        premises = _load_jsonl(os.path.join(premise_path, file))
        project_premises += [premise["declaration"] for premise in premises]
    return project_premises


def generate_api(prompt, model, temperatures, num_samples, max_tokens=512):
    """Generate responses using OpenAI API."""
    texts, scores = [], []
    for temperature in temperatures:
        responses = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who is an expert in the Lean theorem prover."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            n=num_samples,
        )

        for choice in responses.choices:
            content = choice.message.content
            texts.append(content)
            scores.append(0)

    texts, scores = _unique_sorted(texts, scores)
    return texts, scores


def generate_vllm(prompt, model, tokenizer, temperature, num_samples, max_tokens=512, stop=["\n\n\n", "---", "[/TAC]"]):
    """Generate responses using vLLM."""
    texts, scores = [], []
    # Note: When temperature=0.0 (greedy sampling), vLLM requires n=1
    # If we need multiple samples with greedy sampling, use a small temperature > 0
    actual_temperature = temperature
    if temperature == 0.0 and num_samples > 1:
        # Use a very small temperature to get diverse but deterministic-like samples
        actual_temperature = 0.01
    
    params = vllm.SamplingParams(
        n=num_samples,
        temperature=actual_temperature,
        max_tokens=max_tokens,
        stop=stop,
    )
    outputs = model.generate([prompt], params, use_tqdm=False)
    if len(outputs) == 0:
        return [], []
    for output in outputs[0].outputs:
        text = output.text.replace(tokenizer.eos_token, '')
        # Handle case where cumulative_logprob might be None
        if hasattr(output, 'cumulative_logprob') and output.cumulative_logprob is not None:
            num_tokens = len(output.token_ids) if hasattr(output, 'token_ids') and output.token_ids else 1
            score = output.cumulative_logprob / max(num_tokens, 1)
        else:
            # Fallback: use a default score based on text length or set to 0
            score = 0.0
        texts.append(text)
        scores.append(score)
    
    texts, scores = _unique_sorted(texts, scores)
    return texts, scores


def _unique_sorted(texts, scores):
    """Sort texts by scores and remove duplicates."""
    texts_ = []
    scores_ = []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_


def process_responses_GPT4o(responses):
    """Extract Lean code from GPT-4o responses."""
    processed_responses = []
    # Handle both list of lists and single list formats
    if isinstance(responses, list) and len(responses) > 0:
        if isinstance(responses[0], list):
            response_list = responses[0]
        else:
            response_list = responses
    else:
        response_list = responses if isinstance(responses, list) else [responses]
    
    for response in response_list:
        pattern = re.compile(r'```lean(.*?)```', re.DOTALL | re.IGNORECASE)
        match = pattern.search(response)
        if match:
            processed_responses.append(match.group(1).strip())
        else:
            processed_responses.append(response)
    return processed_responses


def extract_lean4_code(solution_str: str):
    """Extract Lean 4 code from markdown code blocks."""
    lean4_codes = re.findall(r"```lean4\n(.*?)\n```", solution_str, re.DOTALL)
    if len(lean4_codes) == 0:
        # Try without lean4 tag
        lean4_codes = re.findall(r"```lean\n(.*?)\n```", solution_str, re.DOTALL)
    if len(lean4_codes) == 0:
        return solution_str.strip()
    else:
        return max(lean4_codes, key=len).strip()


def _prompt_fewshot(tokenizer, theorem_statement, ctx=None, state=None, tactics=None, premises=None, task="tactic_prediction"):
    """Generate prompt based on task type."""
    prompt_dir = "prompt"
    if premises is not None:
        prompt_file = os.path.join(prompt_dir, f"{task}_premise.txt")
    else:
        prompt_file = os.path.join(prompt_dir, f"{task}.txt")
    header = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"
    theorem_statement = header + theorem_statement
    if not os.path.exists(prompt_file):
        # Fallback to default prompt
        if task == "full_proof":
            return f"Prove the following theorem in Lean 4:\n\n{theorem_statement}"
        else:
            return f"Complete the proof:\n\n{theorem_statement}\n\nCurrent state: {state}"
    
    with open(prompt_file, "r") as infile:
        prompt = infile.read()
    
    if task == "tactic_prediction" or task == "tactic_prediction_fewshot":
        prompt = prompt.format(state)
        return prompt
    elif task == "tactic_prediction_context":
        ctx = discard_after_marker(ctx) if ctx else ""
        ctx = truncate_middle(ctx, tokenizer) if ctx else ""
        ctx = ctx + theorem_statement + "\n  " + "\n  ".join(tactics) if tactics else ctx + theorem_statement
        if premises is not None:
            premises_str = "\n".join(premises)
            prompt = prompt.format(ctx, premises_str, state)
        else:
            prompt = prompt.format(ctx, state)
        return prompt
    elif task == "full_proof_context":
        if premises is not None:
            premises_str = "\n".join(premises)
            prompt = prompt.format(ctx, premises_str, theorem_statement)
        else:
            prompt = prompt.format(ctx, theorem_statement)
        prompt = truncate_middle(prompt, tokenizer)
        return prompt
    elif task == "full_proof":
        prompt = prompt.format(theorem_statement)
        return prompt
    else:
        print(f"Error: Task '{task}' is unsupported")
        sys.exit(1)


def _load_model(model_name, tp_degree):
    """Load vLLM model."""
    model = vllm.LLM(
        model=model_name,
        tensor_parallel_size=tp_degree,
        dtype='bfloat16',
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def best_first_search_kimina(example, task, premise_path, model, tokenizer, re_model, re_tokenizer, 
                              prompt_fn, temperature, num_samples, max_tokens=512, max_iters=250):
    """
    Best-first search using kimina for verification.
    Instead of step-by-step tactic evaluation, we generate complete proof candidates
    and verify them with kimina.
    """
    # Handle both "statement" and "theoremStatement" keys
    if "statement" in example:
        statement = example["statement"]
    elif "theoremStatement" in example:
        statement = example["theoremStatement"]
    else:
        return {'success': False, 'msg': "No statement found in example"}
    
    theorem_statement = statement 
    imports = example.get("srcContext", "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n")
    
    print()
    print(f"Problem: {theorem_statement}")
    
    # Queue: (score, proof_candidate, iteration)
    # Lower score = higher priority (we use negative logprob or iteration count)
    queue = [(0.0, "", 0)]  # Start with empty proof
    
    visited = set()  # Track visited proof candidates (normalized)

    for iteration in trange(max_iters):
        if len(queue) == 0:
            break
        
        # Dequeue the tuple with minimum score
        score, current_proof, iter_count = heapq.heappop(queue)
        
        # Skip if we've seen this proof before
        proof_normalized = current_proof.strip()
        if proof_normalized in visited:
            continue
        visited.add(proof_normalized)
        
        # Generate prompt for next step
        retrieved_premises = None
        if premise_path and re_model and re_tokenizer:
            try:
                premises = get_premises(example, premise_path)
                # For full proof generation, we might not need goal-based retrieval
                # But we can still use premises for context
                retrieved_premises = premises[:20] if len(premises) > 20 else premises
            except:
                retrieved_premises = None
        # Generate prompt
        if task == "full_proof" or task == "full_proof_context":
            # Generate complete proof
            # _prompt_fewshot(tokenizer, theorem_statement, ctx, state, tactics, premises, task)
            prompt = prompt_fn(tokenizer, theorem_statement, imports, None, None, retrieved_premises, task)
        else:
            # For tactic prediction, we need to build up the proof
            # Use current proof as context
            current_state = current_proof if current_proof else None
            tactics = current_proof.split('\n') if current_proof else []
            prompt = prompt_fn(tokenizer, theorem_statement, imports, current_state, tactics, retrieved_premises, task)
        
        # Generate proof candidates
        if isinstance(model, str) and "gpt" in model:
            # Use OpenAI API
            proof_candidates, candidate_scores = generate_api(prompt, model, [temperature], num_samples, max_tokens)
            proof_candidates = process_responses_GPT4o(proof_candidates)
        else:
            # Use vLLM
            proof_candidates, candidate_scores = generate_vllm(prompt, model, tokenizer, temperature, num_samples, max_tokens)
        

        # Ensure we have the same number of candidates and scores
        if len(proof_candidates) == 0:
            continue
        if len(candidate_scores) < len(proof_candidates):
            # Pad scores with zeros if needed
            candidate_scores.extend([0.0] * (len(proof_candidates) - len(candidate_scores)))
        
        # Process each candidate
        for proof_candidate, candidate_score in zip(proof_candidates, candidate_scores):
            # Extract Lean code if needed
            lean_proof = extract_lean4_code(proof_candidate)
            
            # Build complete proof
            if current_proof:
                # Append to existing proof
                if task == "full_proof" or task == "full_proof_context":
                    # For full proof, replace or append
                    full_proof = lean_proof
                else:
                    # For tactic prediction, append tactic
                    full_proof = current_proof + "\n  " + lean_proof
            else:
                full_proof = lean_proof
            
            # Construct the complete theorem statement with proof
            complete_theorem = theorem_statement
            if full_proof.strip():
                if complete_theorem.endswith(":="):
                    complete_theorem += "\n  " + full_proof
                elif complete_theorem.endswith("by"):
                    complete_theorem += "\n  " + full_proof
                else:
                    complete_theorem += "\n  " + full_proof
            # remove all the ``` in complete_theorem
            complete_theorem = re.sub(r"```lean4\n", "", complete_theorem)
            complete_theorem = re.sub(r"```lean\n", "", complete_theorem)
            complete_theorem = re.sub(r"```\n", "", complete_theorem)
            complete_theorem = re.sub(r"```", "", complete_theorem)
            
            # Verify with kimina
            success, result = verify_proof_with_kimina(complete_theorem, imports)
            if success:
                print(f"\n✓ Proof found at iteration {iteration}!")
                return {
                    'success': True,
                    'proof': full_proof,
                    'iterations': iteration + 1
                }
            
            # If not successful, add to queue for further exploration
            # Use negative score (lower = higher priority) based on candidate score
            new_score = score - candidate_score  # Better candidates have higher scores
            new_iter = iter_count + 1
            
            if new_iter < max_iters:
                heapq.heappush(queue, (new_score, full_proof, new_iter))
    
    return {'success': False, 'msg': "Search ended", 'iterations': max_iters}


def _print_output(example, out, task):
    """Print output results."""
    print(out)
    if 'proof' in out and out.get('success'):
        # Get statement from example - handle both keys
        statement = example.get("statement") or example.get("theoremStatement", "")
        if "tactic_prediction" in task:
            print(statement + ' := by\n  ' + '\n  '.join(out['proof'].split('\n')))
        if "full_proof" in task:
            print(out['proof'])


def make_output_dir(output_dir):
    """Create output directory with timestamp."""
    dt = datetime.now().strftime("%d-%m-%Y-%H-%M")
    output_dir = os.path.join(output_dir, dt)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def get_full_name(statement):
    """Extract theorem/lemma name from statement."""
    word_list = statement.split()
    for i in range(len(word_list)):
        if "theorem" in word_list[i] or "lemma" in word_list[i]:
            if i + 1 < len(word_list):
                return word_list[i + 1]
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BFS with kimina verification")
    parser.add_argument('--model-name', required=True, help='Model name or path')
    parser.add_argument(
        '--task',
        default='full_proof',
        choices=['tactic_prediction', 'tactic_prediction_context', 'full_proof', 'full_proof_context', 'tactic_prediction_fewshot']
    )
    parser.add_argument(
        '--dataset-name',
        default='mathlib'
    )
    parser.add_argument('--dataset-path', default='data/mathlib.jsonl')
    parser.add_argument('--premise-path', default=None)
    parser.add_argument('--output-dir', default='output/kimina_bfs')
    parser.add_argument('--tp-degree', type=int, default=1)
    parser.add_argument('--max-iters', type=int, default=100)
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--temperatures', type=float, default=0.7)

    
    args = parser.parse_args()

    # Load model
    use_API = False
    if "gpt" in args.model_name:
        use_API = True
        model = args.model_name
        tokenizer = tiktoken.encoding_for_model(args.model_name)
    else:
        model, tokenizer = _load_model(args.model_name, args.tp_degree)

    # Load retriever if premise path is provided
    device = "cuda" 
    if args.premise_path:
        re_tokenizer = transformers.AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean4-retriever-byt5-small")
        re_model = transformers.AutoModelForTextEncoding.from_pretrained("kaiyuy/leandojo-lean4-retriever-byt5-small").to(device)
    else:
        re_model, re_tokenizer = None, None

    output_dir = make_output_dir(args.output_dir)
    examples = _load_data(args.dataset_name, args.dataset_path)
    
    prompt_fn = _prompt_fewshot

    successes = 0
    count = 0
    for example in examples:
        # Get theorem statement - handle both "statement" and "theoremStatement" keys
        if "statement" in example:
            statement = example["statement"]
        elif "theoremStatement" in example:
            statement = example["theoremStatement"]
        else:
            print(f"Warning: No statement found in example {count}")
            continue
        
        example["full_name"] = get_full_name(statement)
        count += 1
        
        out = best_first_search_kimina(
            example, args.task, args.premise_path, 
            model, tokenizer, re_model, re_tokenizer, 
            prompt_fn, args.temperatures, args.num_samples, 
            max_tokens=args.max_tokens, max_iters=args.max_iters
        )
        
        if out['success']: 
            successes += 1
            example["proof"] = out
        else:
            example["proof"] = out
        
        _print_output(example, out, args.task)
        print(f"Successes: {successes}/{count}")
    
    filename = 'result.jsonl'
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as file:
        for entry in examples:
            json.dump(entry, file)
            file.write('\n')

