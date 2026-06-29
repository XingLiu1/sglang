import asyncio
import aiohttp
import json
import time
import random
import argparse
from datetime import datetime
import tqdm
import math
from tqdm.asyncio import tqdm_asyncio
import requests

TIMESTAMP_FORMAT = "%Y-%m-%d-%H-%M-%S"

time_stamp = datetime.now().strftime(TIMESTAMP_FORMAT)

tokenizer = None

global_prefix_ids: list[int] | None = None

def create_prompts(input_len: int, num_req: int=1, prefix_len: int=0, prefix_ids: list[int] | None = None) -> list[list[int]]:
    if prefix_len > 0:
        assert input_len >= prefix_len
        if prefix_ids is None:
            prefix_ids = [random.randint(42, 10000) for _ in range(prefix_len)]
        else:
            assert len(prefix_ids) == prefix_len
    else:
        prefix_ids = []

    return [prefix_ids + [random.randint(42, 10000) for _ in range(input_len-prefix_len)] for _ in range(num_req)]


def detokenize_to_str(token_id_list: list[int]) -> str:
    from transformers import AutoTokenizer
    global tokenizer
    if tokenizer == None:
        assert args.tokenizer_path != ''
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    return tokenizer.decode(token_id_list)


def completion_data(model: str, base_url: str, prompt: list[int], output_len: int=10):
    if args.str_input:
        prompt = detokenize_to_str(prompt)
    data_template = {
        "model": model,
        "stream": True,
        "max_tokens": output_len,
        "temperature": 0.6,
        "prompt": prompt,
        "top_p": 0.95,
        "ignore_eos": True,
        "stream_options" : {
            "include_usage": True},
    }
    url = f"{base_url}/v1/completions"
    return data_template, url
    
    
async def single_task(prompt_ids: list[int], output_len: int = 10, if_print: bool = True):

    data_template, url = completion_data(args.model, args.base_url, prompt=prompt_ids, output_len=output_len)

    t_start = time.time()    
    timeout = aiohttp.ClientTimeout(total=6000)
    connector = aiohttp.TCPConnector(limit=0)  # 连接数
    first_token = True
    last_time = t_start
    ttft = 0
    tpot = []
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async with session.post(url, json=data_template) as response:
            chunks = []
            async for chunk, _ in response.content.iter_chunks():
                if first_token:
                    last_time = time.time()
                    ttft = last_time - t_start
                    first_token = False
                else:
                    current_time = time.time()
                    tpot.append(current_time - last_time)
                    last_time = current_time
                chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
    
    if if_print:
        print(f"prompt_len: {len(prompt_ids)}, ttft: {ttft*1000:.6f} ms", flush=True)
    
    return ttft, tpot


async def prefill_benchmark():

    # bench
    prompt_len_list = []
    assert args.max_input_length_k != 0
    min_input_len = int(args.min_input_length_k*1024)
    max_input_len = args.max_input_length_k*1024
    max_step_len = args.max_input_length_step_k*1024

    prompt_len = min_input_len
    while prompt_len < max_input_len-100:
        prompt_len_list.append(prompt_len)
        if prompt_len <= max_step_len:
            prompt_len *= 2
        else:
            prompt_len += max_step_len
    prompt_len_list.append(max_input_len-100)

    global global_prefix_ids
    global_prefix_ids = create_prompts(max_input_len)[0]

    if args.prefix_cache:
        prefix_ratio_list = [0, 0.2, 0.5, 0.7, 0.9] # [0, 0.2, 0.4, 0.6, 0.8]
        prefix_ratio_list.reverse() # 颠倒一下
        # warmup prefix
        prompt_len = max_input_len - 100
        print(f"warmup prefix start ...", flush=True)
        # warmup 5 times
        for _ in range(5):
            tasks = [
                asyncio.create_task(
                    single_task(
                        global_prefix_ids[:prompt_len], 
                        1,
                        if_print=False,
                    )
                ) for _ in range(args.dp_size*2)
            ]
            results = await asyncio.gather(*tasks)
        print(f"warmup prefix finished", flush=True)
    else:
        prefix_ratio_list = [0]

    chunk_size = args.chunk_size_k * 1024

    save_path = f"{args.save_path}/{time_stamp}_prefill_throughput_results.json"
    result_dict = {
        "single_req": {},
        "concurrency": {},
    }
    

    # test single concurrency (only for dp_size == 1)
    for prefix_ratio in prefix_ratio_list:
        print(f"\nPrefix Ratio: {prefix_ratio:.2f}", flush=True)
        if args.dp_size == 1:
            throughput_list = []
            for prompt_len in prompt_len_list:
                throughput = await eval_normal(prompt_len=prompt_len, prefix_ratio=prefix_ratio)
                throughput_list.append(throughput)

            result_dict["single_req"][prefix_ratio] = {
                "prompt_len_list": prompt_len_list, 
                "throughput_list": throughput_list
            }

            with open(save_path, "w") as f:
                json.dump(result_dict, f, indent=4)
            print(f"save result dict to {save_path}")

            if args.plot:
                plot_results(result_dict)

        if args.concurrency:
            # test concurrency (all dp_size)
            finished_prompt_len_list = []
            if len(prompt_len_list) > 0:
                throughput_list = []
                for prompt_len in prompt_len_list:
                    concurrency = math.ceil(chunk_size / (prompt_len*(1-prefix_ratio))) #??? 8192 / (51200*0.1) = 2

                    # should_profile = (prefix_ratio == 0.9)
                    should_profile = False
                    if should_profile:
                        profile_dir = f"/data/cwlin/profiling/{time.time()}"
                        print(f"[Profile] start_profile for prefix_ratio={prefix_ratio}, prompt_len={prompt_len}, output_dir={profile_dir}")
                        requests.post(f"{args.base_url}/start_profile", json={"output_dir": profile_dir})

                    throughput = await eval_concurrency(
                        prompt_len=prompt_len,
                        concurrency=concurrency, # 2
                        workload=5 if concurrency > 1 else 1, # 2
                        prefix_ratio=prefix_ratio,
                    )

                    if should_profile:
                        print(f"[Profile] stop_profile")
                        requests.post(f"{args.base_url}/stop_profile")

                    throughput_list.append(throughput)
                    finished_prompt_len_list.append(prompt_len)
                
                    result_dict['concurrency'][prefix_ratio] = {
                        "prompt_len_list": finished_prompt_len_list, 
                        "throughput_list": throughput_list
                    }

                    with open(save_path, "w") as f:
                        json.dump(result_dict, f, indent=4)
                    print(f"save result dict to {save_path}")

                    if args.plot:
                        plot_results(result_dict)


async def eval_normal(prompt_len: int, prefix_ratio: float) -> float:
    print(f"prompt_len: {prompt_len} start eval single request throughput")
    prefix_len = int(prompt_len * prefix_ratio)
    num_repeats = args.num_repeats
    tmp_throughput_list = []

    prefix_prompt = global_prefix_ids[:prefix_len]

    prompt_ids_list = create_prompts(
        prompt_len, 
        num_req=num_repeats+1, 
        prefix_len=prefix_len,
        prefix_ids=prefix_prompt,
    )
    #warmup
    ttft, _ = await single_task(prompt_ids_list[-1], if_print=False)
    for prompt_ids in prompt_ids_list[:-1]:
        ttft, _ = await single_task(prompt_ids)
        throughput = (prompt_len-prefix_len)/ttft
        tmp_throughput_list.append(throughput)
    avg_throughput = sum(tmp_throughput_list) / len(tmp_throughput_list)
    return avg_throughput


async def eval_concurrency(
    prompt_len: int, 
    concurrency: int, 
    workload: int=5,
    prefix_ratio: float=0,
) -> float:
    print(f"prompt_len: {prompt_len} start eval concurrency throughput")
    dp_size = args.dp_size
    num_reqs = concurrency * workload * dp_size

    prefix_len = int(prompt_len * prefix_ratio)
    prefix_prompt = global_prefix_ids[:prefix_len]

    prompt_ids_list = create_prompts(
        prompt_len, 
        num_req=num_reqs,
        prefix_len=prefix_len,
        prefix_ids=prefix_prompt,
    )

    t_start = time.time()
    tasks = [
        asyncio.create_task(
            single_task(prompt_ids, output_len=1, if_print=False)
        ) for prompt_ids in prompt_ids_list
    ]
    results = await tqdm_asyncio.gather(*tasks, desc=f"eval concurrency throughput, prompt_len: {prompt_len}")
    t_end = time.time()
    t_duration = t_end - t_start

    throughput = (prompt_len-prefix_len) * num_reqs / t_duration
    
    print(f"prompt_len: {prompt_len}, throughput: {throughput:.2f} tokens / s", flush = True)
    return throughput


def plot_results(result_dict: dict, save_path: str | None = None):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 5))
    for prefix_ratio, results in result_dict["single_req"].items():
        prompt_len_list = results["prompt_len_list"]
        throughput_list = results["throughput_list"]
        compute_len_list = np.array(prompt_len_list) * (1-float(prefix_ratio))

        plt.plot(compute_len_list, throughput_list, marker='o', label=f"Single Request, Prefix={prefix_ratio}")

    for prefix_ratio, results in result_dict["concurrency"].items():
        prompt_len_list = results["prompt_len_list"]
        throughput_list = results["throughput_list"]
        compute_len_list = np.array(prompt_len_list) * (1-float(prefix_ratio))
        
        plt.plot(compute_len_list, throughput_list, marker='o', label=f"Concurrency, Prefix={prefix_ratio}")

    plt.ylim(bottom=0)
    if args.x_log_scale:
        plt.xscale('log')
    plt.xlabel('New Compute Length')
    plt.ylabel('Throughput (tokens/s)')
    plt.title('Prefill Throughput Benchmark')
    plt.legend(bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=4)

    plt.tight_layout(pad=0.1)
    plt.grid(True)
    if save_path == None:
        save_path = f"{args.save_path}/{time_stamp}_prefill_throughput.png"
    print(f"save result figure to {save_path}")
    plt.savefig(save_path)


def replot():
    assert args.result_path != ''
    with open(args.result_path, "r") as f:
        result_dict = json.load(f)

    suffix = '_prefill_throughput_results.json'
    if args.result_path.endswith(suffix):
        prefix = args.result_path.rstrip(suffix)
    else:
        prefix = f"{args.save_path}/{time_stamp}"
    new_save_path = f"{prefix}_prefill_throughput.png"
    plot_results(result_dict=result_dict, save_path=new_save_path)
    


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Prefill Throughput Benchmark')

    argparser.add_argument(
        '--base-url',
        type=str,
        default='http://127.0.0.1:8080',
    )

    argparser.add_argument(
        '--model',
        type=str,
        default='DeepSeek-V3',
    )

    argparser.add_argument(
        '--min-input-length-k',
        type=float,
        default=1,
        help='min input length test, '
        '--min-input-length-k 1 means 1024.'
    )

    argparser.add_argument(
        '--max-input-length-k',
        type=int,
        default=0,
        help='max input length test, '
        '--max-input-length-k 128 means 128k.'
    )

    argparser.add_argument(
        '--max-input-length-step-k',
        type=int,
        default=32,
    )

    argparser.add_argument(
        '--chunk-size-k',
        type=int,
        default=8,
    )

    argparser.add_argument(
        '--num-repeats',
        type=int,
        default=3,
    )

    argparser.add_argument(
        '--prefix-cache',
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    argparser.add_argument(
        '--concurrency',
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    argparser.add_argument(
        '--dp-size',
        type=int,
        default=1,
    )

    argparser.add_argument(
        '--str-input',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Default we use token_ids_list as prompt. '
        'If inference engine does not support token_ids_list as input, '
        'we can set --str-input and we will detokenizer token_ids_list to str.'
    )

    argparser.add_argument(
        '--tokenizer-path',
        type=str,
        default='',
    )

    argparser.add_argument(
        '--plot',
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    argparser.add_argument(
        '--x-log-scale',
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    argparser.add_argument(
        '--save-path',
        type=str,
        default="./",
    )

    argparser.add_argument(
        '--replot',
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    argparser.add_argument(
        '--result-path',
        type=str,
        default="",
    )

    args = argparser.parse_args()

    if args.replot:
        replot()
    else:
        asyncio.run(prefill_benchmark())


# python3 prefill_throughput.py --base-url http://127.0.0.1:8080 --model DeepSeek-V3 --max-input-length-k 128 