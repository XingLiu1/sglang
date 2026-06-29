import asyncio
import aiohttp
import json
import time
import random
import argparse
from datetime import datetime
import numpy as np

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


def completion_data(model: str, base_url: str, prompt: str | list[int], ouptut_len: int=10):
    if args.str_input:
        prompt = detokenize_to_str(prompt)
    data_template = {
        "model": model,
        "stream": True,
        "max_tokens": ouptut_len,
        "temperature": 0.6,
        "prompt": prompt,
        "top_p": 0.95,
        "ignore_eos": True,
        "stream_options" : {
            "include_usage": True},
    }
    url = f"{base_url}/v1/completions"
    return data_template, url
    
    
async def single_task(prompt_ids: list[int], output_len: int=10, sleep_time: float=0, if_print: bool=False):

    if sleep_time > 0:
        await asyncio.sleep(sleep_time)

    data_template, url = completion_data(args.model, args.base_url, prompt=prompt_ids, ouptut_len=output_len)

    t_start = time.time()    
    timeout = aiohttp.ClientTimeout(total=6000)
    connector = aiohttp.TCPConnector(limit=0)  # 连接数
    first_token = True
    first_token_time = t_start
    last_token_time = t_start
    ttft = 0
    response_interval = []
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async with session.post(url, json=data_template) as response:
            async for chunk in response.content:

                chunk = chunk.decode('utf-8')
                if chunk.startswith("data: "):
                    chunk_str = chunk.strip()[6:].strip()
                    if chunk_str != "[DONE]":
                        chunk_dict = json.loads(chunk_str)
                        if first_token:
                            first_token_time = time.time()
                            first_token = False
                            last_token_time = first_token_time
                        else:
                            current_time = time.time()
                            response_interval_time = round((current_time - last_token_time)*1000, 2)
                            response_interval.append(response_interval_time)
                            last_token_time = current_time
                        if 'usage' in chunk_dict and chunk_dict.get("usage") is not None:
                            usage_data = {
                                'prompt_tokens': chunk_dict["usage"]["prompt_tokens"],
                                'completion_tokens': chunk_dict["usage"]["completion_tokens"],
                                'total_tokens': chunk_dict["usage"]["total_tokens"]
                            }
                    else:
                        last_token_time = time.time()

    ttft = first_token_time - t_start
    tpot = np.median(np.array(response_interval)).item()/1000
    
    if if_print:
        print(f"ttft: {ttft*1000:.2f} ms, step time: {tpot * 1000:.2f} ms, "
              f"output_len: {usage_data['completion_tokens']}, "
              f"len_response_interval = {len(response_interval)}, "
              f"response_interval = {response_interval}", 
              flush=True)
    return ttft, tpot
    
async def decode_benchmark():

    bs_list = []
    max_bs_step = args.max_batch_size_step
    cur_bs = args.min_batch_size
    while cur_bs <= args.max_batch_size:
        bs_list.append(cur_bs)
        if cur_bs < max_bs_step:
            cur_bs *= 2
        else:
            cur_bs += max_bs_step
    
    max_kv_cache_len = args.max_total_kvcache_length_k*1024
    min_input_len = int(args.min_input_length_k*1024)
    max_input_len = args.max_input_length_k*1024
    max_step_len = args.max_input_length_step_k*1024

    num_repeats = args.num_repeats
    dp_size = args.dp_size

    assert args.other_request_output_length > args.target_request_output_length
    reserve_output_length = args.other_request_output_length + 20

    global global_prefix_ids
    random.seed(60)
    global_prefix_ids = create_prompts(max_input_len)[0]
    prompt_len = max_input_len - reserve_output_length
    print(f"warmup prefix start ...", flush=True)
    tasks = [
        asyncio.create_task(
            single_task(
                global_prefix_ids[:prompt_len], 
                1,
            )
        ) for _ in range(dp_size*2)
    ]
    results = await asyncio.gather(*tasks)
    print(f"warmup prefix finished", flush=True)

    save_path = f"{args.save_path}/{time_stamp}_decode_tpot_results.json"

    # bench
    tpot_var_bs_list = []
    for i, bs in enumerate(bs_list):
        print(f"\n--- Local BS={bs} ---")
        max_local_input_len = min(
            max_kv_cache_len // bs, 
            max_input_len
        ) - reserve_output_length
        assert max_local_input_len > 0
        prompt_len_list = []
        prompt_len = min_input_len
        while prompt_len < max_local_input_len:
            prompt_len_list.append(prompt_len)
            if prompt_len <= max_step_len:
                prompt_len *= 2
            else:
                prompt_len += max_step_len
        prompt_len_list.append(max_local_input_len)

        tpot_var_len_list = []
        for prompt_len in prompt_len_list:
            tmpt_tpot_list = []
            print(f"\nPrompt length: {prompt_len}")
            for _ in range(num_repeats):
                prompt_ids_list = create_prompts(
                    prompt_len, 
                    num_req=bs*dp_size,
                    prefix_len=prompt_len,
                    prefix_ids=global_prefix_ids[:prompt_len]
                )

                tasks = []
                for i, prompt_ids in enumerate(prompt_ids_list):
                    if i == bs*dp_size-1:
                        tasks.append(
                            asyncio.create_task(
                                single_task(
                                    prompt_ids, 
                                    args.target_request_output_length, 
                                    args.target_request_wait_time, 
                                    if_print=True
                                )
                            )
                        )
                    else:
                        tasks.append(
                            asyncio.create_task(
                                single_task(
                                    prompt_ids, 
                                    args.other_request_output_length,
                                )
                            )
                        )
                results = await asyncio.gather(*tasks)
                tpot = results[-1][1]
                tmpt_tpot_list.append(tpot)
            avg_tpot = sum(tmpt_tpot_list) / len(tmpt_tpot_list)
            tpot_var_len_list.append(avg_tpot)
        
        tpot_var_bs_list.append(
            {
                "bs": bs,
                "prompt_len_list": prompt_len_list,
                "tpot_var_len_list": tpot_var_len_list
            }
        )

        result_dict = {
            "bs_list": bs_list[:len(tpot_var_bs_list)],
            "tpot_var_bs_list": tpot_var_bs_list
        }

        with open(save_path, "w") as f:
            json.dump(result_dict, f, indent=4)
        print(f"save result dict to {save_path}")

        if args.plot:
            plot_results(result_dict)


def plot_results(result_dict: dict, save_path: str | None = None):
    import matplotlib.pyplot as plt
    import numpy as np

    bs_list = result_dict['bs_list']
    tpot_var_bs_list = result_dict['tpot_var_bs_list']

    plt.figure(figsize=(10, 5))
    for i, bs in enumerate(bs_list):
        prompt_len_list = tpot_var_bs_list[i]["prompt_len_list"]
        tpot_var_len_list = np.array(tpot_var_bs_list[i]["tpot_var_len_list"]) * 1000
        plt.plot(prompt_len_list, tpot_var_len_list, marker='o', label=f'bs={bs}')
    

    if args.x_log_scale:
        plt.xscale('log')
    plt.ylim(bottom=0)
    plt.xlabel('Context Length')
    plt.ylabel('Step Time (ms)')
    plt.title('Decode TPOT Benchmark')
    plt.legend(bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=7)

    plt.tight_layout(pad=0.1)
    plt.grid(True)
    if save_path == None:
        save_path = f"{args.save_path}/{time_stamp}_decode_tpot.png"
    print(f"save result figure to {save_path}")
    plt.savefig(save_path)


def replot():
    assert args.result_path != ''
    with open(args.result_path, "r") as f:
        result_dict = json.load(f)

    suffix = '_decode_tpot_results.json'
    if args.result_path.endswith(suffix):
        prefix = args.result_path.rstrip(suffix)
    else:
        prefix = f"{args.save_path}/{time_stamp}"
    new_save_path = f"{prefix}_decode_tpot.png"
    plot_results(result_dict=result_dict, save_path=new_save_path)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Decode Step Time Benchmark')

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
        '--max-total-kvcache-length-k',
        type=int,
        default=32,
    )

    argparser.add_argument(
        '--min-batch-size',
        type=int,
        default=1,
    )

    argparser.add_argument(
        '--max-batch-size',
        type=int,
        default=32,
    )

    argparser.add_argument(
        '--max-batch-size-step',
        type=int,
        default=16,
    )

    argparser.add_argument(
        '--other-request-output-length',
        type=int,
        default=128,
    )

    argparser.add_argument(
        '--target-request-output-length',
        type=int,
        default=32,
    )

    argparser.add_argument(
        '--target-request-wait-time',
        type=float,
        default=0.1,
    ) 

    argparser.add_argument(
        '--num-repeats',
        type=int,
        default=3,
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
        default=True
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
        asyncio.run(decode_benchmark())


# python3 decode_throughput.py --base-url http://127.0.0.1:8080 --model DeepSeek-V3 --max-input-length-k 128 --max-total-kvcache-length-k 300 --max-batch-size 32 