"""
vLLM 推理性能基准测试脚本
用法: python benchmark.py --model <model_path> --tp-size <n>
"""

import argparse
import json
import time
from openai import OpenAI


def benchmark_throughput(client, model_name, num_requests=100, input_len=512, output_len=512):
    """测试推理吞吐量"""
    import random
    import string

    # 生成随机 prompt
    def random_prompt(length):
        return "".join(random.choices(string.ascii_letters + " ", k=length))

    print(f"Running {num_requests} requests (input={input_len}, output={output_len})...")

    total_tokens = 0
    total_time = 0

    for i in range(num_requests):
        prompt = random_prompt(input_len)
        start = time.time()

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=output_len,
            temperature=0.0,
        )

        elapsed = time.time() - start
        tokens = len(response.choices[0].message.content)
        total_tokens += tokens
        total_time += elapsed

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{num_requests}] avg: {total_tokens/total_time:.1f} tok/s")

    throughput = total_tokens / total_time
    print(f"\nResults:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} tok/s")

    return {
        "num_requests": num_requests,
        "input_length": input_len,
        "output_length": output_len,
        "total_tokens": total_tokens,
        "total_time": round(total_time, 2),
        "throughput": round(throughput, 1),
    }


def benchmark_ttft(client, model_name, num_requests=50, input_len=512):
    """测试首 Token 延迟"""
    import random
    import string

    def random_prompt(length):
        return "".join(random.choices(string.ascii_letters + " ", k=length))

    print(f"Testing TTFT with {num_requests} requests...")

    ttfts = []
    for i in range(num_requests):
        prompt = random_prompt(input_len)
        start = time.time()

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0.0,
        )

        ttft = (time.time() - start) * 1000  # ms
        ttfts.append(ttft)

    avg_ttft = sum(ttfts) / len(ttfts)
    p50_ttft = sorted(ttfts)[len(ttfts) // 2]
    p99_ttft = sorted(ttfts)[int(len(ttfts) * 0.99)]

    print(f"\nTTFT Results:")
    print(f"  Avg: {avg_ttft:.1f} ms")
    print(f"  P50: {p50_ttft:.1f} ms")
    print(f"  P99: {p99_ttft:.1f} ms")

    return {
        "avg_ttft_ms": round(avg_ttft, 1),
        "p50_ttft_ms": round(p50_ttft, 1),
        "p99_ttft_ms": round(p99_ttft, 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCU LLM Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Model path or name")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=512)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    results = {"model": args.model, "tp_size": args.tp_size}

    results["throughput"] = benchmark_throughput(
        client, args.model, args.num_requests, args.input_len, args.output_len
    )
    print()
    results["ttft"] = benchmark_ttft(client, args.model, num_requests=50, input_len=args.input_len)

    # 保存结果
    output_file = f"benchmark_{args.model.replace('/', '_')}_tp{args.tp_size}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
