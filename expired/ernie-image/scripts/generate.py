#!/usr/bin/env python3
"""
百度文心 ERNIE-Image-Turbo 图片生成脚本

通过飞桨 AI Studio 星河社区 API 生成图片，使用 requests 库调用。
API Key 从 .env 文件中读取。
"""

import argparse
import base64
import json
import os
import sys
import requests
from pathlib import Path


def load_env(env_path: str = None) -> str:
    """
    从 .env 文件加载 AISTUDIO_API_KEY

    Args:
        env_path: .env 文件路径，默认为脚本所在目录的上级目录下的 .env

    Returns:
        API Key 字符串

    Raises:
        SystemExit: 当 .env 文件不存在或 API Key 为空时退出
    """
    if env_path is None:
        skill_dir = Path(__file__).resolve().parent.parent
        env_path = skill_dir / ".env"
    else:
        env_path = Path(env_path)

    if not env_path.exists():
        print(f"错误: .env 文件不存在: {env_path}")
        print(f"请创建 .env 文件并写入: AISTUDIO_API_KEY=your_api_key_here")
        print(f"获取 API Key: https://aistudio.baidu.com/account/accessToken")
        sys.exit(1)

    api_key = None
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("AISTUDIO_API_KEY="):
                api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

    if not api_key:
        print(f"错误: .env 文件中未找到 AISTUDIO_API_KEY")
        print(f"请在 {env_path} 中写入: AISTUDIO_API_KEY=your_api_key_here")
        print(f"获取 API Key: https://aistudio.baidu.com/account/accessToken")
        sys.exit(1)

    return api_key


def generate_image(
    prompt: str,
    api_key: str,
    size: str = "1024x1024",
    n: int = 1,
    response_format: str = "b64_json",
    seed: int = None,
    use_pe: bool = True,
    num_inference_steps: int = 8,
    guidance_scale: float = 1.0,
    output: str = "output.png",
) -> list:
    """
    调用 ERNIE-Image-Turbo API 生成图片

    Args:
        prompt: 图片描述文本
        api_key: AI Studio Access Token
        size: 图片尺寸，支持 1024x1024/1376x768/1264x848/1200x896/896x1200/848x1264/768x1376
        n: 生成图片数量，1-4
        response_format: 返回格式，b64_json 或 url
        seed: 随机种子，None 则随机
        use_pe: 是否开启提示词增强
        num_inference_steps: 推理步数，4-20
        guidance_scale: 引导强度，1.0-7.5
        output: 输出文件路径（仅 b64_json 模式）

    Returns:
        生成的图片文件路径列表

    Raises:
        SystemExit: 当 API 调用失败时退出
    """
    url = "https://aistudio.baidu.com/llm/lmapi/v3/images/generations"

    headers = {
        "Authorization": f"bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "ERNIE-Image-Turbo",
        "prompt": prompt,
        "n": n,
        "response_format": response_format,
        "size": size,
        "use_pe": use_pe,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }

    if seed is not None:
        payload["seed"] = seed

    print(f"正在生成图片...")
    print(f"  Prompt: {prompt}")
    print(f"  尺寸: {size}")
    print(f"  数量: {n}")

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到 API 服务器，请检查网络连接")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("错误: API 请求超时，请稍后重试")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"错误: API 返回状态码 {resp.status_code}")
        try:
            error_data = resp.json()
            print(f"  错误信息: {json.dumps(error_data, ensure_ascii=False, indent=2)}")
        except Exception:
            print(f"  响应内容: {resp.text[:500]}")
        if resp.status_code == 401:
            print("  提示: 请检查 .env 中的 AISTUDIO_API_KEY 是否正确")
        sys.exit(1)

    result = resp.json()
    output_paths = []

    if response_format == "b64_json":
        output_stem = Path(output).stem
        output_suffix = Path(output).suffix or ".png"
        output_dir = Path(output).parent

        for i, item in enumerate(result.get("data", [])):
            b64_data = item.get("b64_json")
            if not b64_data:
                print(f"警告: 第 {i+1} 张图片无 b64_json 数据，跳过")
                continue

            try:
                image_bytes = base64.b64decode(b64_data)
            except Exception as e:
                print(f"错误: 解码第 {i+1} 张图片失败: {e}")
                continue

            if n == 1:
                file_path = output_dir / f"{output_stem}{output_suffix}"
            else:
                file_path = output_dir / f"{output_stem}_{i}{output_suffix}"

            with open(file_path, "wb") as f:
                f.write(image_bytes)

            output_paths.append(str(file_path))
            print(f"  已保存: {file_path}")

    elif response_format == "url":
        for i, item in enumerate(result.get("data", [])):
            img_url = item.get("url")
            if img_url:
                output_paths.append(img_url)
                print(f"  图片链接 {i+1}: {img_url}")

    if not output_paths:
        print("错误: 未生成任何图片")
        sys.exit(1)

    return output_paths


def main():
    """
    命令行入口，解析参数并调用图片生成 API
    """
    parser = argparse.ArgumentParser(
        description="百度文心 ERNIE-Image-Turbo 图片生成工具"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="图片描述文本",
    )
    parser.add_argument(
        "--size", "-s",
        type=str,
        default="1024x1024",
        choices=[
            "1024x1024", "1376x768", "1264x848", "1200x896",
            "896x1200", "848x1264", "768x1376",
        ],
        help="图片尺寸 (默认: 1024x1024)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="生成图片数量 1-4 (默认: 1)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.png",
        help="输出文件路径 (默认: output.png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，固定后可复现结果",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="b64_json",
        choices=["b64_json", "url"],
        help="返回格式 (默认: b64_json)",
    )
    parser.add_argument(
        "--no-pe",
        action="store_true",
        help="关闭提示词增强",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="推理步数 4-20 (默认: 8)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.0,
        help="引导强度 1.0-7.5 (默认: 1.0)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help=".env 文件路径",
    )

    args = parser.parse_args()

    api_key = load_env(args.env)

    generate_image(
        prompt=args.prompt,
        api_key=api_key,
        size=args.size,
        n=args.n,
        response_format=args.format,
        seed=args.seed,
        use_pe=not args.no_pe,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        output=args.output,
    )


if __name__ == "__main__":
    main()
