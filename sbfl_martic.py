import os
from pathlib import Path
from typing import List
import pandas as pd
from agent_step import AgentStep


def convert_path_to_tuple(input_dir: str, output_dir: str, is_hierarchical=False, usage_percentage=1.0):
    """
    转换路径到元组格式
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    :param is_hierarchical: 是否使用层级结构
    :param usage_percentage: 使用日志的百分比 (0.0-1.0)，例如0.5表示只使用前50%的日志
    """
    input_path = Path(input_dir)
    path_dir = input_path / "llm_cluster3"
    output_path = Path(output_dir)
    tuple_path = output_path / "tuple"
    tuple_path.mkdir(parents=True, exist_ok=True)

    all_agentsteps = set()
    file_agentsteps = []
    file_agentsteps_all = []
    # file_bigrams = []  # 新增:存储每个文件的二元组序列

    indexed_path_files = []
    for path_file in path_dir.glob("*.jsonl"):
        try:
            index = int(path_file.stem)
            indexed_path_files.append((index, path_file))
        except ValueError:
            continue

    indexed_path_files.sort(key=lambda x: x[0])

    for _, path_file in indexed_path_files:
        with open(path_file, "r", encoding="utf-8") as f:
            steps = set()
            steps_all = []
            if input_path.parent.name != "algorithmgenerated_plan":
                next(f, None)
            if f is not None:
                # 首先读取所有行以计算要使用的行数
                all_lines = f.readlines()
                total_lines = len(all_lines)
                lines_to_use = int(total_lines * usage_percentage)
                
                # 只处理前 lines_to_use 行
                for i, line in enumerate(all_lines):
                    if i >= lines_to_use:
                        break
                    step = AgentStep.from_jsonl_line(line, is_hierarchical)
                    if step and step.agent == "Computer_terminal":
                        continue
                    if step:
                        steps.add(step)
                        steps_all.append(step)

            file_agentsteps.append(steps)
            all_agentsteps.update(steps)
            file_agentsteps_all.append(steps_all)


            tuple_file = tuple_path / f"{path_file.stem}.tuple"
            with open(tuple_file, "w", encoding="utf-8") as out:
                for step in sorted(steps, key=lambda x: (x.agent, x.action, x.state)):
                    out.write(f"{step}\n")


    all_agentsteps_sorted = sorted(all_agentsteps, key=lambda x: (x.agent, x.action, x.state))
    with open(tuple_path / "all.tuple", "w", encoding="utf-8") as out:
        for step in all_agentsteps_sorted:
            out.write(f"{step}\n")

    return file_agentsteps, all_agentsteps_sorted, file_agentsteps_all


def generate_step_matrix(file_agentsteps: List[set], all_agentsteps_sorted: List[AgentStep],
                         input_dir: str, output_dir: str, file_agentsteps_all: List[List]):
    matrix = []
    for step in all_agentsteps_sorted:
        row = []
        for i in range(len(file_agentsteps)):
            if step in file_agentsteps[i]:
                row.append(file_agentsteps_all[i].count(step))
            else:
                row.append(0)
        # row = [1 if step in step_set else 0 for step_set in file_agentsteps]
        matrix.append(row)
    df = pd.DataFrame(matrix)

    matrix_file = Path(output_dir) / "step_matrix.csv"
    df.to_csv(matrix_file, index=False, header=False)


    answer_file = Path(input_dir) / "answers.txt"
    test_file = Path(output_dir) / "test"
    if answer_file.exists():
        with open(answer_file, "r", encoding="utf-8") as fin, open(test_file, "w", encoding="utf-8") as fout:
            for line in fin:
                if line.startswith("[Expected Answer]:"):
                    break
                parts = line.strip().split("###")
                if len(parts) >= 3:
                    fout.write(parts[1].strip() + "\n")
    else:
        print(f"Warning: {answer_file} not found.")

    return str(matrix_file), str(test_file)


def get_all_steps_and_all_bigrams(input_dir: str):
    all_agentsteps = []
    with open(os.path.join(input_dir, "tuple", "all.tuple") , "r", encoding="utf-8") as out:
        for line in out.readlines():
            all_agentsteps.append(line.strip())
            # step = AgentStep.from_jsonl_line(line)
            # all_agentsteps.append(step)
    return all_agentsteps


def get_all_matrix_files(input_dir: str):
    return os.path.join(input_dir, "step_matrix.csv"), os.path.join(input_dir, "test")


def generate_all_matrix(analysis_dir: str, output_dir: str, usage_percentage=1.0):
    """
    生成所有矩阵
    :param analysis_dir: 分析目录
    :param output_dir: 输出目录
    :param usage_percentage: 使用日志的百分比 (0.0-1.0)
    """
    # 如果使用部分日志，在输出路径中添加百分比标识
    if usage_percentage < 1.0:
        output_dir = f"{output_dir}_p{int(usage_percentage*100)}"
    
    for dirname in os.listdir(analysis_dir):
        input_example = os.path.join(analysis_dir, dirname)
        if not os.path.isdir(input_example): continue
        output_example = os.path.join(output_dir, dirname)
        os.makedirs(output_example, exist_ok=True)
        file_agentsteps, all_agentsteps_sorted, file_agentsteps_all = convert_path_to_tuple(input_example, output_example, usage_percentage=usage_percentage)
        generate_step_matrix(file_agentsteps, all_agentsteps_sorted, input_example, output_example, file_agentsteps_all)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SBFL matrices from trajectories.")
    parser.add_argument("--usage_percentage", type=float, default=1.0, 
                       help="The percentage of a log file used (0.0-1.0), e.g., 0.5 means using first 50%% of logs")
    args = parser.parse_args()
    
    # 验证 usage_percentage 的范围
    if not 0.0 < args.usage_percentage <= 1.0:
        parser.error("usage_percentage must be in range (0.0, 1.0]")
    
    print(f"Generating matrices with {args.usage_percentage*100:.1f}% of logs...")
    
    generate_all_matrix("data/trajectories/gaia_level1", "data/sbfl_matrixs/hand_crafted", args.usage_percentage)
    generate_all_matrix("data/trajectories/gaia_level2", "data/sbfl_matrixs/hand_crafted", args.usage_percentage)
    generate_all_matrix("data/trajectories/gaia_level3", "data/sbfl_matrixs/hand_crafted", args.usage_percentage)
    generate_all_matrix("data/trajectories/assistantbench_medium", "data/sbfl_matrixs/hand_crafted", args.usage_percentage)
    generate_all_matrix("data/trajectories/assistantbench_hard", "data/sbfl_matrixs/hand_crafted", args.usage_percentage)

    generate_all_matrix("data/trajectories/algorithmgenerated_plan", "data/sbfl_matrixs/algorithm_generated", args.usage_percentage)
    generate_all_matrix("data/trajectories/algorithmgenerated_normal", "data/sbfl_matrixs/algorithm_generated", args.usage_percentage)





