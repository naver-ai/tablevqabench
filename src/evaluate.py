"""
TableVQA-Bench
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
"""

import argparse
import os

from typing import List
from evaluate_tablevqa_dataset import evaluate_datasets

def main(args):
    evaluation_datasets = args.evaluation_datasets
    score_keys = "accuracy"   
    os.makedirs(args.output_path, exist_ok=True)
    print(f"# [info] Output Folder: {args.output_path}") 

    # get evaluation dataset paths
    evaluation_dataset_paths = _get_input_paths(
        evaluation_datasets,
        args.root_input_path,
    )      
    evaluate_datasets(
        evaluation_dataset_paths,
        args.output_path,
        score_keys
    )

def _get_input_paths(
    evaluation_datasets: List[str],
    input_path: str,
) -> List[str]:
    
    input_files = []    
    for evaluation_dataset in evaluation_datasets:     
        input_file = os.path.join(input_path, evaluation_dataset + f".json")    
        print(f"# [info] input file: {input_file}")
        input_files.append(input_file)    
    print("")
    return input_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="TableVQA-Bench Evaluator",
        description="Evaluatable Dataset: VWTQ, VWTQ-Syn, VTabFACT, FinTabNetQA",
    )     
    parser.add_argument(
        "--root_input_path",
        type=str,
        default="./outputs/gpt4v",
        help="root input path of result files",
    )
    parser.add_argument(
        "--evaluation_datasets",
        type=str,
        required=True,
        nargs="+",
        help="names of evaluation dataset. candidates: vwtq, vwtq_syn, vtabfact, fintabnetqa",    
    )   
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs/gpt4v_evaluated",
        help="evaluation output path",
    )
    args = parser.parse_args()
    main(args)
