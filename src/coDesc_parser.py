# Converts CoDesc_Decsription.txt file (which is a list of json) to multiple json files, each represnting one entry.
import json
import os

def process_file(src_file: str, output_dir: str):
    with open(src_file, "r") as f:
        line1 = f.readline()  # [

        count = 0
        subfolder_count = 0
        while True:
            example_lines = []
            end = False
            for i in range(0, 11):
                line = f.readline()
                if line == "]" or line == "":
                    end = True
                example_lines.append(line)
            
            if end:
                break

            if example_lines[-1][-2] == ",":
                example_lines[-1] = example_lines[-1][:-2]

            subfolder = os.path.join(output_dir, f"subfolder_{subfolder_count}")
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)

            with open(os.path.join(subfolder, f"example_{count % 50000}.json"), "w") as out:
                for line in example_lines:
                    out.write(line)

            print(f"subfolder_{subfolder_count}/example_{count % 50000}.json")

            count += 1
            if count % 50000 == 0:
                subfolder_count += 1

def load_example(index : int, base_dir : str):
    with open(os.path.join(base_dir, f"subfolder_{index // 50000}/example_{index % 50000}.json"), "r") as f:
            data = json.load(f)
    
    return data["original_code"].replace("\\\"", "\"").replace("\\n", "\n")

def load_examples(indices : list[int], base_dir : str):
    return [load_example(index, base_dir) for index in indices]

def large_text_file_dump(indices : list[int], base_dir : str, DELIMITER : str, output_file : str):
    examples = load_examples(indices, base_dir)
    examples[0] = f"{DELIMITER}{examples[0]}"
    with open(os.path.join(base_dir, output_file), "w") as f:
        for example in examples:
            try:
                f.write(f"{example}{DELIMITER}" + "\n")
            except:
                print(example)