from Constants import *
from run_model import *
from utils import *
import sys
import re

if __name__ == '__main__':
    # 1. Read input file
    manifest_f = sys.argv[1]
    manifest = []
    file = open(manifest_f, "r")
    lines = file.readlines()
    for line in lines:
        manifest.append(line.strip())
    path = manifest[0]
    manifest.pop(0)
    ids = []
    for line in manifest:
        if re.search("\d",line):
            ids.append(line)

    # 3. create single input without json files
    create_data_without_answers(path_dir=path, f_type='input_question_answers', story_ids=ids)

    # 4. Predict from model
    with open(os.devnull, "w", encoding='utf-8') as target:
        sys.stdout = target
        run_model(train_file='./data/train.json', 
                    predict_file=f'{path}/input_question_answers.json', 
                    do_train=False, do_predict=True)
    # 5. Read Predictions from predicted json and print to stdout
    sys.stdout = sys.__stdout__
    predicted_answers_all = "./output_dir/predictions.json"
    ans = []
    with open(predicted_answers_all) as f:
        qa_dict = json.load(f)

    for id in ids:
        file = open(path + id + ".questions", "r")
        lines = file.readlines()
        for line in lines:
            se = re.search("QuestionID\: (.*)",line)
            if se:
                q_id = se.group(1).strip()
                print("QuestionID:", q_id)
                print("Answer:", qa_dict[q_id])
                print("")


