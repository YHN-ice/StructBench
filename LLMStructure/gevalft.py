import glob, json, os

from api import gpt_judge

_URL = "./benchmarkMiniTest"

if __name__ == "__main__":
    filepath = _URL
    filenames = glob.glob(f'{filepath}/**/**/**/*.json')


    def extract_QA(data):
        return data['Q'], data['A']


    infile_QA = list(extract_QA(json.load(open(file, 'r'))) for file in filenames)
    # model_id = "base_finetune" # base_finetune , fewshot_finetune , mixrulehint_finetune , before_finetune , mix_finetune , rulehint_finetune ,
    model_ids = ["before_finetune"]


    def batch_trim(*args)-> [str]:
        def trim(s: str) -> str:
            return ''.join(s.split())
        return map(trim, args)

    def get_pair(c):
        return c['label'], c['predict']


    def mean(nums: [float]) -> float:
        return sum(nums) / len(nums)


    for model_id in model_ids:
        collect_true = []
        judge_res_array = []
        judge_hist_array = []


        factory_res_dir = os.path.join("..", "LLaMA-Factory", "eval_chkpts_minitest", model_id)
        if not model_id.startswith("before"):
            factory_res_dir = os.path.join(factory_res_dir, "checkpoint-5700")

        factory_filename = os.path.join(factory_res_dir, "generated_predictions.jsonl")
        factory_output = [factory_res_dir + f"human_eval_{tag}.jsonl" for tag in ["res", "hist"]]
        factory_label_pred_pairs = list(get_pair(json.loads(s)) for s in open(factory_filename, 'r').readlines())

        res_output, hist_output = factory_output
        for i, (inf, (factory_label, factory_response)) in enumerate(zip(infile_QA, factory_label_pred_pairs)):
            thin_q, a, factory_label, factory_response = batch_trim(inf[0], inf[1], factory_label, factory_response)
            if a != factory_label: # this branch is expected to never be executed
                judge_res_array.append(f"differ at index{i}: {a}!={factory_label}")
                judge_hist_array.append([])
                print(f"differ at index{i}: {a}!={factory_label}")
            else:
                judge_res, judge_hist = gpt_judge(factory_response, a, thin_q)
                print(f"the judge is {judge_res}")
                collect_true.append(1.0 if judge_res == "True" else 0)
                judge_res_array.append(judge_res)
                judge_hist_array.append(judge_hist)

            with open(res_output, 'w') as f:
                f.write(f'{judge_res_array[-1]}\n')
            with open(hist_output, 'w') as f:
                f.write(f'{json.dumps(judge_hist_array[-1])}\n')
        print(f"the judges are {collect_true}")
        print(f"acc for model {model_id} is {mean(collect_true)}")
