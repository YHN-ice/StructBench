import functools
import glob, os, json
import time

import MYSQL, MYjson, MYlatex, MYmarkdown, MYorg, MYtree, MYxml, MYyaml, MYPython
import testee
from api import get_output_filename


def ROUGEx(pred, gold):
    from rouge_score import rouge_scorer
    scoreer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) # skipped if exception ocurred
    return scoreer.score(gold, pred)


def bleu_4(pred, gold):
    import nltk
    return nltk.translate.bleu_score.sentence_bleu([gold], pred)


def eval_metric(preds, labels):
    import jieba
    import numpy as np
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_chinese import Rouge
    score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
    for pred, label in zip(preds, labels):
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))

        if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
            result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
        else:
            rouge = Rouge()
            scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
            result = scores[0]

        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))

        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))
    return {k: float(np.mean(v)) for k, v in score_dict.items()}


def test():
    pred = "She was a boy from a long time ago in ancient Greek"
    gold = "False"
    print("BLEU-4:",bleu_4(pred, gold))
    print("ROUGE-{1,2,l}:",ROUGEx(pred, gold))
    print("ROUGE-1:", list(k for k in ROUGEx(pred, gold)['rouge1']))


def skip(include, filename):
    return 'all' not in include and all(inc not in filename for inc in include)


def get_stats(root_dir, tp, tst, baseline, rule_hint_method, include, res_dict):
    filenames = glob.glob(f'./{root_dir}/{tp.__name__}/**/**/*.json')
    sample_size = len(filenames)
    exact_match_cnt = []
    gpt_judge_true_cnt = []
    rouge_scores = []
    preds, labels = [], []
    valid_cnt = 0
    for i in range(sample_size):
        if skip(include, filenames[i]):
            continue
        root_output = f'output_load/{tst.ident}' + ("_engi_" if not baseline else '_base_')
        output_filename = get_output_filename(filenames[i], root_output, rule_hint_method)
        if not os.path.exists(output_filename):
            print(f"test result {output_filename} is not available yet.")
            continue
        with open(output_filename, 'r') as f:
            valid_cnt+=1
            content = json.load(f)
            _predict, _gold = content['hist'][-1], content['gold']
            preds.append(_predict)
            labels.append(_gold)
            [em_ctnt, rg_ctnt, gpt_judge_ctnt] = content['scores']
            exact_match_cnt.append(1.0 if em_ctnt else 0.0)
            if gpt_judge_ctnt == "True":
                gpt_judge_true_cnt.append(1.0)
            elif gpt_judge_ctnt == "False":
                gpt_judge_true_cnt.append(0.0)
            rouge_scores.append(rg_ctnt)

    expect_cnt = len(filenames)
    actual_cnt = valid_cnt
    base_engi = 'base' if baseline else 'engi'
    if not exact_match_cnt and not gpt_judge_true_cnt and not rouge_scores:
        result = f"-({actual_cnt}/{expect_cnt}){tp.__name__}:{tst.ident}:{base_engi}:No data yet."
    else:
        def avg(l):
            return sum(l)/len(l)
        rg1_res, rg2_res, rgl_res, bleu4_res = ['not available'] * 4

        # comment out since eval_metric is slow
        # scores = eval_metric(preds, labels)
        # rg1_res, rg2_res, rgl_res, bleu4_res = scores['rouge-1'], scores['rouge-2'], scores['rouge-l'], scores['bleu-4']

        result = f"+({actual_cnt}/{expect_cnt}){tp.__name__}:{tst.ident}:{base_engi}:\n\texact_match:{avg(exact_match_cnt)}\n\tgpt_judge:{avg(gpt_judge_true_cnt)}\n\troughe_mean:{avg(rouge_scores)}"
        result += f'\n\tbleu-4:{bleu4_res}'
        result += f'\n\trouge1:{rg1_res}'
        result += f'\n\trouge2:{rg2_res}'
        result += f'\n\trougeL:{rgl_res}'
        if tst.ident not in res_dict:
            res_dict[tst.ident] = {}
        if base_engi not in res_dict[tst.ident]:
            res_dict[tst.ident][base_engi] = {}
        if tp.__name__ not in res_dict[tst.ident][base_engi]:
            res_dict[tst.ident][base_engi][tp.__name__] = {}
        res_dict[tst.ident][base_engi][tp.__name__] = {"em":avg(exact_match_cnt), "gpt":avg(gpt_judge_true_cnt),"r1":avg(rouge_scores), 'rg1': rg1_res, 'rg2': rg2_res, 'rgl': rgl_res, 'bleu4':bleu4_res}

    dual_write_line(result)
    return actual_cnt, expect_cnt


def static_var(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_var(path="get_stat.txt")
def dual_write_line(line):
    with open(dual_write_line.path, 'a+') as fd:
        fd.write(str(line)+'\n')


def stat(include_label):
    types = [MYtree.Tree, MYSQL.SQL, MYjson.JSON, MYyaml.YAML, MYxml.XML, MYmarkdown.Markdown, MYorg.ORG, MYlatex.LaTeX]
    tp_2_id = {v.__name__: i for (i, v) in enumerate(types)}
    tsts = [testee.spark_api, testee.minimax_api, testee.baidu_api, testee.gpt4_api, testee.gpt4_base_api]
    root_dir_options = ['benchmarkMiniTest','2ShotbenchmarkMiniTest','RuleHintedbenchmarkMiniTest:noappend','RuleHintedbenchmarkMiniTest:append', '1ShotbenchmarkMiniTest']
    for root_dir in root_dir_options:
        res_dict = {}
        actual_cnt, expect_cnt = 0, 0
        root_dir_escaped = root_dir.replace(':', '_')
        dual_write_line.path = f'stats/{root_dir_escaped}_stat.txt'

        open(dual_write_line.path, 'w').close() # clear

        liner = '=' * 42
        dual_write_line(f"{liner}{root_dir}{liner}")
        if ':' in root_dir:
            root_dir, append_method = root_dir.split(':')
        else:
            append_method = None
        for tp in types:
            for tst in tsts:
                dac, dec = get_stats(root_dir, tp, tst, False, append_method, include_label, res_dict)
                actual_cnt += dac
                expect_cnt += dec
                dac, dec = get_stats(root_dir, tp, tst, True, append_method, include_label, res_dict)
                actual_cnt += dac
                expect_cnt += dec
        dual_write_line(f"overall:({actual_cnt}/{expect_cnt})")
        dual_write_line(tp_2_id)
        for tst in res_dict:
            for prpt in res_dict[tst]:
                em, gpt, r1 = [0] * 9, [0] * 9, [0] * 9
                for tp in res_dict[tst][prpt]:
                    em[tp_2_id[tp]] = res_dict[tst][prpt][tp]['em']
                    gpt[tp_2_id[tp]] = res_dict[tst][prpt][tp]['gpt']
                    r1[tp_2_id[tp]] = res_dict[tst][prpt][tp]['r1']
                dual_write_line("\t".join([tst, prpt]))
                dual_write_line("\t".join(["em:", "& ".join(["{:.3f}".format(n) for n in em])]))
                dual_write_line("\t".join(["gpt:", "& ".join(["{:.3f}".format(n) for n in gpt])]))
                dual_write_line("\t".join(["r1:", "& ".join(["{:.3f}".format(n) for n in r1])]))
        def get_input_len():
            filenames = glob.glob(f'./{root_dir}/**/**/**/*.json')
            input_lens = []
            for filename in filenames:
                if skip(include_label, filename): continue
                with open(filename, 'r') as f:
                    input_lens.append(len(json.load(f)['input']))
            return sum(input_lens)/len(input_lens)
        dual_write_line(f"Avg input file len is {get_input_len()} bytes")


if __name__ == '__main__':
    stat(['hard'])
    # pred = ["She was a boy from a long time ago in ancient Greek", "A Cat sit on a mattras"]
    # gold = ["False", "Mattras below a cat"]
    # print(eval_metric(pred, gold))
