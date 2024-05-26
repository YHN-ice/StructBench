#!/usr/bin/env python3
import glob, json, os, time, openai
from rouge_score import rouge_scorer

import Chat
from time import sleep

import testee

openai.api_key = os.getenv("OPENAI_API_KEY")

wait_duration = 60
# RULE_HINT_METHOD = 'append'
RULE_HINT_METHOD = 'noappend'
def get_output_filename(filename, root_output, rule_hint_method=RULE_HINT_METHOD):
    if rule_hint_method == 'append':
        output_filename = filename.replace("Hintedbenchmark", "append_hint" + root_output)
    else:
        output_filename = filename.replace("benchmark", root_output)
    return output_filename


default_model = "openaigpt4"  # or gpt-35-xx or custom
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

def MYcompletion(messages=None, model=default_model):
    if messages == None:
        messages = [{"role": "system", "content": "you are Shakespear now"},
                    {"role": "user", "content": "To live or die, this is a"}]
    suc = False
    response = None
    truncate = False
    try:
        if model == "gpt-4-4":
            response = Chat.api_wrapper_gpt44(messages=messages)
        elif model == "custom":
            response = Chat.api_wrapper_custom(messages=messages)
        elif model == "gpt4xhs":
            response = Chat.api_wrapper_xhs(message_or_prompt=messages)
        elif model == "openaigpt4":
            response = Chat.api_wrapper_opanai(message_or_prompt=messages)
        else:
            response = Chat.api_wrapper(messages=messages)
        suc = True
    except openai.error.Timeout as e:
        # Handle timeout error, e.g. retry or log
        print(f"OpenAI API request timed out: {e}")
        pass
    except openai.error.APIError as e:
        # Handle API error, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        pass
    except openai.error.APIConnectionError as e:
        # Handle connection error, e.g. check network or log
        print(f"OpenAI API request failed to connect: {e}")
        pass
    except openai.error.InvalidRequestError as e:
        # Handle invalid request error, e.g. validate parameters or log
        print(f"OpenAI API request was invalid: {e}")
        if 'Please reduce the length of the messages' in str(e):
            truncate = True
        pass
    except openai.error.AuthenticationError as e:
        # Handle authentication error, e.g. check credentials or log
        print(f"OpenAI API request was not authorized: {e}")
        pass
    except openai.error.PermissionError as e:
        # Handle permission error, e.g. check scope or log
        print(f"OpenAI API request was not permitted: {e}")
        pass
    except openai.error.RateLimitError as e:
        # Handle rate limit error, e.g. wait or log
        print(f"OpenAI API request exceeded rate limit: {e}")
        pass
    except openai.error.ServiceUnavailableError as e:
        # Handle service unavailable error, e.g. wait or log
        print(f"OpenAI service unavailable: {e}")
        pass
    except Exception as e:
        print(f"api errors other than openai: {e}")
    return suc, response, truncate


def open_wrapper(file_path, mode='r', *args, **kwargs):
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return open(file_path, mode, *args, **kwargs)


def evaluate(pred, target):
    return target.strip() in pred


def rouge1_f1_eval(pred, ref):
    return scorer.score(ref, pred)['rouge1'][2]


def gpt_judge(pred, ref, question):
    system = "You are a judge that will decide whether a answer is semantically right based on the question and ground truth. Output should be one of 'True' of 'False' and nothing else."
    prompt = f'Question: {question}\nAnswer: {pred}\nGroundTruth: {ref}'
    history = query_fetch_wrapper(system, prompt)
    if "content" not in history[-1]:
        judge_res = ""
    else:
        judge_res = history[-1]['content']
    if "True" in judge_res:
        return "True", history
    elif "False" in judge_res:
        return "False", history
    else:
        print("Judge result contains none of True of False, invalid judge.")
        return "N/A: invalid judge", history


def query_one_sample_from_file_with_testee(data, testee_api: testee.API):
    system = data['system']
    history = [system]

    q, a = data['Q'], data['A']
    prompt = q
    history.append(q)

    response = testee_api.query((prompt, system))
    if not response:
        with open_wrapper("malfunctioning.txt", 'a') as f:
            f.write(f"{data['src_file_path']}\n")
        print(response)
        return [], None, None, None
    history.append(response)

    gold = a
    thin_q = data['data']['Q']
    judge_res, judge_hist = gpt_judge(response, a, thin_q)
    judge_histories = judge_hist
    hist_score = (evaluate(response, a), rouge1_f1_eval(response, a), judge_res)
    return history, hist_score, gold, judge_histories


# TODO begin
def get_system(Type):
    if Type == MYSQL.SQL:
        return f'you are a datatables parser, you are required to answer questions pertaining to the given datatables, the rows with same primeKey refer to the same person.'
    return f'you are a {Type.__name__} file parser, you are required to answer questions pertaining to the given {Type.__name__} file.'


def get_instr(question):
    return f'Question: {question}\n'


def get_input(inputdata):
    return f'Input: \n{inputdata}\n'


def get_hint(system, Q, baseline=False):
    if baseline:
        return '', []
    hint, history = query_for_hint(system, Q) # Hint Elicitation
    return f'In your process of tackle the problem, there are some potential difficulties and latent obstacles for you to notice:\n\n{hint}', history


def query_fetch_wrapper(system, prompt):
    suc, fail_cnt = False, 0
    history = [{"role": "system",
                "content": system}, {"role": "user", "content": prompt}]
    while not suc:
        suc, completion, trunc = MYcompletion(messages=history)
        if not suc:
            if trunc:
                # return "N/A: messages too long", []
                if len(history) == 2:
                    # try trunc
                    history = history[1:]
                else:
                    print("The messages is too large. Skipping this one")
                    # give up
                    history.append({"role": "user", "content": "N/A: messages too long"})
                    return history
            else:
                if fail_cnt > 10:
                    print("The messages exceed attempt counts. Skipping this one")
                    history.append({"role": "user", "content": "N/A: failed too many times"})
                    return history
                fail_cnt += 1
                print(f"API fail, wait 60*{fail_cnt} second...")
                sleep(wait_duration * fail_cnt)
    print("-response fetched(fetch wrapper).")
    if isinstance(completion, dict):
        history.append(completion['choices'][0]['message'])
    else:
        history.append(completion.choices[0].message)
    return history


def query_for_hint(system, Q):
    prompt = f"Question: {Q}\n\nBefore you see the input file and start to answer, let's come up with some potential difficulties and obstacles that might hinder your process of giving the correct answer."
    history = query_fetch_wrapper(system, prompt)
    hint_lines = history[-1]['content'].split('\n')
    hint_lines_filtered = list(filter(lambda x: not x.startswith("Now"), hint_lines))
    return "\n".join(hint_lines_filtered), history


# TODO end

def load_for_a_type(Type, filename, baseline=False):
    with open(filename, 'r') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"error reading file {filename}")
            print(f"error details: {e}")
    if 'rule_hint' in data:
        if RULE_HINT_METHOD == 'append':
            data['Q'] += f'\n{data["rule_hint"]}'
        else:
            from rule_hint import transform
            data = transform(data)
    system = get_system(Type)
    instr = get_instr(data["Q"])
    _input = get_input(data["input"])
    hint, hint_hist = get_hint(system, data["Q"], baseline=baseline)
    def build_example(Q,A,input):
        return f'{get_instr(Q)}\n{get_input(input)}\nAnswer:\n{A}\n'
    few_shot = '\n'.join([build_example(**qai) for qai in data['examples']]) + '\n' if 'examples' in data else ''
    return {'system': system, 'Q': f'{few_shot}{instr}\n{_input}\n{hint}\nAnswer:\n', 'A': data['A'],
            'labels': [data['label']], 'data': data, 'hint_histories': hint_hist, 'src_file_path': filename}


def only_support_load():
    raise RuntimeError("only loading dataset from disk is supported now, generating in the fly is deprecated")


def for_a_type(Type, sample_size=24, load=False, baseline=True, model=default_model, api_testee=testee.gpt3_api,
               starting_idx=0):
    filenames = None

    if load:
        # change me
        filenames = glob.glob(f'./3ShotbenchmarkMiniTest/{Type.__name__}/**/**/*.json')
        sample_size = len(filenames)
    else:
        only_support_load()
    for i in range(starting_idx, sample_size):
        is_simple = i < sample_size // 2
        cate = 'simple' if is_simple else 'complex'
        cate = 'load' if load else cate
        root_output = (f'output_load/{api_testee.ident}' if load else 'output_gen') + (
            "_engi_" if load and not baseline else '_base_')
        output_filename = get_output_filename(filenames[i], root_output)
        json_dict = {}

        if os.path.exists(output_filename):
            continue
        print(
            f'+i={i} time={time.asctime()} simple={is_simple}, type={Type.__name__} cate={cate}, output={output_filename}, src={filenames[i]}')
        if load:
            json_dict['benchmark_filename'] = filenames[i]
            load_data = load_for_a_type(Type, filenames[i], baseline=baseline)
            hist, scores, gold, judge_histories = query_one_sample_from_file_with_testee(load_data, api_testee)
            # request failed to complete due to API error
            if not hist and not scores and not gold and not judge_histories:
                print(f"!!! skipping {filenames[i]}")
                continue
            if not baseline:
                json_dict['hint_query_history'] = load_data['hint_histories']
        else:
            only_support_load()
        print(f'-i={i} time={time.asctime()} simple={is_simple}, type={Type.__name__}')

        json_dict.update(
            {"hist": hist, 'scores': scores, 'gold': gold, 'judge_histories': judge_histories, "judge_model": model})

        with open_wrapper(output_filename, 'a+',
                          encoding="utf-8") as file:
            json.dump(json_dict, file, indent=4, ensure_ascii=False)
        if (i + 1) % (max(sample_size // 3, 1)) == 0:
            sleep(wait_duration * 2)
            print('ZzZzZzZz...')


def for_a_type_wrapper(api_testee):
    types = [MYSQL.SQL, MYjson.JSON, MYyaml.YAML, MYtree.Tree, MYxml.XML, MYorg.ORG, MYlatex.LaTeX, MYmarkdown.Markdown,
             MYPython.PYTHON]
    for tp in types:
        for_a_type(tp, load=True, baseline=True, api_testee=api_testee)
        print(f"={tp.__name__} base eval {api_testee.ident} done.=")
        for_a_type(tp, load=True, baseline=False, api_testee=api_testee)
        print(f"={tp.__name__} engi eval {api_testee.ident} done.=")


def looped_parallel():
    raise NotImplementedError


def looped():
    # apis = [ testee.baidu_api, testee.gpt4_api, testee.spark_api, testee.gpt4_base_api, testee.minimax_api]#TODO, fix api: insufficient balance. testee.minimax_api
    apis = [ testee.baidu_api, testee.minimax_api] #TODO, fix api: insufficient token. testee.spark, expired xhs*
    types = [MYSQL.SQL, MYjson.JSON, MYyaml.YAML, MYtree.Tree, MYxml.XML, MYorg.ORG, MYlatex.LaTeX, MYmarkdown.Markdown]
    for tp in types:
        for mapi in apis:
            # [['TextRetrieval'], ['TextRetrieval'], ['Syntax']]
            # for_a_type(MYxml.XML, load=True, baseline=True, api_testee=mapi, starting_idx=28 if mapi.ident=='gpt3davinci' else 0)
            for_a_type(tp, load=True, baseline=True, api_testee=mapi)
            print(f"={tp.__name__} base eval done.=")
            for_a_type(tp, load=True, baseline=False, api_testee=mapi)  # the first half was complex, latter is simple
            print(f"={tp.__name__} engi eval done.=")


def unrolled():
    raise NotImplementedError


if __name__ == '__main__':
    # looped_parallel()
    # unrolled()
    looped()
