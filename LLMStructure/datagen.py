from datetime import datetime
from time import sleep
import glob

import MYSQL, MYmarkdown, MYtree, MYxml, MYjson, MYyaml, MYlatex, json, MYorg
from api import open_wrapper

type_set = [MYSQL.SQL, MYjson.JSON, MYxml.XML, MYtree.Tree, MYyaml.YAML, MYmarkdown.Markdown, MYlatex.LaTeX, MYorg.ORG]


def write_data_for_sample(sample, typename, level, i, few_shot, with_rule_hint=False): # few_shot is just list of other samples
    for idx,((q, a, input_thin), label_list) in enumerate(zip(sample['QA_to_save'], sample['labels'])):
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S.%f")
        to_dump = {'Q': q.strip(), 'A': a.strip(), 'label': ([typename, level] if level is not None else [typename]) + label_list,
                   'input': input_thin, 'timestamp': timestamp,
                   'input_hash': hash(input_thin)}

        if few_shot: # if len(few_shot) > 0, add examples to data
            key_dict = ['Q', 'A', 'input']
            def build_sub_dict(q_a_input):
                return {key_dict[i]: v for i, v in enumerate(q_a_input)}
            to_dump['examples'] = [build_sub_dict(eg['QA_to_save'][idx]) for eg in few_shot]
        if with_rule_hint:
            from rule_hint import fetch_rule_hint
            to_dump['rule_hint'] = fetch_rule_hint(typename, idx)

        # q1 and q3 could have exact same labels,
        # thus resulting in same file name, using Q hash to name file instead
        filename = f'{abs(hash(q)) % 42}.{timestamp}.{i}'
        label_as_directory = '_'.join(label_list)

        if few_shot:
            root_name = f'{len(few_shot)}Shotbenchmark'
        elif with_rule_hint:
            root_name = 'RuleHintedbenchmarkTest'
        else: 
            root_name = 'benchmark'
        write_path = f'{root_name}/{typename}/{level}/{label_as_directory}/{filename}_{idx}.json'
        with open_wrapper( write_path, 'a+', encoding="utf-8") as f:
            print(f'i={i}, path={write_path}')
            json.dump(to_dump, f, indent=4, ensure_ascii=False)


def procedure_gen(few_shot=0, with_rule_hint=False):
    i = 0
    while i < 4: # Train-80, Test-20, MiniTest-4
        for Type in type_set:
            for level in ['simple', 'hard']:
                sample = Type.gen_QAs(level == 'simple')
                shot_example = list(Type.gen_QAs(level == 'simple') for _ in range(few_shot))
                write_data_for_sample(sample, Type.__name__, level, i, shot_example, with_rule_hint=with_rule_hint)
        i += 1


if __name__ == '__main__':
    procedure_gen(few_shot=2)
