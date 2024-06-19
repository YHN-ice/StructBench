# Overview

This repo is an archive for

1. [StructBench](./Dataset) Dataset 
2. Code used to [generate the dataset](./LLMStructure/datagen.py)
    
    
# Repo Structure

```shell
.
├── APITestOutput
│   ├── 1Shotoutput_load #(API test result for 1ShotbenchmarkMiniTest)
│   ├── 2Shotoutput_load #(API test result for 2ShotbenchmarkMiniTest)
│   ├── RuleHintedoutput_load #(API test result for RuleHintedbenchmarkMiniTest)
│   ├── Ruleappend_hintoutput_load #(API test result for RuleHintedbenchmarkMiniTest, but with RuleHint appended)
│   ├── output_load #(API test result for benchmarkMiniTest)
│   └── stats #(test result produced by get_stat.py)
├── Dataset
│   ├── 1ShotbenchmarkMiniTest #(StructBench Dataset 1 Shot Version, MiniTest split, Train/Test split not available)
│   ├── 2ShotbenchmarkMiniTest #(StructBench Dataset 2 Shot Version, MiniTest split)
│   ├── 2ShotbenchmarkTest #(StructBench Dataset 2 Shot Version, Test split)
│   ├── 2ShotbenchmarkTrain #(StructBench Dataset 2 Shot Version, Train split)
│   ├── RuleHintedbenchmarkMiniTest #(RuleHint variant of StructBench Dataset, MiniTest split)
│   ├── RuleHintedbenchmarkTest #(RuleHint variant of StructBench Dataset, Test split)
│   ├── RuleHintedbenchmarkTrain #(RuleHint variant of StructBench Dataset, Train split)
│   ├── benchmarkMiniTest #(Base StructBench Dataset, MiniTest split)
│   ├── benchmarkTest #(Base StructBench Dataset, Test split)
│   └── benchmarkTrain #(Base StructBench Dataset, Train split)
├── LICENSE
├── LLMStructure
│   ├── Chat.py #(Wrappers for APIs)
│   ├── MYSQL.py #(generator for Tabular samples)
│   ├── MYjson.py #(generator for JSON samples)
│   ├── MYlatex.py #(generator for LaTeX samples)
│   ├── MYmarkdown.py #(generator for Markdown samples)
│   ├── MYorg.py #(generator for ORG samples)
│   ├── MYtree.py #(generator for Tree samples, also imported by other MYxxx.py file as a template)
│   ├── MYxml.py #(generator for XML samples)
│   ├── MYyaml.py #(generator for YAML samples)
│   ├── Person.py #(data Object for Tabular data)
│   ├── RuleHint.txt #(list of manually written RuleHint)
│   ├── api.py #(main code for making query)
│   ├── datagen.py #(dataset generator entry)
│   ├── get_stat.py #(collect test result)
│   ├── gevalft.py #(evaluate finetune model inference output)
│   ├── name.py #(helper class for manage identifiers filled inside input texts template)
│   ├── requirements.txt
│   ├── rule_hint.py #(script for parsing RuleHint.txt)
│   ├── structure.py #(common stub for all structure rich text generator)
│   └── testee.py #(main code for manage api)
├── LLaMA-Factory
│   ├── finetune.sh #(finetune invocation)
│   └── minitest.sh #(inference invocation)
└── README.md

21 directories, 24 files
```

[^1]: Only provided the script to invoke [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main?tab=readme-ov-file), please refer to their documentation for details.
[^2]: Model used for finetuning is [openlm-research/open_llama_7b](https://huggingface.co/openlm-research/open_llama_7b)

