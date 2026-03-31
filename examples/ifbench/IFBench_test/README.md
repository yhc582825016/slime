---
dataset_info:
  features:
  - name: key
    dtype: string
  - name: prompt
    dtype: string
  - name: instruction_id_list
    sequence: string
  - name: kwargs
    list:
    - name: N
      dtype: float64
    - name: capital_frequency
      dtype: 'null'
    - name: capital_relation
      dtype: 'null'
    - name: end_phrase
      dtype: 'null'
    - name: first_word
      dtype: 'null'
    - name: forbidden_words
      dtype: 'null'
    - name: frequency
      dtype: 'null'
    - name: keyword
      dtype: string
    - name: keyword1
      dtype: string
    - name: keyword2
      dtype: string
    - name: keyword3
      dtype: string
    - name: keyword4
      dtype: string
    - name: keyword5
      dtype: string
    - name: keywords
      dtype: 'null'
    - name: language
      dtype: 'null'
    - name: let_frequency
      dtype: 'null'
    - name: let_relation
      dtype: 'null'
    - name: letter
      dtype: 'null'
    - name: m
      dtype: int64
    - name: max_words
      dtype: float64
    - name: min_words
      dtype: float64
    - name: n
      dtype: int64
    - name: n_end
      dtype: int64
    - name: n_start
      dtype: int64
    - name: nth_paragraph
      dtype: 'null'
    - name: num_bullets
      dtype: 'null'
    - name: num_highlights
      dtype: 'null'
    - name: num_paragraphs
      dtype: 'null'
    - name: num_placeholders
      dtype: 'null'
    - name: num_sections
      dtype: 'null'
    - name: num_sentences
      dtype: 'null'
    - name: num_words
      dtype: 'null'
    - name: options
      dtype: string
    - name: percentage
      dtype: float64
    - name: postscript_marker
      dtype: 'null'
    - name: prompt_to_repeat
      dtype: string
    - name: reference_text
      dtype: string
    - name: relation
      dtype: 'null'
    - name: section_spliter
      dtype: 'null'
    - name: sep
      dtype: string
    - name: small_n
      dtype: float64
    - name: word
      dtype: string
  splits:
  - name: train
    num_bytes: 162966
    num_examples: 300
  download_size: 94446
  dataset_size: 162966
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

## License

This dataset is licensed under ODC-BY-1.0. It is intended for research and educational use in accordance with Ai2's [Responsible Use Guidelines](https://allenai.org/responsible-use). This dataset includes output data generated from third party models that are subject to separate terms governing their use. 

## Citation
Please cite:
```
@misc{pyatkin2025generalizing,
   title={Generalizing Verifiable Instruction Following}, 
   author={Valentina Pyatkin and Saumya Malik and Victoria Graf and Hamish Ivison and Shengyi Huang and Pradeep Dasigi and Nathan Lambert and Hannaneh Hajishirzi},
   year={2025},
   eprint={TODO},
   archivePrefix={arXiv},
   primaryClass={cs.CL}
}
```
