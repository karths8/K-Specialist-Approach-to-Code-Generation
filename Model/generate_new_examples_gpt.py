import json
from openai import OpenAI
import random
import argparse


parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--part', default=1, type=int, help="which part")
parser.add_argument('--total', default=5, type=int, help="total number of parts")
args = parser.parse_args()


def split_list(input_list, num_parts):
    if num_parts <= 0:
        raise ValueError("Number of parts should be greater than zero")
    
    avg_part_length = len(input_list) // num_parts
    remainder = len(input_list) % num_parts

    result = []
    start = 0

    for i in range(num_parts):
        part_length = avg_part_length + 1 if i < remainder else avg_part_length
        end = start + part_length
        result.append(input_list[start:end])
        start = end

    return result


def sample_keywords(length, keyword_list):
    words = list(keyword_list.keys())
    weights = list(keyword_list.values())
    # sample = list(set(random.choices(words, weights=weights, k=length)))
    sample = list(set(random.choices(words, k=length)))
    while len(sample) < length:
        # additional_samples = random.choices(words, weights=weights, k=length-len(sample))
        additional_samples = random.choices(words, k=length-len(sample))
        sample.extend(set(additional_samples))
    return sample

def build_example_template(data_examples):
    examples = []
    for i in data_examples:
        examples.append({"role":"user", "content":i["user"]})
        examples.append({"role":"assistant", "content":i["assistant"]})
    return examples

def make_example(example, n,keyword_freq_dict, test=False):
    # user_prompt = 'It is not necessary to use all the keywords in the list given to generate the question, code and asserts. You may only use a few of them if you want.'
    test_prompt = 'Give one more example based on a similar Keyword list, It is not necessary to use all these keywords: '
    user_prompt = 'Give one example based on this Keyword list. It is not necessary to use all these keywords: '
    keyword_length = random.randint(2, 5)
    extra_keywords = random.sample(list(keyword_freq_dict.keys()), keyword_length)
    # print(example['keywords'])
    # print(extra_keywords)
    # print(set(example["keywords"]+extra_keywords))
    # print(list(set(example["keywords"]+extra_keywords)))
    # print(random.shuffle(list(set(example["keywords"]+extra_keywords))))
    keywords = list(set(example["keywords"]+extra_keywords))
    random.shuffle(keywords)
    keywords = str(keywords)
    if test:
        return {'user':f'{test_prompt}\n\n[Keywords {n}]\n\n{keywords}\n\n[/Keywords {n}]'}, keywords
    else:
        assert_str = example['asserts']
        return {'user':f"{user_prompt}\n\n[Keywords {n}]\n\n{keywords}\n\n[/Keywords {n}]", 'assistant':f'[Question {n}]\n\n{example["question"]}\n\n[/Question {n}]\n\n[Code {n}]\n\n{example["code"]}\n\n[/Code {n}]\n\n[Asserts {n}]\n\n{assert_str}\n\n[/Asserts {n}]'}, keywords

def main():
    part = args.part
    total_parts = args.total
    
    with open('/workspace/CS762_Project/Data_files/keywords/enriched_xlcost.json', 'r') as json_file:
        data = json.load(json_file)
    data = split_list(data, total_parts)[part-1]
    print(f'Lenght of data is {len(data)}')
    client = OpenAI(
        api_key='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'
    )

    keyword_freq_dict = {}
    for i in data:
        for j in i['keywords']:
            if j not in keyword_freq_dict:
                keyword_freq_dict[j] = 0
            keyword_freq_dict[j]+=1
    
    system_prompt = """
    You are an expert programmer who can understand intricate details about computer science programming questions, code, and assert statements. Given a question you can extract keywords from them and given a list of keywords, you have the ability to frame meaningful programming questions that are associated with those keywords!
    
    Given a list of keywords under [Keywords] ... [/Keywords], try to use as many of them as possible to frame a meaningful programming question under [Question] ... [/Question], the code under [Code] ... [/Code] to solve that question, and a few assert statements under [Asserts] ... [/Asserts] to test that code. You do not need to use all the keywords in the list to formulate a question.
    """
    src = 'xlcost-diverse-instruct'
    results = []
    counter=0
    for idx1 in range(len(data)):
        example_samples = [data[idx1]]
        print(f'Sample {idx1} started!')
        data_examples = []
        for idx, ex in enumerate(example_samples):
            t, fs_keywords = make_example(ex,idx+1,keyword_freq_dict)
            data_examples.append(t)
        completion_dicts = build_example_template(data_examples)
        # print(json.dumps(completion_dicts, indent=2))
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(completion_dicts)
        # keyword_length = random.randint(7, 12)
        # test_keywords = {'keywords': sample_keywords(keyword_length, filtered_all_keywords)}
        example = example_samples[0]
        test_t, test_keywords = make_example(example, idx+2,keyword_freq_dict, test=True)
        test_ex = [{'role':'user','content':test_t['user']}]
        messages.extend(test_ex)
        # print(json.dumps(messages, indent=2))
        print(f'Prompt for Sample {idx1}\n\n')
        print('\n'.join(c['content'] for c in messages))
        model_name = 'gpt-3.5-turbo-1106'
        sampled_temp = round(random.uniform(0.7, 0.95), 3)
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=sampled_temp
        )
        print(f'Sample {idx1} generated!')
        print(f'Completion for Sample {idx1}')
        print(response.choices[0].message.content)
        results.append({'few_shot_keywords':fs_keywords,'test_keywords':test_keywords,'prompt_tokens':response.usage.prompt_tokens,'total_tokens':response.usage.total_tokens,'completion_tokens':response.usage.completion_tokens,'completion':response.choices[0].message.content,'model_name':model_name,'source':src, 'temperature': sampled_temp})
        with open(f'xlcost-diverse-instruct_same_keyword_list_Dec14_part_{part}.json', 'w') as json_file:
            json.dump(results, json_file)

if __name__=='__main__':
    main()