import argparse
import json
import os
import re
import time

from simplex_sdk import SimplexClient

from car_rewrite_model.model import CarRewriteSynonymsReplace, CarRewriteBaseKeywordsNewProcess


# phrases_before_colon_file='/data/share/liuchang/car_rewirte_compare/remove_words'
# with open(phrases_before_colon_file, 'r', encoding='utf8') as f:
#     phrases = f.readlines()
#     phrases = [n.strip() for n in phrases]
# colon_phrase_pattern = r'\b(?:{})[ \t]*[:：]'.format('|'.join(phrases))

def change_position_by_colon(txt,colon_phrase_pattern):
    splited=re.split(colon_phrase_pattern,txt)
    splited=[s for s in splited if s.strip()]

    if len(splited)<2:
        return txt

    return ' '.join([splited[-1]]+splited[:-1])

def get_result(txt_list, domain_list, model,batch_size=32):
    inputs = []

    content_and_domain=zip(txt_list, domain_list)
    for i, data in enumerate(content_and_domain):
        txt,domain=data
        d = {'id': i, 'content': txt.strip(), 'domain': domain}
        inputs.append(d)

    result = []

    call_time = (len(inputs) + batch_size - 1) // batch_size
    for i in range(call_time):
        print('call num: {}'.format(i))
        result.extend(model.predict(inputs[i * batch_size:(i + 1) * batch_size]))
    return result


def synonyms_rewrite_model(**kwargs):
    pos_model = '/data/share/liuchang/car_articles/pos.model'
    phrases_before_colon = '/data/share/liuchang/car_rewirte_compare/remove_words'
    forbidden_output_file='/data/share/liuchang/train_bert_lm/forbidden_output'
    antonym_dict_file='/data/share/liuchang/train_bert_lm/merged_antonym_dict.json'


    # pos_model = 'oss://modelzoo/dev/pos_model_for_car_rewrite/pos.model'
    # phrases_before_colon = 'oss://modelzoo/dev/car_rewrites/phrases_before_colon'

    model = CarRewriteSynonymsReplace(pos_model_path=pos_model, all_mask=False,
                                      phrases_before_colon=phrases_before_colon,
                                      antonym_dict_file=antonym_dict_file,
                                      forbidden_output_file=forbidden_output_file,
                                      **kwargs)

    return model


def keywords_rewirte_model(**kwargs):
    data_dir = '/data/share/liuchang/car_rewirte_compare/keywords'
    vocab_path = os.path.join(data_dir, 'vocab.txt')
    keywords_path = os.path.join(data_dir, 'keywords.txt')
    idf_json_file = os.path.join(data_dir, 'idf_scores_dict.json')
    high_freq_token_file = os.path.join(data_dir, 'high_freq_words.txt')

    model = CarRewriteBaseKeywordsNewProcess(vocab_path=vocab_path,
                                             keywords_path=keywords_path,
                                             idf_json_path=idf_json_file,
                                             high_freq_token_path=high_freq_token_file,
                                             **kwargs)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model",
                        choices=["keywords", "synonym", "mass", 'mask'],
                        help="Run type.")
    parser.add_argument("--change_position",
                        default=True)
    parser.add_argument("--input",
                         default='/data/share/liuchang/car_rewirte_compare/synonyms/local_test/raw_10_80_p3_p10')
    args = parser.parse_args()
    print(args)

    input_file = args.input
    # input_file = '/data/share/liuchang/car_rewirte_compare/laosiji_examples_5000_txt'

    with open(input_file, 'r', encoding='utf8') as f:
        txts = f.readlines()
        txts = [n.strip() for n in txts]

    domain_file = '/data/share/liuchang/car_rewirte_compare/domain_5000'
    with open(domain_file, 'r', encoding='utf8') as f:
        domain_list = f.readlines()
        domain_list = [n.strip() for n in domain_list]
        domain_list=domain_list+['']*(len(txts)-len(domain_list))

    # DATA_SIZE = 3
    # STRS=['驾驶性能超好，还真别说，会开车的，开着瑞虎7操控感觉肯定也不错，高速路行驶：不费劲。']*DATA_SIZE

    # txts = txts[:10]

    name_to_model = {
        'keywords': keywords_rewirte_model,
        'synonym': synonyms_rewrite_model
    }

    model = name_to_model[args.model](change_position=args.change_position,
                                      w2v_path='/data/share/liuchang/car_articles/w2v/20epoch_cut_spm/w2v.model')
    # model=SimplexClient('CarRewriteBaseKeywords')


    batch_size = max(64,len(txts)//100)
    result = get_result(txts, domain_list, model, batch_size)

    out_dir = '/data/share/liuchang/car_rewirte_compare/synonyms/local_test'
    for d in result:
        for word in d['rewrite_tokens']:
            d['rewrite_tokens'][word]=','.join(d['rewrite_tokens'][word])

    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    # print(json_str)

    suffix = 'w2v_rewrites'
    basename=os.path.basename(input_file)
    # time_str = time.strftime("%y-%m-%d-%H-%M-%S")
    # json_output=os.path.join(out_dir, 'rewrites_{}.json'.format(time_str))
    json_output = os.path.join(out_dir,basename + '_{}.json'.format(suffix))
    with open(json_output, 'w', encoding='utf8') as f:
        f.write(json_str)

    # compare_output = os.path.join(out_dir, 'rewrites_compare_{}'.format(time_str))
    compare_output = os.path.join(out_dir,basename + '_{}'.format(suffix))
    with open(compare_output, 'w',
              encoding='utf8') as f:
        for raw, n in zip(txts, result):
            f.write('content: ' + raw + '\n')
            f.write('rewrite: ' + n['rewrite_content'] + '\n\n')
