import json
import os
import time

from car_rewrite_model.model import CarRewriteSynonymsReplace
from simplex_sdk import SimplexClient

BATCH_SIZE=160

def get_result(values,model):
    inputs=[]
    for i,n in enumerate(values):
        d={'id':i,'content':n.strip(),'domain':''}
        inputs.append(d)

    result=[]

    call_time=(len(inputs)+BATCH_SIZE-1)//BATCH_SIZE
    for i in range(call_time):
    #     print('call num: {}'.format(i))
        result.extend(model.predict(inputs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]))
    return result

if __name__ == '__main__':

    pos_model = '/data/share/liuchang/car_articles/pos.model'
    phrases_before_colon='/data/share/liuchang/car_rewirte_compare/remove_words'

    # pos_model = 'oss://modelzoo/dev/pos_model_for_car_rewrite/pos.model'
    # phrases_before_colon = 'oss://modelzoo/dev/car_rewrites/phrases_before_colon'

    # model = CarRewriteSynonymsReplace(pos_model_path=pos_model, all_mask=False,
    #                                   phrases_before_colon=phrases_before_colon)

    model=SimplexClient('')
    data_size = 3

    # strs=['驾驶性能超好，还真别说，会开车的，开着瑞虎7操控感觉肯定也不错，高速路行驶：不费劲。']*data_size
    input_file='/data/share/liuchang/car_rewirte_compare/laosiji_examples_5000_txt'
    with open(input_file, 'r', encoding='utf8') as f:
        txts = f.readlines()
        txts = [ n.strip() for n in txts ]

    txts=txts[10]
    result = get_result(txts,model)

    out_dir='/data/share/liuchang/car_rewirte_compare/synonyms/local_test'
    json_str=json.dumps(result,ensure_ascii=False,indent=2)
    print(json_str)

    time_str = time.strftime("%y-%m-%d-%H-%M-%S")
    with open(os.path.join(out_dir,'rewrites_{}.json'.format(time_str)), 'w', encoding='utf8') as f:
        f.write(json_str)

    with open(os.path.join(out_dir,'rewrites_compare_{}'.format(time_str)), 'w',
              encoding='utf8') as f:
        for raw, n in zip(txts, result):
            f.write('content: ' + raw + '\n')
            f.write('rewrite: ' + n['rewrite_content'] + '\n\n')