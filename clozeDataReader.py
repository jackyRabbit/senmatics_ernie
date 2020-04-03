# 主要用来加载完形填空的测试数据

# 加载单词的数据
import paddle.fluid as fluid
from ernie_masked_lm import ErnieModel

from utils.init import init_checkpoint,init_pretraining_params
from ernieDataReader import ErnieDataReader,process
import json
import numpy as np
from tokenization import FullTokenizer

# 建立一个模型
def create_model(pyreader_name,ernie_config,cloze_config):
    # 这里之前要定义一个cloze_config
    max_seq_len = cloze_config['max_seq_len']
    # 数据输入模型的格式
    src_ids = fluid.layers.data(name='src_ids', shape=[-1, max_seq_len, 1], dtype='int64')
    pos_ids = fluid.layers.data(name='pos_ids', shape=[-1, max_seq_len, 1], dtype='int64')
    sent_ids = fluid.layers.data(name='sent_ids', shape=[-1, max_seq_len, 1], dtype='int64')
    input_mask = fluid.layers.data(name='input_mask', shape=[-1, max_seq_len, 1], dtype='float32')
    mask_pos = fluid.layers.data(name='mask_pos', shape=[-1, 1], dtype='int64')
    mask_label = fluid.layers.data(name='mask_label', shape=[-1, 1], dtype='int64')

    # 定义一个reader从这儿往模型中传入数据
    # 改进，这儿我们不用其自带的reader，因此不定义数据容易 # data1


    # 因此需要数据的读入需要外部提供一个实际的输入接口

    # 定义数据读入的reader,这里我们单个输入，不转为batch
    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        config=ernie_config,
        weight_sharing=True,
        use_fp16=False)
    mean_mask_lm_loss = ernie.get_pretraining_output(mask_label,mask_pos)# 这个实际上是一个躯壳（模型），真正要里面的值的时候，需要run去抓取才能得到
    return mean_mask_lm_loss

def load_model():
    ernie_config_path = "D:\workspace\project\\NLPcase\senti_continue_ernie\config\\ernie_config.json"
    cloze_config_path = "D:\workspace\project\\NLPcase\senti_continue_ernie\config\\cloze_config.json"
    ernie_config = json.loads(open(ernie_config_path, 'r', encoding='utf-8').read())
    cloze_config = json.loads(open(cloze_config_path, 'r', encoding='utf-8').read())
    use_cuda = False
    test_prog = fluid.Program()
    test_startup = fluid.Program()
    with fluid.program_guard(test_prog, test_startup):
        with fluid.unique_name.guard():
            pro = create_model(pyreader_name="test_reader", ernie_config=ernie_config,
                                              cloze_config=cloze_config)  # 得到这个reader的方法，现在需要实际往
    place = fluid.CUDAPlace(0) if use_cuda == True else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(test_startup)
    int_checkpoint_path = cloze_config['int_checkpoint']
    assert int_checkpoint_path is not None, "[FATAL] Please use --init_checkpoint '/path/to/checkpoints' \
                                                      to specify you pretrained model checkpoints"

    # 这里引用一个工具，用于记载模型所需要的参数
    init_pretraining_params(exe, int_checkpoint_path, test_prog)
    return exe,test_prog,pro
# 然后进行每个批量的预测

def topn_predict(batch, test_prog,exe,pro):
    src_ids, sent_ids, pos_ids, input_mask, mask_pos,mask_label = batch
    feed_dict = {
        "src_ids": src_ids,
        "pos_ids": pos_ids,
        "sent_ids": sent_ids,
        "input_mask": input_mask,
        "mask_pos": mask_pos,
        "mask_label":mask_label
    }

    loss = exe.run(program=test_prog, feed=feed_dict,fetch_list=[pro.name])
    return loss


def find_topn_candidates(sentences,exe,vocab_file,batch_size=1,test_prog=None,pro=None):
    d = ErnieDataReader(vocab_path=vocab_file, data=sentences, batch_size=1)
    res = []
    while True:
        batch = d.next_predict_batch(batch_size)
        if batch is not None:
            loss = topn_predict(batch, test_prog,exe,pro)
            res.append(loss)
            print(loss)
        else:
            break

    return loss

if __name__ == "__main__":
    file = "D:\\workspace\\project\\NLPcase\\senti_continue_ernie\\data\\test.txt"
    exe,test_prog,pro = load_model()
    sentences = process().read_file(file)
    vocab_file = "D:\\workspace\\project\\NLPcase\\senti_continue_ernie\\config\\vocab.txt"
    loss = find_topn_candidates(sentences,exe,vocab_file,batch_size=1,test_prog=test_prog,pro=pro)








#
#
#
#
#
#
#
#
#
#
# def predict_wrapper(exe,cloze_config,test_prog=None,fetch_list=None,batch=None):
#     # 判断int_checkpoint文件是否存在
#     int_checkpoint_path = cloze_config['int_checkpoint']
#     assert int_checkpoint_path is not None, "[FATAL] Please use --init_checkpoint '/path/to/checkpoints' \
#                                                   to specify you pretrained model checkpoints"
#
#     # 这里引用一个工具，用于记载模型所需要的参数
#     init_pretraining_params(exe,int_checkpoint_path,test_prog)
#     # init_checkpoint(exe, int_checkpoint_path, test_prog)
#     def predict(exe=exe,feed_dict=batch):
#         each_topn_predictions = exe.run(fetch_list=fetch_list, program=test_prog,feed=feed_dict)# ok,用最原生的来测试，用feed，这样就可以往模型中注入数据
#         return each_topn_predictions
#     return predict
#
# # 在这儿定义自己的主函数
# def main():
#     ernie_config_path = "D:\workspace\project\\NLPcase\senti_continue_ernie\config\\ernie_config.json"
#     cloze_config_path = "D:\workspace\project\\NLPcase\senti_continue_ernie\config\\cloze_config.json"
#     ernie_config = json.loads(open(ernie_config_path, 'r', encoding='utf-8').read())
#     cloze_config = json.loads(open(cloze_config_path, 'r', encoding='utf-8').read())
#     use_cuda = False
#     test_prog = fluid.Program()
#     test_startup = fluid.Program()
#     with fluid.program_guard(test_prog,test_startup):
#         with fluid.unique_name.guard():
#             test_pyreader,pro = create_model(pyreader_name="test_reader",ernie_config=ernie_config,cloze_config=cloze_config)# 得到这个reader的方法，现在需要实际往
#             #里面放入数据，因此可以放在predict上面，由test_pyreader往实际中填入数据#data2
#             # topn_predictions = fluid.layers.topk(pro,k=10,name="top10")# 这个必须在test_prog中
#
#     test_prog = test_prog.clone(for_test=True)
#     place = fluid.CUDAPlace(0) if use_cuda == True else fluid.CPUPlace()
#     exe = fluid.Executor(place)
#     exe.run(test_startup)
#
#     # 进行预测
#     predict = predict_wrapper(exe, ernie_config, cloze_config, test_prog=test_prog, pyreader=test_pyreader, fetch_list=[pro.name])
#     predictions = predict()

