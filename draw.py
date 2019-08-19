# -*- coding: utf-8 -*-

#    --log_dir为日志存放地址，输入格式为--log_dir=C:\Users\DELL\Desktop\logs
#    --tools为所用框架名称，输入格式为--tools=tensorflow,pytorch,mxnet'
#    --models为每个框架下模型的名称，输入格式为--models=alexnet,resnet50,vgg16,inception3
#    --data_type为训练所用的数据类型(real代表真实数据，synt代表合成数据)，输入格式为--data_type=real,synt

#    当可以运行两机八卡时，请将代码被注释的部分取消注释
#    不同模型的日志提取只需要在read_one_file与throughput_average中相应特性即可





import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# 设置中文字体和负号正常显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

GPUs = [1, 4]


def parse_arguements(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, help='', default='./log')
    parser.add_argument('--output_dir', type=str, help='', default='./output')
    parser.add_argument('--tools',type=str,help='',default='tensorflow,pytorch,mxnet')
    parser.add_argument('--models', type=str, help='', default='alexnet,resnet50,vgg16,inception3')
    parser.add_argument('--data_type', type=str, help='', default='real,synt')
    return parser.parse_args(argv)


def read_one_file(tool,model,filename):
    #此处根据日志特点进行提取适配
    if tool=='tensorflow':
        if model=='alexnet'or model=='resnet50'or model=='vgg16' or model=='inception3'or model=='yolov3':
            job_str = open(filename).read()
            lines = job_str.splitlines()
            for l in lines:
                if r"total images/sec:" in l:
                    num = l.split(':')[-1].strip()
            return float(num)
    #     if model=='':
    #
    # if tool=='mxnet':
    #
    # if tool=='pytorch':




def throughput_average(tool,model,logdir):
    # 计算平均吞吐率
    # 输入日志路径
    # 返回平均吞吐率
    if tool=='tensorflow':
        if model=='alexnet'or model=='resnet50'or model=='vgg16' or model=='inception3'or model=='yolov3':
            num_files = len(os.listdir(logdir))
            sum = 0
            for i in range(1, num_files + 1):
                logtxt = os.path.join(logdir, str(i) + '.log')
                sum += read_one_file(tool, model, logtxt)
            avg = sum / num_files
            return int(avg)
    #    if model=='':
    # if tool=='mxnet':
    #
    # if tool=='pytorch':


def speed_up(model_avg):
    # 计算加速比
    # 返回模型加速比数组
    i = 1
    model_speedup = []
    tmp = model_avg[0]
    for avg in model_avg:
        if i == 1:
            speedup = 1
        else:
            speedup = avg / tmp
        tmp = avg
        model_speedup.append(speedup)
        i += 1
    return model_speedup

def throughput_rate_histogram(args):
    #绘制直方图
    data_type = args.data_type.split(',')
    tool_catalog = os.listdir(args.log_dir)
    tools = args.tools.split(',')
    models = args.models.split(',')
    tool_path = os.path.join(args.log_dir, tools[0])
    model_path = os.path.join(os.path.join(tool_path, models[0]), data_type[0])
    test_count=os.listdir(model_path)
    for type in data_type:
        if type == 'real':
            for tool in tools:
                model_tests = np.zeros([len(models), len(test_count)],dtype=float)
                if tool in tool_catalog:
                    tool_path = os.path.join(args.log_dir, tool)
                    model_catalog = os.listdir(tool_path)
                    for model in models:
                        i=0
                        if model in model_catalog:
                            model_path = os.path.join(os.path.join(tool_path, model), 'real')
                            model_tests[i, 0] = throughput_average(tool,model,os.path.join(model_path, '1n1c'))
                            model_tests[i, 1] = throughput_average(tool,model,os.path.join(model_path, '1n4c'))
                            # model_tests[i, 2] = throughput_average(tool,model,os.path.join(model_path, '2n8c'))
                            i += 1

                    base_num = range(len(models))
                    print(model_tests)
                    hist1 = plt.bar(base_num, height=list(model_tests[:, 0]), width=0.4, alpha=0.8, color='orange',
                                    label="1n1c")
                    hist2 = plt.bar([i + 0.4 for i in base_num], height=list(model_tests[:, 1]), width=0.4,
                                    color='orange', label="1n4c")
                    hist3 = plt.bar([i + 0.8 for i in base_num], height=list(model_tests[:, 2]), width=0.4,
                                    color='orange', label="2n8c")
                    plt.ylim(0, model_tests.max() * 1.2)
                    plt.ylabel("Image/sec")
                    plt.xticks([index + 0.2 for index in base_num], models)
                    plt.xlabel("Models")
                    plt.title(tool)
                    plt.legend()
                    for rect in hist1:
                        height = rect.get_height()
                        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
                    for rect in hist2:
                        height = rect.get_height()
                        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
                    for rect in hist3:
                        height = rect.get_height()
                        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
                    plt.savefig(tool + '_histogram' + '.png')
                    plt.show()




        if type == 'synt':
            for tool in tools:
                i = 0
                model_tests = np.zeros([len(models), len(test_count)],dtype=float)
                if tool in tool_catalog:
                    tool_path = os.path.join(args.log_dir, tool)
                    model_catalog = os.listdir(tool_path)
                    for model in models:
                        if model in model_catalog:
                            model_path = os.path.join(os.path.join(tool_path, model), 'synt')
                            model_tests[i, 0] = throughput_average(tool,model,os.path.join(model_path, '1n1c'))
                            model_tests[i, 1] = throughput_average(tool,model,os.path.join(model_path, '1n4c'))
                            # model_tests[i, 2] = throughput_average(tool,model,os.path.join(model_path, '2n8c'))
                            i+=1

                    base_num = range(len(models))
                    print(model_tests)
                    hist1 = plt.bar(base_num, height=list(model_tests[:, 0]), width=0.4, alpha=0.8, color='orange',
                                    label="1n1c")
                    hist2 = plt.bar([i + 0.4 for i in base_num], height=list(model_tests[:, 1]), width=0.4,
                                    color='orange', label="1n4c")
                    # hist3 = plt.bar([i + 0.8 for i in base_num], height=list(model_tests[:, 2]), width=0.4,
                    #                 color='orange', label="2n8c")
                    plt.ylim(0, model_tests.max() * 1.2)
                    plt.ylabel("Image/sec")
                    plt.xticks([index + 0.2 for index in base_num], models)
                    plt.xlabel("Models")
                    plt.title(tool)
                    plt.legend()
                    for rect in hist1:
                        height = rect.get_height()
                        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
                    for rect in hist2:
                        height = rect.get_height()
                        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
                    # for rect in hist3:
                    #     height = rect.get_height()
                    #     plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
                    plt.savefig(tool + '_histogram' + '.png')
                    plt.show()

def speedup_line(args):
    # 绘制加速比曲线
    tool_catalog = os.listdir(args.log_dir)
    data_type = args.data_type.split(',')
    for type in data_type:
        if type == 'real':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set(xlim=[0, 8], ylim=[0, 8], title='model_speedup(real data)',
                   ylabel='speedup(x)', xlabel='GPUs')
            tools = args.tools.split(',')
            for tool in tools:
                if tool in tool_catalog:
                    tool_path = os.path.join(args.log_dir, tool)
                    model_catalog = os.listdir(tool_path)
                    models = args.models.split(',')
                    for model in models:
                        if model in model_catalog:
                            model_path = os.path.join(os.path.join(tool_path, model), 'real')
                            model_avgs_real = []
                            real_avg1 = throughput_average(tool,model,os.path.join(model_path, '1n1c'))
                            model_avgs_real.append(real_avg1)
                            real_avg2 = throughput_average(tool,model,os.path.join(model_path, '1n4c'))
                            model_avgs_real.append(real_avg2)
                            real_avg3 = throughput_average(tool,model,os.path.join(model_path, '2n8c'))
                            model_avgs_real.append(real_avg3)
                            model_speedup_real = speed_up(model_avgs_real)
                            ax.plot(GPUs, model_speedup_real, label="$" + model + "$")
                    fig.suptitle(tool)
                    plt.legend()
                    plt.savefig(tool+'_speedup' + '.png')
            plt.show()

        if type == 'synt':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set(xlim=[0, 8], ylim=[0, 8], title='model_speedup(synthetic data)',
                   ylabel='speedup(x)', xlabel='GPUs')
            tools = args.tools.split(',')
            for tool in tools:
                if tool in tool_catalog:
                    tool_path = os.path.join(args.log_dir, tool)
                    model_catalog = os.listdir(tool_path)
                    models = args.models.split(',')
                    print(models)
                    for model in models:
                        if model in model_catalog:
                            model_path = os.path.join(os.path.join(tool_path, model), 'synt')
                            model_avgs_synt = []
                            real_avg1 = throughput_average(tool,model,os.path.join(model_path, '1n1c'))
                            model_avgs_synt.append(real_avg1)
                            real_avg2 = throughput_average(tool,model,os.path.join(model_path, '1n4c'))
                            model_avgs_synt.append(real_avg2)
                            # real_avg3 = throughput_average(tool,model,os.path.join(model_path, '2n8c'))
                            # model_avgs_synt.append(real_avg3)
                            model_speedup_synt = speed_up(model_avgs_synt)
                            ax.plot(GPUs, model_speedup_synt, label="$" + model + "$")
                    fig.suptitle(tool)
                    plt.legend()
                    plt.savefig(tool+'_speedup' + '.png')
            plt.show()




def main(args):
    # 绘制加速比曲线
    # 基本实现思路，调用os.path.listdir()获取文件夹中的文件，调取os.path.join()向下遍历搜索
    throughput_rate_histogram(args)
    speedup_line(args)


if __name__ == '__main__':
    main(parse_arguements(sys.argv[1:]))
