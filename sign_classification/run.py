from json import dumps, load
from os import environ
from os.path import join
from sys import argv
import csv
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


def check_test(data_dir):
    with open(join(data_dir, 'output/output.csv')) as fout:
        lines = fout.readlines()
        output = {}
        for line in lines:
            filename, class_id = line.rstrip('\n').split(',')
            output[filename] = class_id

    with open(join(data_dir, 'gt/gt.csv')) as fgt:
        next(fgt)
        lines = fgt.readlines()
        gt = {}
        for line in lines:
            filename, class_id = line.rstrip('\n').split(',')
            gt[filename] = class_id

    correct = 0
    total = len(gt)
    for k, v in gt.items():
        if output[k] == v:
            correct += 1

    accuracy = correct / total

    res = 'Ok, accuracy %.4f' % accuracy
    if environ.get('CHECKER'):
        print(res)
    return res


def grade(data_path):
    results = load(open(join(data_path, 'results.json')))
    result = results[-1]['status']
    if not result.startswith('Ok'):
        res = {'description': '', 'mark': 0}
    else:
        accuracy_str = result[13:]
        accuracy = float(accuracy_str)

        if accuracy >= 0.93:
            mark = 10
        elif accuracy >= 0.90:
            mark = 8
        elif accuracy >= 0.85:
            mark = 6
        elif accuracy >= 0.80:
            mark = 4
        elif accuracy >= 0.75:
            mark = 2
        elif accuracy > 0:
            mark = 1
        else:
            mark = 0

        res = {'description': accuracy_str, 'mark': mark}
    if environ.get('CHECKER'):
        print(dumps(res))
    return res


def run_single_test(data_dir, output_dir, seed):
    from fit_and_classify import fit_and_classify, extract_hog, extract_data
    from glob import glob
    from numpy import zeros
    from os.path import basename, join
    from skimage.io import imread

    train_dir = join(data_dir, 'train')
    test_dir = join(data_dir, 'test')

    def read_gt(gt_dir):
        fgt = open(join(gt_dir, 'gt.csv'))
        next(fgt)
        lines = fgt.readlines()

        filenames = []
        labels = zeros(len(lines))
        for i, line in enumerate(lines):
            filename, label = line.rstrip('\n').split(',')[:2]
            filenames.append(filename)
            labels[i] = int(label)

        return filenames, labels

    HOG_FILENAME = 'train_hog_file_size88_block4_423fixed.csv'

    def dump_features(path, filenames):
        hog_length = len(extract_hog(imread(join(path, filenames[0]))))
        data = zeros((len(filenames), hog_length))
        with open(HOG_FILENAME, mode='w') as hog_file:
            hog_writer = csv.writer(hog_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            hog_writer.writerow(['filename', 'hog_vector'])
            for i in range(0, len(filenames)):
                filename = join(path, filenames[i])
                data[i, :] = extract_hog(imread(filename))
                hog_writer.writerow([filename, ','.join(np.asarray(np.round(data[i], 3), dtype=str))])
                if i % 100 == 0:
                    print('{} done'.format(i))

    def extract_features(path, filenames):
        hog_length = len(extract_hog(imread(join(path, filenames[0]))))
        data = zeros((len(filenames), hog_length))
        hog_data = pd.read_csv(HOG_FILENAME)
        hog_data = hog_data.set_index('filename')
        for i in range(0, len(filenames)):
            filename = join(path, filenames[i])
            data[i, :] = np.asarray(hog_data.loc[filename].hog_vector.split(','), dtype=float)
            if i % 5000 == 0:
                print('{} done'.format(i))

        train_data, test_data = extract_data(seed)

        train_data['filenames'] = 'public_tests/00_test_img_input/train/' + train_data['filenames']
        test_data['filenames'] = 'public_tests/00_test_img_input/train/' + test_data['filenames']
        train_data = train_data.merge(hog_data, how='inner', left_on='filenames', right_on='filename')
        test_data = test_data.merge(hog_data, how='inner', left_on='filenames', right_on='filename')
        return train_data, test_data

    train_filenames, train_labels = read_gt(train_dir)
    test_filenames = []
    for path in sorted(glob(join(test_dir, '*png'))):
        test_filenames.append(basename(path))

    # train_features = extract_features(train_dir, train_filenames, 'train_hog_file.csv')
    # test_features = extract_features(test_dir, test_filenames, 'test_hog_file.csv')

    # dump_features(train_dir, train_filenames)
    train_data, test_data = extract_features(train_dir, train_filenames)
    train_features = np.stack(train_data['hog_vector'].apply(lambda x: np.asarray(x.split(','), dtype=float)))
    test_features = np.stack(test_data['hog_vector'].apply(lambda x: np.asarray(x.split(','), dtype=float)))
    train_labels = np.array(train_data['class_id'])
    test_labels = np.array(test_data['class_id'])

    y = fit_and_classify(train_features, train_labels, test_features)

    print("Length of test: {}".format(len(y)))
    correct = 0
    incorrect = 0
    conf_matrix = np.zeros((43, 43), dtype=int)
    miss = np.zeros((43), dtype=int)
    total = np.zeros((43), dtype=int)
    for i in range(len(y)):
        conf_matrix[y[i], test_labels[i]] += 1
        total[test_labels[i]]+=1
        if y[i] == test_labels[i]:
            correct += 1
        else:
            incorrect += 1
            miss[test_labels[i]]+=1
            # print("Name: {}, expected: {}, actual: {}".format(test_data.filenames[i], test_labels[i], y[i]))

    print(HOG_FILENAME)
    print("Correct: {}; Incorrect: {}; Accuracy: {}; Seed: {}".format(correct, incorrect, correct / len(y) * 100, seed))
    print("Per class stats:")
    for i in range(43):
        conf_matrix[i][i] //= 100
        print("Total: {}, missed: {}, miss percentage: {}".format(total[i], miss[i], miss[i] / total[i]))

    df_cm = pd.DataFrame(conf_matrix, index=[i for i in range(43)], columns=[i for i in range(43)])
    plt.figure(figsize=(15, 11))
    sn.heatmap(df_cm, annot=True)
    plt.show()

    # with open(join(output_dir, 'output.csv'), 'w') as fout:
    #     for i, filename in enumerate(test_filenames):
    #         print('%s,%d' % (filename, y[i]), file=fout)


if __name__ == '__main__':
    if environ.get('CHECKER'):
        # Script is running in testing system
        if len(argv) != 4:
            print('Usage: %s mode data_dir output_dir' % argv[0])
            exit(0)

        mode = argv[1]
        data_dir = argv[2]
        output_dir = argv[3]

        if mode == 'run_single_test':
            run_single_test(data_dir, output_dir)
        elif mode == 'check_test':
            check_test(data_dir)
        elif mode == 'grade':
            grade(data_dir)
    else:
        # Script is running locally, run on dir with tests
        if len(argv) != 2:
            print('Usage: %s tests_dir' % argv[0])
            exit(0)

        from glob import glob
        from json import dump
        from re import sub
        from time import time
        from traceback import format_exc
        from os import makedirs
        from os.path import exists
        from shutil import copytree

        tests_dir = argv[1]

        results = []
        for input_dir in sorted(glob(join(tests_dir, '[0-9][0-9]_*_input'))):
            output_dir = sub('input$', 'check', input_dir)
            run_output_dir = join(output_dir, 'output')
            makedirs(run_output_dir, exist_ok=True)
            gt_src = sub('input$', 'gt', input_dir)
            gt_dst = join(output_dir, 'gt')
            if not exists(gt_dst):
                copytree(gt_src, gt_dst)

            try:
                start = time()
                for i in range(10):
                    run_single_test(input_dir, run_output_dir, i + 100)
                end = time()
                running_time = end - start
            except:
                status = 'Runtime error'
                traceback = format_exc()
            else:
                try:
                    status = check_test(output_dir)
                except:
                    status = 'Checker error'
                    traceback = format_exc()

            test_num = input_dir[-8:-6]
            if status == 'Runtime error' or status == 'Checker error':
                print(test_num, status, '\n', traceback)
                results.append({'status': status})
            else:
                print(test_num, '%.2fs' % running_time, status)
                results.append({
                    'time': running_time,
                    'status': status})

        dump(results, open(join(tests_dir, 'results.json'), 'w'))
        res = grade(tests_dir)
        print('Mark:', res['mark'], res['description'])
