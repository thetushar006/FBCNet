import numpy as np
import torch
import sys
import os
import time
import xlwt
import csv
import random
import math
import copy
import argparse
from codes.centralRepo.eegDataset import eegDataset
from codes.centralRepo.baseModel import baseModel
import codes.centralRepo.networks as networks
import codes.centralRepo.transforms as transforms
from codes.centralRepo.saveData import fetchData
import DataLoadingUtils.LoadKUMulti as ld
from functools import partial

def get_args():
    """To get arguments passed to the script"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="D:\Data\KU_Dataset\BCIdataset\DB_mat\multiviewPython",
                        help="Path to dataset")
    parser.add_argument("--test_sub", type=int, default=0, help="Index of test subject")
    parser.add_argument("--dataset_idx", choices=[0, 1], type=int, default=0, help="0: KU Binary, 1: KU Multi")
    parser.add_argument("--classes", choices=[0, 1, 2], type=int, default=0, help="0: Forward vs Backward, 1:Cylindrical vs Lumbrical, 2: Forward vs Cylindrical")
    parser.add_argument("--move_type", choices=[0, 1], type=int, default=0, help="for KU Multi data only; 0: MI, 1: realMove")
    parser.add_argument("--pre_transform", action="store_true", default=False, help="Whether to transform all data to multiview before training")
    return parser.parse_args()

def build_config(args):
    config = {}
    masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASETS = ["KU_Binary", "KU_Multi"]
    # Data load options:
    config['preloadData'] = False  # whether to load the complete data in the memory

    # Random seed
    config['randSeed'] = 20190821

    # KU Multi Data Config
    if args.dataset_idx == 1:
        config['move_type'] = args.move_type

        # Classes to use for classification for KU MultiClass Data
        if args.classes == 0:
            config['class_combination'] = [1, 4]
        elif args.classes == 1:
            config['class_combination'] = [2, 6]
        else:
            config['class_combination'] = [2, 4]

    # Network related details
    config['batchSize'] = 16

    if args.dataset_idx == 0:
        config['modelArguments'] = {'nChan': 20, 'nTime': 1000, 'dropoutP': 0.5,
                                'nBands': 9, 'm': 32, 'temporalLayer': 'LogVarLayer',
                                'nClass': 2, 'doWeightNorm': True}
    else:
        config['modelArguments'] = {'nChan': 60, 'nTime': 1000, 'dropoutP': 0.5,
                                    'nBands': 9, 'm': 32, 'temporalLayer': 'LogVarLayer',
                                    'nClass': 2, 'doWeightNorm': True}


    # Training related details
    config['modelTrainArguments'] = {
        'stopCondi': {'c': {'Or': {'c1': {'MaxEpoch': {'maxEpochs': 1500, 'varName': 'epoch'}},
                                   'c2': {'NoDecrease': {'numEpochs': 200, 'varName': 'valInacc'}}}}},
        'classes': [0, 1], 'sampler': 'RandomSampler', 'loadBestModel': True,
        'bestVarToCheck': 'valInacc', 'continueAfterEarlystop': True, 'lr': 1e-3}
    config['transformArguments'] = None

    # FilerBank details for transform
    config['transform'] = {'filterBank': {
        'filtBank': [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]], 'fs': 250,
        'filtType': 'filter'}}

    # add some more run specific details.
    config['cv'] = 'trainTest'
    config['kFold'] = 1
    config['data'] = 'raw'
    config['subTorun'] = args.test_sub
    config['trainDataToUse'] = 1  # How much data to use for training
    config['validationSet'] = 0.2  # how much of the training data will be used a validation set

    # network initialization details:
    config['loadNetInitState'] = True
    config['pathNetInitState'] = f"KU_{DATASETS[args.dataset_idx]}"

    # %% Define data path things here. Do it once and forget it!
    # Input data base folder:
    toolboxPath = os.path.dirname(masterPath)
    config['inDataPath'] = args.data_path

    # Input data datasetId folders
    modeInFol = 'multiviewPython'  # FBCNet uses multi-view data

    # Path to the input data labels file
    config['inLabelPath'] = os.path.join(config['inDataPath'], 'dataLabels.csv')

    # Output folder:
    # Lets store all the outputs of the given run in folder.
    config['outPath'] = os.path.join(toolboxPath, 'output')

    # Network initialization:
    config['pathNetInitState'] = os.path.join(masterPath, 'netInitModels', config['pathNetInitState'] + '.pth')
    # check if the file exists else raise a flag
    config['netInitStateExists'] = os.path.isfile(config['pathNetInitState'])

    randomFolder = str(time.strftime("%Y-%m-%d--%H-%M", time.localtime())) + '-' + str(random.randint(1, 1000))
    config['outPath'] = os.path.join(config['outPath'], randomFolder, '')
    # create the path
    if not os.path.exists(config['outPath']):
        os.makedirs(config['outPath'])
    print('Outputs will be saved in folder : ' + config['outPath'])

    # Write the config dictionary
    dictToCsv(os.path.join(config['outPath'], 'config.csv'), config)

    # %% Check and compose transforms
    if config['transformArguments'] is not None:
        if len(config['transformArguments']) > 1:
            transform = transforms.Compose(
                [transforms.__dict__[key](**value) for key, value in config['transformArguments'].items()])
        else:
            transform = transforms.__dict__[list(config['transformArguments'].keys())[0]](
                **config['transformArguments'][list(config['transformArguments'].keys())[0]])
    else:
        transform = None
    return config, transform

def setRandom(seed):
    '''
    Set all the random initializations with a given seed

    '''
    # Set np
    np.random.seed(seed)

    # Set torch
    torch.manual_seed(seed)

    # Set cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def excelAddData(worksheet, startCell, data, isNpData = False):
    '''
        Write the given max 2D data to a given given worksheet starting from the start-cell.
        List will be treated as a row.
        List of list will be treated in a matrix format with inner list constituting a row.
        will return the modified worksheet which needs to be written to a file
        isNpData flag indicate whether the incoming data in the list is of np data-type
    '''
    #  Check the input type.
    if type(data) is not list:
        data = [[data]]
    elif type(data[0]) is not list:
        data = [data]
    else:
        data = data

    # write the data. starting from the given start cell.
    rowStart = startCell[0]
    colStart = startCell[1]

    for i, row in enumerate(data):
        for j, col in enumerate(row):
            if isNpData:
                worksheet.write(rowStart+i, colStart+j, col.item())
            else:
                worksheet.write(rowStart+i, colStart+j, col)

    return worksheet

def dictToCsv(filePath, dictToWrite):
    """
    Write a dictionary to a given csv file
    """
    with open(filePath, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dictToWrite.items():
            writer.writerow([key, value])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_multiview_KU_binary_data(data, config, sub):
    # extract subject data
    subIdx = [i for i, x in enumerate(data.labels) if x[3] in sub]
    subData = copy.deepcopy(data)
    subData.createPartialDataset(subIdx, loadNonLoadedData=True)

    trainData = copy.deepcopy(subData)
    testData = copy.deepcopy(subData)

    # Isolate the train -> session 0 and test data-> session 1
    if len(subData.labels[0]) > 4:
        idxTrain = [i for i, x in enumerate(subData.labels) if x[4] == '0']
        idxTest = [i for i, x in enumerate(subData.labels) if x[4] == '1']
    else:
        raise ValueError("The data can not be divided based on the sessions")

    testData.createPartialDataset(idxTest)
    trainData.createPartialDataset(idxTrain)

    # extract the desired amount of train data:
    trainData.createPartialDataset(list(range(0, math.ceil(len(trainData) * config['trainDataToUse']))))

    # isolate the train and validation set
    valData = copy.deepcopy(trainData)
    valData.createPartialDataset(list(range(
        math.ceil(len(trainData) * (1 - config['validationSet'])), len(trainData))))
    trainData.createPartialDataset(list(range(0, math.ceil(len(trainData) * (1 - config['validationSet'])))))

    return trainData, valData, testData

def get_multiview_KU_multi_data(test_subject, config, args):
    ALL_SUBJECTS = np.array(range(1,26))
    train_subjects = ALL_SUBJECTS[ALL_SUBJECTS != test_subject]
    labels_idx_for_classif = config['class_combination']
    ALL_LABELS = ld.LoadKUMulti().ALL_LABELS
    MOVE_TYPE = ld.LoadKUMulti().ALL_MOVE_TYPE[config['move_type']]
    labels_for_classif = [ALL_LABELS[idx] for idx in labels_idx_for_classif]
    print(f"Labels chosen for multi-class classification = {labels_for_classif}")
    print(f"Test Subject {test_subject} \n")
    load_data = partial(ld.LoadKUMulti().get_multi_subject_data, labels_for_classif,
                        MOVE_TYPE, args.data_path)
    x_train, y_train = load_data(train_subjects)
    x_test, y_test = load_data([test_subject])
    FS = 250
    multiview_transform = transforms.filterBank(**config["transform"]["filterBank"])
    combined_data_train = {'data': x_train, 'label': y_train}
    combined_data_test = {'data': x_test, 'label': y_test}
    train_data = eegDataset(dataPath="", dataLabelsPath="", data=combined_data_train, pretransform=args.pre_transform,
                            transform=multiview_transform)
    test_data = eegDataset(dataPath="", dataLabelsPath="", data=combined_data_test, pretransform=args.pre_transform,
                           transform=multiview_transform)
    return train_data, test_data

def experiment(args):
    config, transform = build_config(args)

    # check and Load the data
    if args.dataset_idx == 0:
        print('Data loading in progress')
        data = eegDataset(dataPath=config['inDataPath'], dataLabelsPath=config['inLabelPath'],
                          preloadData=config['preloadData'], transform=transform)
        print('Data loading finished')

    network = networks.FBCNet

    # Load the net and print trainable parameters:
    net = network(**config['modelArguments'])
    print('Trainable Parameters in the network are: ' + str(count_parameters(net)))

    # %% check and load/save the the network initialization.
    if config['loadNetInitState']:
        if config['netInitStateExists']:
            netInitState = torch.load(config['pathNetInitState'])
        else:
            setRandom(config['randSeed'])
            net = network(**config['modelArguments'])
            netInitState = net.to('cpu').state_dict()
            torch.save(netInitState, config['pathNetInitState'])

    # Find all the subjects to run
    if args.dataset_idx == 0:
        subs = sorted(set([d[3] for d in data.labels]))
        nSub = len(subs)
        test_subs = [subs[args.test_sub]]
    else:
        # test_subs = [args.test_sub]
        test_subs = range(1,26)

    # %% Let the training begin
    trainResults = []
    valResults = []
    testResults = []

    for iSub, sub in enumerate(test_subs):

        start = time.time()
        if args.dataset_idx == 0:
            trainData, valData, testData = get_multiview_KU_binary_data(data, config, sub)
        else:
            trainData, testData = get_multiview_KU_multi_data(sub, config, args)
            valData = []

        # Call the network for training
        setRandom(config['randSeed'])
        net = network(**config['modelArguments'])
        net.load_state_dict(netInitState, strict=False)

        outPathSub = os.path.join(config['outPath'], 'sub' + str(iSub))
        model = baseModel(net=net, resultsSavePath=outPathSub, batchSize=config['batchSize'], nGPU=0)
        model.train(trainData, valData, testData, **config['modelTrainArguments'])

        # extract the important results.
        trainResults.append([d['results']['trainBest'] for d in model.expDetails])
        # valResults.append([d['results']['valBest'] for d in model.expDetails])
        testResults.append([d['results']['test'] for d in model.expDetails])

        # save the results
        # results = {'train:': trainResults[-1], 'val: ': valResults[-1], 'test': testResults[-1]}
        results = {'train:': trainResults[-1], 'test': testResults[-1]}
        dictToCsv(os.path.join(outPathSub, 'results.csv'), results)

        # Time taken
        print("Time taken = " + str(time.time() - start))

        del trainData, testData

    # %% Extract and write the results to excel file.

    # lets group the results for all the subjects using experiment.
    # the train, test and val accuracy and cm will be written

    trainAcc = [[r['acc'] for r in result] for result in trainResults]
    trainAcc = list(map(list, zip(*trainAcc)))
    # valAcc = [[r['acc'] for r in result] for result in valResults]
    # valAcc = list(map(list, zip(*valAcc)))
    testAcc = [[r['acc'] for r in result] for result in testResults]
    testAcc = list(map(list, zip(*testAcc)))

    print("Results sequence is train, val , test")
    print(trainAcc)
    # print(valAcc)
    print(testAcc)

    # append the confusion matrix
    trainCm = [[r['cm'] for r in result] for result in trainResults]
    trainCm = list(map(list, zip(*trainCm)))
    trainCm = [np.concatenate(tuple([cm for cm in cms]), axis=1) for cms in trainCm]

    # valCm = [[r['cm'] for r in result] for result in valResults]
    # valCm = list(map(list, zip(*valCm)))
    # valCm = [np.concatenate(tuple([cm for cm in cms]), axis=1) for cms in valCm]

    testCm = [[r['cm'] for r in result] for result in testResults]
    testCm = list(map(list, zip(*testCm)))
    testCm = [np.concatenate(tuple([cm for cm in cms]), axis=1) for cms in testCm]

    print(trainCm)
    # print(valCm)
    print(testCm)
    # %% Excel writing
    book = xlwt.Workbook(encoding="utf-8")
    for i, res in enumerate(trainAcc):
        sheet1 = book.add_sheet('exp-' + str(i + 1), cell_overwrite_ok=True)
        sheet1 = excelAddData(sheet1, [0, 0], ['SubId', 'trainAcc', 'testAcc'])
        # sheet1 = excelAddData(sheet1, [0, 0], ['SubId', 'trainAcc', 'valAcc', 'testAcc'])
        sheet1 = excelAddData(sheet1, [1, 0], [[sub] for sub in test_subs])
        sheet1 = excelAddData(sheet1, [1, 1], [[acc] for acc in trainAcc[i]], isNpData=True)
        # sheet1 = excelAddData(sheet1, [1, 2], [[acc] for acc in valAcc[i]], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 3], [[acc] for acc in testAcc[i]], isNpData=True)

        # write the cm
        for isub, sub in enumerate(test_subs):
            sheet1 = excelAddData(sheet1,
                                  [len(trainAcc[0]) + 5, 0 + isub * len(config['modelTrainArguments']['classes'])], sub)
        sheet1 = excelAddData(sheet1, [len(trainAcc[0]) + 6, 0], ['train CM:'])
        sheet1 = excelAddData(sheet1, [len(trainAcc[0]) + 7, 0], trainCm[i].tolist(), isNpData=False)
        # sheet1 = excelAddData(sheet1, [len(trainAcc[0]) + 11, 0], ['val CM:'])
        # sheet1 = excelAddData(sheet1, [len(trainAcc[0]) + 12, 0], valCm[i].tolist(), isNpData=False)
        sheet1 = excelAddData(sheet1, [len(trainAcc[0]) + 17, 0], ['test CM:'])
        sheet1 = excelAddData(sheet1, [len(trainAcc[0]) + 18, 0], testCm[i].tolist(), isNpData=False)

    book.save(os.path.join(config['outPath'], 'results.xls'))


if __name__ == "__main__":
    args = get_args()
    experiment(args)
