
import os

class DataObject:
    def __init__(self):
        self.precision_map = {}
        self.recall_map = {}
        self.f1_map = {}
        self.count_map = {}

    def print(self):
        print(f'{"Name":<30} {"Precision":>10} {"Recall":>10} {"F1":>10} {"Count":>8}')
        print('-' * 68)
        for key in self.precision_map.keys():
            print(f'{key:<30} '
                  f'{(self.precision_map[key] / self.count_map[key]):>10.2f} '
                  f'{(self.recall_map[key] / self.count_map[key]):>10.2f} '
                  f'{(self.f1_map[key] / self.count_map[key]):>10.2f} '
                  f'{self.count_map[key]:>8}')


def calculate_classification_report_from_runs(folder):
    files_contents = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            with open(f'{folder}/{file}', "r") as f:
                Data = DataObject()
                file_contents = f.readlines()
                for line in file_contents:
                    array = line.split('  ')
                    new_array = []
                    for item in array:
                        if item == '':
                            continue
                        else:
                            new_array.append(item.replace('\n', '').replace(' ', ''))
                    if len(new_array) > 1:
                        try:
                            Data.precision_map[new_array[0]] = float(new_array[1])
                            Data.recall_map[new_array[0]] = float(new_array[2])
                            Data.f1_map[new_array[0]] = float(new_array[3])
                            Data.count_map[new_array[0]] = int(new_array[4])
                        except Exception as e:
                            print('skipping; value was not a float')
                files_contents.append([f'{file}', Data])

    CombinedData = DataObject()
    for key in files_contents[0][1].f1_map.keys():
        CombinedData.f1_map[key] = 0
        CombinedData.precision_map[key] = 0
        CombinedData.recall_map[key] = 0
        CombinedData.count_map[key] = 0

    for file in files_contents:
        dataObj = file[1]
        for key in dataObj.f1_map.keys():
            if type(dataObj.f1_map[key]) == str:
                continue

            weighted_f1 = dataObj.f1_map[key] * dataObj.count_map[key]
            weighted_precision = dataObj.precision_map[key] * dataObj.count_map[key]
            weighted_recall = dataObj.recall_map[key] * dataObj.count_map[key]
            CombinedData.f1_map[key] += weighted_f1
            CombinedData.precision_map[key] += weighted_precision
            CombinedData.recall_map[key] += weighted_recall
            CombinedData.count_map[key] += dataObj.count_map[key]

    CombinedData.print()


if __name__ == '__main__':
    current_dir = os.getcwd()
    calculate_classification_report_from_runs(current_dir + '/autorunclassificationreport/ASTCFGDDG')

