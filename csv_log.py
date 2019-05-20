import csv
import os
import time


class LogRecord():
    def __init__(self, path='./TrainLog', file_name=None, head=None, time_stamp=True):
        self.path = path
        self.create

        if not head:
            self.head = ['Epoch', 'confusion_Loss', 'label_Loss', 'Acc_src']
        else:
            self.head = head

        if not file_name:
            self.file_name = 'Train Log'
        else:
            self.file_name = file_name
        if time_stamp:
            timestamp = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
            self.file_name = self.file_name + '-' + timestamp

        self.file = self.path + '/' + self.file_name + '.csv'
        with open(self.file, 'a', newline='') as file:
            csv_write = csv.writer(file, dialect='excel')
            csv_write.writerow(self.head)

    @property
    def create(self):
        try:
            os.mkdir(self.path)
        except IOError:
            print('{} has exist!'.format(self.path))
        return self.path

    def write(self, content):
        with open(self.file, 'a', newline='') as file:
            csv_write = csv.writer(file, dialect='excel')
            csv_write.writerow(content)


if __name__ == '__main__':
    loger = LogRecord()
    loger.write(['ddd', 555, 888, 999])
