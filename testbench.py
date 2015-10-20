import cPickle
from random import randrange
from sys import stdout

from math import floor

from ModelFactory import *
from output48_39 import *


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self):
        pass


def my_print(title, content=None, switch=True):
    if switch is False:
        return
    if content is None:
        print BColors.OKGREEN, "*", title, ":", BColors.ENDC
        return
    print BColors.OKGREEN, "*", title, ":", BColors.ENDC, "\n", content


class TestBench:
    def __init__(self):
        # Define training file
        self.train_file = "train_sub.ark"
        self.train_input_data = []
        self.answer_map_file = "answer_map_sub.txt"
        self.train_answer_data = []

        # Define test file
        self.test_data = []
        self.test_input_file = "test.ark"
        self.test_output_file = "test_ans.csv"
        self.W_parm = None
        self.B_parm = None

        # Define model parameter
        self.adagrad_enable = False  # should be defined in the model
        self.layer = [512]

        self.input_dimension = 69  # Dimension of input vector
        self.output_dimension = 48  # Dimension of output vector
        self.batch_number = 3  # Number of batch size
        self.lr = 0.001

        self.modified = False
        self.correct = 0
        self.total = 0
        self.current = 0
        self.cost = {}
        self.acc = []

        self.model = None

        self.ans_type = [
            "aa", "ae", "ah", "ao", "aw", "ax", "ay", "b", "ch", "cl", "d",
            "dh", "dx", "eh", "el", "en", "epi", "er", "ey", "f", "g", "hh",
            "ih", "ix", "iy", "jh", "k", "l", "m", "ng", "n", "ow", "oy", "p",
            "r", "sh", "sil", "s", "th", "t", "uh", "uw", "vcl", "v", "w",
            "y", "zh", "z"
        ]

        self.prompt = "Enter a command:"
        while True:
            command = raw_input(BColors.HEADER + self.prompt + BColors.ENDC)
            self.__exec_command__(command)

    def create_model(self):
        self.model = ModelFactory(
            self.input_dimension,
            self.output_dimension,
            self.layer,
            self.batch_number,
            self.lr
        )

    def __exec_command__(self, cmd):
        dispatcher = {
            "status": self._get_status,
            "train": self._train_data,
            "load": self._load_parameter,
            "run": self._run,
            "output": self._output_result,
            "set": self._set,
            "quit": self._quit
        }
        commands = cmd.strip().split(';')

        for x in commands:
            m = x.strip().split()
            if len(m) == 0:
                return
            func = dispatcher.get(m[0], lambda (xx): self.__do_nothing__)
            func(m)

    '''
    #################### Dispatcher Functions ##############################
    '''

    def __do_nothing__(self, x):
        pass

    def _get_status(self, param):
        if self.model is not None:
            my_print("Model created")
            if self.modified:
                my_print("Correct ratio", self)
                pass
            else:
                pass
            # my_print("Adagrad enable", "True" if self.adagrad_enable else "False")
            # my_print("Dropout enable", "True" if self.dropout_enable else "False")
            pass
        else:
            my_print("Model haven't created")
            my_print("Layer", self.layer)
            my_print("Batch number", self.batch_number)
            my_print("Input dimension", self.input_dimension)
            my_print("Output dimension", self.output_dimension)
            my_print("Learning rate", self.lr)

    @staticmethod
    def update_display(cur, total, cur_epoch, cost, acc):
        stdout.write(
            "\r" + BColors.OKBLUE + "Progress: %d/%d (%.2f %%)" % (cur, total, float(cur) / total * 100) +
            "\t" + "Current epoch: %d" % cur_epoch +
            "\t" + "Cost: %.2f" % cost +
            "\t" + "ACC: %.2f" % acc
        )
        pass

    def _train_data(self, param):
        if len(self.train_input_data) == 0:
            self._load_train_data()
        if self.model is None:
            self.create_model()

        epoch = 1
        train_segment = 400
        train_segment = int(floor(train_segment / self.batch_number) * self.batch_number)
        train_time = train_segment
        if len(param) > 1 and param[1].isdigit():
            train_time *= int(param[1])
        train_batch_x = range(self.batch_number)
        train_batch_y = range(self.batch_number)
        i = 0
        m = 0
        while i < train_time:
            for j in range(self.batch_number):
                num = random.randint(0, train_segment)

                train_batch_x[j], train_batch_y[j] = self.get_one_data(num)

            self.model.train_one(train_batch_x, train_batch_y)
            i += self.batch_number
            m += self.batch_number
            if m > train_segment:
                # one epoch
                epoch += 1
                m = 0
            cost = self.model.cost_function(train_batch_x, train_batch_y)
            acc = self._test(train_segment)
            TestBench.update_display(cur=i,
                                     total=train_time,
                                     cur_epoch=epoch,
                                     cost=cost,
                                     acc=acc
                                     )
        stdout.write("\n")

    def get_one_data(self, num):
        type_index = self.ans_type.index(str(self.train_answer_data[num][1].split('\n')[0]))
        y = [0] * 48
        y[type_index] = 1
        return self.train_input_data[num][1:70], y
        pass

    def _run(self, param):
        pass

    def _output_result(self, param):
        pass

    def _set(self, param):
        set_dispatcher = {
            "layer": self._set_layer,
            "batch": self._set_batch
        }
        if len(param) > 2:
            set_command = set_dispatcher.get(param[1], lambda: "Set parameter undefined. Do nothing")
            set_command(param)
            return
        else:
            print BColors.WARNING + "Set parameter without any argument. Do nothing." + BColors.ENDC
            return

    def _set_layer(self, param):
        start_with = 2
        layer = []
        if self.model is not None:
            if len(param) > 3 and param[2] == "force":
                start_with = 3
            else:
                x = raw_input(BColors.WARNING + "Set layer number without save parameters? [y/N]" + BColors.ENDC)
                if x == "y" or x == 'Y':
                    pass
                    self.model = None
                else:
                    return

        for i in range(start_with, len(param)):
            if param[i].isdigit():
                d = int(param[i])
                layer.append(d)

        self.layer = layer
        my_print("Layer are set to " + self.layer.__str__())

    def _set_batch(self, param):
        start_with = 2
        if self.model is not None:
            if len(param) > 3 and param[2] == "force":
                start_with = 3
                pass
            else:
                x = raw_input(BColors.WARNING + "Set layer number without save parameters? [y/N]" + BColors.ENDC)
                if x == "y" or x == 'Y':
                    self.model = None
                    pass
                else:
                    return

        if param[start_with].isdigit():
            d = int(param[start_with])
            self.batch_number = d
            my_print("Batch number is set to %d" % self.batch_number)

    def _load_parameter(self, param):
        filename_w = None
        filename_b = None
        if param[1] == "id":
            filename_w = "parameter_W_%s.txt" % param[2]
            filename_b = "parameter_B_%s.txt" % param[2]
            filename_i = "parameter_I_%s.txt" % param[2]
        my_print("Load parameters from parameter_X_%s.txt" % param[2])

        # TODO: Keep parameters of two test consistent

        try:
            if self.model is None:
                i_parm_data = file(filename_i, 'rb')
                self.create_model()

            w_parm_data = file(filename_w, 'rb')
            b_parm_data = file(filename_b, 'rb')
            w_parm = cPickle.load(w_parm_data)
            b_parm = cPickle.load(b_parm_data)
            self.model.load_parm(w_parm, b_parm)
        except IOError:
            my_print(BColors.FAIL + "File not found. Do nothing." + BColors.ENDC)
            return

    def _quit(self, param):
        if self.modified is True:
            if len(param) > 1 and param[1] == "force":
                exit()
            else:
                x = raw_input(BColors.WARNING + "Exit without save parameters? [y/N]" + BColors.ENDC)
                if x == "y" or x == 'Y':
                    exit()
                else:
                    return
        exit()

    def _load_train_data(self):
        my_print("Load training input data from %s" % self.train_file)
        for line in open(self.train_file, 'r'):
            input_x = line.split()
            input_x = [TestBench.float_convert(i) for i in input_x]
            self.train_input_data.append(input_x)

        my_print("Load training answer data from %s" % self.answer_map_file)
        for line in open(self.answer_map_file, 'r'):
            ans_x = line.split(',')
            self.train_answer_data.append(ans_x)

    def _load_test_data(self):
        for line in open(self.test_input_file, 'r'):
            test_x = line.split()
            test_x = [TestBench.float_convert(x) for x in test_x]
            self.test_data.append(test_x)

    def _output_test_file(self):
        my_print("Writing test answer data to %s", self.answer_map_file)
        test_stream = open(self.test_output_file, 'w')
        test_c = MAP()
        y = None
        test_stream.write('Id,Prediction\n')
        self.model.load_parm(self.W_parm, self.B_parm)

        for i in range(len(self.test_data)):
            ya = self.model.y_evaluated_function([self.test_data[i][1:self.input_dimension]], y)
            value = str(
                (self.test_data[i][0], test_c.map(ya, ))
            )
            test_stream.write(value)
            test_stream.write('\n')
        # print test.W_array[0].get_value()
        return

    def _output_training_progress(self):
        pass

    def _test(self, training_segment):
        c = MAP()
        err = 0
        _1 = len(self.train_input_data) - training_segment
        for m in range(_1):
            # print self.train_input_data[training_segment + m][1:70]
            ya = self.model.y_evaluated_function(
                [self.train_input_data[training_segment + m][1:70]],
                None
            )[0]
            if [c.map(ya)] != [str(self.train_answer_data[training_segment + m][1].split('\n')[0])]:
                err += 1
            else:
                pass
                # print [c.map(Ya)]
                # print [str(ans[m][1].split('\n')[0])]
        return 1.0 - err / float(_1)

    def run(self, batch):
        training_segment = 1000000

        batch_number = batch * 1000
        X = None
        Y = None
        i = 0
        acc = 0.0
        W_new = []
        B_new = []
        c = MAP()
        while True:
            X = []
            yy = []
            for k in range(batch_number):
                num = randrange(0, training_segment)
                if i >= 1000000:  # i >= 1124823:
                    i = 0
                    err = 0.0
                    for m in range(124823):
                        Ya = self.model.y_evaluated_function([self.train[1000000 + m][1:70]], Y)[0]
                        if [c.map(Ya)] != [str(self.ans[1000000 + m][1].split('\n')[0])]:
                            err += 1
                            # print [c.map(Ya)]
                            # print [str(ans[m][1].split('\n')[0])]
                    acc = 1.0 - err / 124823.0
                    # print err
                    print acc
                typeidx = self.anstype.index(str(self.ans[num][1].split('\n')[0]))
                y = [0] * 48
                y[typeidx] = 1
                yy.append(y)
                X.append(self.train[num][1:70])
                i += 1
            Y = yy
            if i % 10000 == 0:
                print i
                # print test.y_evaluated_function(X,Y)
                # print [test.W_array[0].get_value(),test.W_array[1].get_value()]
            self.model.train_one(X, Y)

    def save_parameter(self):
        f = file('parameter_W.txt', 'wb')
        cPickle.dump(self.model.W, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        f = file('parameter_B.txt', 'wb')
        cPickle.dump(self.model.B, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def get_correctness_ratio(correct, total):
        return correct / total

    @staticmethod
    def float_convert(num):
        try:
            return float(num)
        except ValueError:
            return num

'''
    Test part
'''




'''
train = []
ans = []
for line in train_data:
    input_x = line.split()
    input_x = [float_convert(i) for i in input_x]
    train.append(input_x)
for line in ans_data:
    ans_x = line.split(',')
    ans.append(ans_x)
'''

'''

'''
'''
    Test part
'''

'''
c = MAP()
for i in range(400):
    #print [train[i][1:70]]
    Ya = test.y_evaluated_function([train[i][1:70]], Y)[0]
    print c.map(Ya)
'''

# print test.train_one(X, Y)

test = TestBench()
