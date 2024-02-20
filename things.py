import random as ran
import math
import time

try:
    from time import sleep
except Exception as e:
    pass

from cmu_graphics import *
import queue
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

try:
    app.width, app.height = 400, 400
except Exception as e:
    print(f"App.width is readonly\nApp.height is readonly\n")

app.setMaxShapeCount(10_000)
app.background = gradient("black",rgb(229,229,229))
app._can_run = True

class Adam:
    def __init__(self, gradients, parameters):
        self.parameters = parameters
        self.gradients = gradients
        self.m = []
        self.v = []
        self.m_hat = []
        self.y_hat = []

    def minimize(self, gradients, parameters, time_passed):
        for i, grad in enumerate(gradients):
            if isinstance(grad, list):
                self.m[i] += grad[i]
                self.v[i] += grad[i] ** 2
            else:
                self.m[i] += grad
                self.v[i] += grad ** 2

            self.m_hat.append(self.m[i] / (1 - 1 / time_passed))
            self.y_hat.append(self.v[i] / (1 - 1 / time_passed))


        for param, mhat, vhat in zip(parameters, self.m_hat, self.v_hat):
            param -= mhat / math.sqrt(vhat)

        return parameters

class Predator:
    def __init__(self, board, color, id_num):
        self.board = board
        self.color = color

        available_tiles = []
        for row in range(len(self.board)):
            for col in range(len(self.board)):
                tile = self.board[row][col]
                if tile.fill not in ("red", "forestGreen", "blue"):
                    available_tiles.append(board[row][col])

        self.body = ran.choice(available_tiles)
        self.body.fill = self.color
        self.id_num = id_num
        self.id = Label(self.id_num,self.body.centerX,self.body.centerY,size=(self.body.width+self.body.height)//2)
        self.health = 100

        self.row, self.col = self.get_position()[0], self.get_position()[1]

        self.num_threads = 4
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        self.lock = Lock()

        self.memory = [0]

        self.inputs = self.rec_flatten([i + self.one_hot_encode() + [self.get_position()[0] + self.get_position()[1]] + [self.health] + self.memory for i in self.get_positions()])

        self.brain1 = [ran.uniform(-1, 1) for _ in range(len(self.inputs))]
        self.brain2 = [ran.uniform(-1, 1) for _ in range(len(self.inputs))]
        self.brain3 = [ran.uniform(-1, 1) for _ in range(len(self.inputs))]
        self.brain4 = [ran.uniform(-1, 1) for _ in range(len(self.inputs))]

        self.bias1 = [ran.uniform(-1, 1)  for _ in range(len(self.inputs))]
        self.bias2 = [ran.uniform(-1, 1)  for _ in range(len(self.inputs))]
        self.bias3 = [ran.uniform(-1, 1)  for _ in range(len(self.inputs))]
        self.bias4 = [ran.uniform(-1, 1)  for _ in range(len(self.inputs))]

        #clear inputs
        self.inputs = []

        self.learning_rate1 = ran.uniform(0,1)
        self.learning_rate2 = ran.uniform(0,1)
        self.learning_rate3 = ran.uniform(0,1)
        self.learning_rate4 = ran.uniform(0,1)

        self.sigmoid = lambda x: 1 / (1 + math.exp(-x))
        self.sigmoid_deriv = lambda x: self.sigmoid(x) * (1 - self.sigmoid(x))

        self.relu = lambda x: max(0.0, x)
        self.relu_deriv = lambda x: 1 if x > 0 else 0

        self.tanh = lambda x: math.tanh(x)
        self.tanh_deriv = lambda x: 1 - math.tanh(x)**2

        self.swish = lambda x: x * 1/(1+math.exp(-x))
        self.swish_deriv = lambda x: (1/(1+math.exp(-x))) + x * 1/(1+math.exp(-x)) * (1-1/(1+math.exp(-x)))

        self.softplus = lambda x: math.log(math.exp(x) + 1)
        self.softplus_deriv = lambda x: self.sigmoid(x)

        self.max_tanh = lambda x: max(0.0, self.tanh(x))
        self.max_tanh_deriv = lambda x: 1 if x > 0 else math.tanh(x)

        self.activations = [self.sigmoid, self.relu, self.tanh, self.swish, self.max_tanh]
        self.activations_derivs = [self.sigmoid_deriv, self.relu_deriv, self.tanh_deriv, self.swish_deriv, self.max_tanh_deriv]

        self.ran_act1 = ran.choice(self.activations)
        self.ran_act2 = ran.choice(self.activations)
        self.ran_act3 = ran.choice(self.activations)
        self.ran_act4 = ran.choice(self.activations)

        self.ran_act1_idx = self.activations.index(self.ran_act1)
        self.ran_act2_idx = self.activations.index(self.ran_act2)
        self.ran_act3_idx = self.activations.index(self.ran_act3)
        self.ran_act4_idx = self.activations.index(self.ran_act4)

        self.ran_act1_deriv = self.activations_derivs[self.ran_act1_idx]
        self.ran_act2_deriv = self.activations_derivs[self.ran_act2_idx]
        self.ran_act3_deriv = self.activations_derivs[self.ran_act3_idx]
        self.ran_act4_deriv = self.activations_derivs[self.ran_act4_idx]

    def softmax(self, logits):
        exp_logits = [math.exp(logit) for logit in logits]
        sum_exp_logits = sum(exp_logits)
        softmax_probs = [logit / sum_exp_logits for logit in exp_logits]
        return softmax_probs

    def flatten(self, lst):
        return [item for sublst in lst for item in sublst]

    def rec_flatten(self, arr):
        result = []
        for item in arr:
            if isinstance(item, list):
                result.extend(self.rec_flatten(item))
            else:
                result.append(item)
        return result

    def l2_norm(self, lst):
        res = 0

        for item in lst:
            res += item ** 2

        return math.sqrt(res)

    def norm_data(self, data):
        norms = [self.l2_norm(row) for row in data]
        return [[x / norm for x in row] for row, norm in zip(data, norms)]

    def reshape(self, flat_arr, original_arr):
        def reconstruct(arr_structure):
            if isinstance(arr_structure, list):
                return [reconstruct(sublist) for sublist in arr_structure]
            else:
                return flat_arr.pop(0)

        return reconstruct(original_arr)

    def pad(self, inputs, weights, biases):
        inputs = self.rec_flatten(inputs)
        weights = self.rec_flatten(weights)
        biases = self.rec_flatten(biases)

        print(f"inputs: {len(inputs)}, weights: {len(weights)}, biases: {len(biases)}")

        max_length = max(len(inputs), len(weights), len(biases))

        input_padding = [0] * (max_length - len(inputs))
        weight_padding = [0] * (max_length - len(weights))
        bias_padding = [0] * (max_length - len(biases))

        inputs += input_padding
        weights += weight_padding
        biases += bias_padding

        return inputs, weights, biases

    def dotp(self, m1, m2):
        if len(m1) > len(m2):
            largerM = m1
            shorterM = m2
        else:
            largerM = m2
            shorterM = m1

        res = 0
        for lM in largerM:
            for sM in shorterM:
                res += lM * sM

        return res

    def fcm_1d(self, inputs, weights, biases):
        res = 0
        for i in inputs:
            for idx, w in enumerate(weights):
                res += (i*w)+biases[idx]
        return res

    def fcm_2d(self, inputs, weights, biases):
        c = []
        for i in range(len(inputs)):
            row = []
            for j in range(len(weights)):
                element = (inputs[i][j] * weights[i][j]) + biases[i][j]
                row.append(element)

            c.append(sum(row))

        return self.flatten(c)

    def get_position(self):
        return [(rounded(self.body.centerX) // CELL_SIZE) % CELL_SIZE, (rounded(self.body.centerY) // CELL_SIZE) % CELL_SIZE]

    def get_positions(self):
        self_pos = []
        agents_pos = []
        prey_pos = []
        food_pos = []
        empty_pos = []

        for row in range(len(self.board)):
            for col in range(len(self.board)):
                if self.board[row][col] == self.body:
                    self_pos.append(row + 1)
                    self_pos.append(col + 1)
                elif self.board[row][col].fill == self.color:
                    agents_pos.append(row + 1)
                    agents_pos.append(col + 1)
                elif self.board[row][col] == "forestGreen":
                    prey_pos.append(row + 1)
                    prey_pos.append(col + 1)
                elif self.board[row][col].fill == "blue":
                    food_pos.append(row + 1)
                    food_pos.append(col + 1)
                elif self.board[row][col].fill == None:
                    empty_pos.append(row + 1)
                    empty_pos.append(col + 1)

        return self_pos, agents_pos, prey_pos, food_pos, empty_pos

    def one_hot_encode(self):
        encoded = []
        for row in range(len(self.board)):
            for col in range(len(self.board)):
                if self.board[row][col] == self.board[self.get_position()[0]][self.get_position()[1]]:
                    encoded.append(1)
                elif self.board[row][col].fill == self.color:
                    encoded.append(3)
                elif self.board[row][col].fill == "forestGreen":
                    encoded.append(2)
                elif self.board[row][col].fill == "blue":
                    encoded.append(4)
                elif self.board[row][col].fill == None:
                    encoded.append(0)

        return encoded

    def calcFitness(self):
        health_fitness = self.health / 100

        self_pos, agents_pos, prey_pos, food_pos, empty_pos = self.get_positions()

        prey_distance = math.sqrt((app.width**2)+(app.height**2))
        for i in range(0, len(prey_pos), 2):
            distance = math.sqrt((self_pos[0] - prey_pos[i]) ** 2 + (self_pos[1] - prey_pos[i + 1]) ** 2)
            prey_distance = min(prey_distance, distance)
        prey_fitness = 1 - (prey_distance / (len(self.board) * math.sqrt(2)))

        lifespan_fitness = len(self.memory) / len(self.memory) * 10

        fitness = 1 / (1 + sum([health_fitness, prey_fitness, lifespan_fitness]))

        return fitness

    def start_threads(self):
        future = self.executor.submit(self.move)
        future.result()

    def move(self):
        self.lock.acquire()

        inputs = self.rec_flatten([i + self.one_hot_encode() + [self.get_position()[0] + self.get_position()[1]] + [self.health] + self.memory for i in self.get_positions()])
        self.decision = self.get_decision(inputs)
        print(f"Pred: id:{self.id_num} health:{self.health}, action:{self.decision}")

        if self.decision == "up":
            if self.row > 0:
                new_tile = self.board[self.row - 1][self.col]
                self.eat(new_tile)
                if new_tile.fill == None:
                    self.board[self.row][self.col].fill = None
                    self.body = new_tile
                    self.row -= 1
                    new_tile.fill = self.color

        if self.decision == "down":
            if self.row < len(self.board) - 1:
                new_tile = self.board[self.row + 1][self.col]
                self.eat(new_tile)
                if new_tile.fill == None:
                    self.board[self.row][self.col].fill = None
                    self.body = new_tile
                    self.row += 1
                    new_tile.fill = self.color

        if self.decision == "left":
            if self.col > 0:
                new_tile = self.board[self.row][self.col - 1]
                self.eat(new_tile)
                if new_tile.fill == None:
                    self.board[self.row][self.col].fill = None
                    self.body = new_tile
                    self.col -= 1
                    new_tile.fill = self.color

        if self.decision == "right":
            if self.col < len(self.board[0]) - 1:
                new_tile = self.board[self.row][self.col + 1]
                self.eat(new_tile)
                if new_tile.fill == None:
                    self.board[self.row][self.col].fill = None
                    self.body = new_tile
                    self.col += 1
                    new_tile.fill = self.color

        if self.decision == "None":
            if (self.row and self.col):
                new_tile = self.board[self.row][self.col]
                self.eat(new_tile)
                if new_tile.fill == None:
                    self.board[self.row][self.col].fill = None
                    self.body = new_tile
                    self.row
                    self.col
                    new_tile.fill = self.color

        self.health -= 1
        self.id.centerX, self.id.centerY = self.body.centerX, self.body.centerY

        self.backprop()

        self.lock.release()

    def eat(self, tile):
        if tile.fill == "forestGreen":
            for prey in preys:
                if tile == prey.body:
                    prey.health = 0
                    prey.executor.shutdown()
                    app.deadPreyators.append(prey)
                    preys.remove(prey)
                    app.group.remove(prey.body)
                    app.group.remove(prey.id)
                    tile.fill = None
                    tile.border = "black"
                    print(f"Prey {prey.id_num} has been eaten and removed.")
                    self.health += 100
                    app._can_run = False

    def get_decision(self, inputs):
        output1 = self.ran_act1(self.relu(self.fcm_1d(inputs, self.brain1, self.bias1)))
        output2 = self.ran_act2(self.relu(self.fcm_1d(inputs, self.brain2, self.bias2)))
        output3 = self.ran_act3(self.relu(self.fcm_1d(inputs, self.brain3, self.brain3)))
        output4 = self.ran_act4(self.relu(self.fcm_1d(inputs, self.brain4, self.brain4)))

        #xor logic gate
        output1 = self.sigmoid(output1)
        output2 = self.sigmoid(output2)
        output3 = self.sigmoid(output3)
        output4 = self.sigmoid(output4)

        self.o1 = 0
        self.o2 = 0
        self.o3 = 0
        self.o4 = 0

        #if output is greater than 0.5 we use
        if output1 >= 0.5:
            self.o1 = 1
        if output2 >= 0.5:
            self.o2 = 1
        if output3 >= 0.5:
            self.o3 = 1
        if output4 >= 0.5:
            self.o4 = 1

        output1 = math.tanh(sum([self.o1 * self.fcm_1d(self.brain1, self.memory, self.bias1)]))
        output2 = math.tanh(sum([self.o2 * self.fcm_1d(self.brain2, self.memory, self.bias2)]))
        output3 = math.tanh(sum([self.o3 * self.fcm_1d(self.brain3, self.memory, self.bias3)]))
        output4 = math.tanh(sum([self.o4 * self.fcm_1d(self.brain4, self.memory, self.bias4)]))

        """
        output1 = math.tanh(self.o1 * self.dotp(self.memory, self.brain1))
        output2 = math.tanh(self.o2 * self.dotp(self.memory, self.brain2))
        output3 = math.tanh(self.o3 * self.dotp(self.memory, self.brain3))
        output4 = math.tanh(self.o4 * self.dotp(self.memory, self.brain4))
        """

        #xor logic gate
        so1 = self.sigmoid(output1)
        so2 = self.sigmoid(output2)
        so3 = self.sigmoid(output3)
        so4 = self.sigmoid(output4)

        self.so1 = 0
        self.so2 = 0
        self.so3 = 0
        self.so4 = 0

        #if output is greater then 0.5 we use
        if so1 >= 0.5:
            self.so1 = 1
        if so2 >= 0.5:
            self.so2 = 1
        if so3 >= 0.5:
            self.so3 = 1
        if so4 >= 0.5:
            self.so4 = 1

        outputs = [direction for output,direction in zip([self.so1,self.so2,self.so3,self.so4],[0.5,-1,-0.5,1]) if output == 1]
        if len(outputs) > 1:
            outputs = [ran.choice(outputs)]

        #print(f"id:{self.id_num},o:{outputs}")
        match outputs:
            case [0.5]:
                self.memory.append(self.o1)
                if self.row > 0 and self.board[self.row-1][self.col].fill != "black":
                    return "up"
            case [-1]:
                self.memory.append(self.o2)
                if self.col < len(self.board[0])-1 and self.board[self.row][self.col+1].fill != "black":
                    return "right"
            case [-0.5]:
                self.memory.append(self.o3)
                if self.row < len(self.board)-1 and self.board[self.row+1][self.col].fill != "black":
                    return "down"
            case [1]:
                self.memory.append(self.o4)
                if self.col > 0 and self.board[self.row][self.col-1].fill != "black":
                    return "left"
            case [0]:
                return "None"

    def custom_loss(self, y, o):
        epsilon = 1e-10  # Small constant to avoid math domain errors
        o = max(epsilon, min(1 - epsilon, o))  # Clip 'o' to avoid values too close to 0 or 1

        return -y * math.log(o) - (1 - y) * math.log(1 - o)

    def backprop(self):
        self.y = self.calcFitness()

        loss1 = self.custom_loss(self.y, self.o1)
        loss2 = self.custom_loss(self.y, self.o2)
        loss3 = self.custom_loss(self.y, self.o3)
        loss4 = self.custom_loss(self.y, self.o4)

        od1 = self.relu_deriv(self.o1)
        od2 = self.relu_deriv(self.o2)
        od3 = self.relu_deriv(self.o3)
        od4 = self.relu_deriv(self.o4)

        for w in range(len(self.brain1)):
            w -= loss1 * self.o1 * od1 * self.learning_rate1

        for w in self.brain2:
            w -= loss2 * self.o2 * od2 * self.learning_rate2

        for w in self.brain3:
            w -= loss3 * self.o3 * od3 * self.learning_rate3

        for w in self.brain4:
            w -= loss4 * self.o4 * od4 * self.learning_rate4

class Prey:
    def __init__(self, board, color, id_num):
        self.board = board
        self.color = color

        available_tiles = []
        for row in range(len(self.board)):
            for col in range(len(self.board)):
                tile = self.board[row][col]
                if tile.fill not in ("red", "forestGreen", "blue"):
                    available_tiles.append(self.board[row][col])

        self.body = ran.choice(available_tiles)
        self.body.fill = self.color
        self.id_num = id_num
        self.id = Label(self.id_num,self.body.centerX,self.body.centerY,size=(self.body.width+self.body.height)//2)
        self.health = 100

        self.row, self.col = self.get_position()[0], self.get_position()[1]

        self.num_threads = 4
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        self.lock = Lock()

        self.memory = [0]

        self.inputs = self.rec_flatten([i + self.one_hot_encode() + [self.get_position()[0] + self.get_position()[1]] + [self.health] + self.memory for i in self.get_positions()])

        self.brain1 = [ran.uniform(-1, 1) for _ in range(len(self.inputs))]
        self.brain2 = [ran.uniform(-1, 1) for _ in range(len(self.inputs))]
        self.brain3 = [ran.uniform(-1, 1) for _ in range(len(self.inputs))]
        self.brain4 = [ran.uniform(-1, 1) for _ in range(len(self.inputs))]

        self.bias1 = [ran.uniform(-1, 1) for _ in range(len(self.inputs))]
        self.bias2 = [ran.uniform(-1, 1) for _ in range(len(self.inputs))]
        self.bias3 = [ran.uniform(-1, 1) for _ in range(len(self.inputs))]
        self.bias4 = [ran.uniform(-1, 1) for _ in range(len(self.inputs))]

        #clear inputs
        self.inputs = []

        self.learning_rate1 = ran.uniform(0,1)
        self.learning_rate2 = ran.uniform(0,1)
        self.learning_rate3 = ran.uniform(0,1)
        self.learning_rate4 = ran.uniform(0,1)

        self.sigmoid = lambda x: 1 / (1 + math.exp(-x))
        self.sigmoid_deriv = lambda x: self.sigmoid(x) * (1 - self.sigmoid(x))

        self.relu = lambda x: max(0.0, x)
        self.relu_deriv = lambda x: 1 if x > 0 else 0

        self.tanh = lambda x: math.tanh(x)
        self.tanh_deriv = lambda x: 1 - math.tanh(x)**2

        self.swish = lambda x: x * 1/(1+math.exp(-x))
        self.swish_deriv = lambda x: (1/(1+math.exp(-x))) + x * 1/(1+math.exp(-x)) * (1-1/(1+math.exp(-x)))

        self.softplus = lambda x: math.log(math.exp(x) + 1)
        self.softplus_deriv = lambda x: self.sigmoid(x)

        self.max_tanh = lambda x: max(0.0, self.tanh(x))
        self.max_tanh_deriv = lambda x: 1 if x > 0 else math.tanh(x)

        self.activations = [self.sigmoid, self.relu, self.tanh, self.swish, self.max_tanh]
        self.activations_derivs = [self.sigmoid_deriv, self.relu_deriv, self.tanh_deriv, self.swish_deriv, self.max_tanh_deriv]

        self.ran_act1 = ran.choice(self.activations)
        self.ran_act2 = ran.choice(self.activations)
        self.ran_act3 = ran.choice(self.activations)
        self.ran_act4 = ran.choice(self.activations)

        self.ran_act1_idx = self.activations.index(self.ran_act1)
        self.ran_act2_idx = self.activations.index(self.ran_act2)
        self.ran_act3_idx = self.activations.index(self.ran_act3)
        self.ran_act4_idx = self.activations.index(self.ran_act4)

        self.ran_act1_deriv = self.activations_derivs[self.ran_act1_idx]
        self.ran_act2_deriv = self.activations_derivs[self.ran_act2_idx]
        self.ran_act3_deriv = self.activations_derivs[self.ran_act3_idx]
        self.ran_act4_deriv = self.activations_derivs[self.ran_act4_idx]

    def softmax(self, logits):
        exp_logits = [math.exp(logit) for logit in logits]
        sum_exp_logits = sum(exp_logits)
        softmax_probs = [logit / sum_exp_logits for logit in exp_logits]
        return softmax_probs

    def flatten(self, lst):
        return [item for sublst in lst for item in sublst]

    def rec_flatten(self, arr):
        result = []
        for item in arr:
            if isinstance(item, list):
                result.extend(self.rec_flatten(item))
            else:
                result.append(item)
        return result

    def l2_norm(self, lst):
        res = 0

        for item in lst:
            res += item ** 2

        return math.sqrt(res)

    def norm_data(self, data):
        norms = [self.l2_norm(row) for row in data]
        return [[x / norm for x in row] for row, norm in zip(data, norms)]

    def reshape(self, flat_arr, original_arr):
        def reconstruct(arr_structure):
            if isinstance(arr_structure, list):
                return [reconstruct(sublist) for sublist in arr_structure]
            else:
                if flat_arr:
                    return flat_arr.pop(0)
                else:
                    raise ValueError("Not enough elements in flat_arr to reshape the original structure.")

        return reconstruct(original_arr)

    def pad(self, inputs, weights, biases):
        inputs = self.rec_flatten(inputs)
        weights = self.rec_flatten(weights)
        biases = self.rec_flatten(biases)

        max_length = max(len(inputs), len(weights), len(biases))

        input_padding = [0] * (max_length - len(inputs))
        weight_padding = [0] * (max_length - len(weights))
        bias_padding = [0] * (max_length - len(biases))

        inputs += input_padding
        weights += weight_padding
        biases += bias_padding

        return inputs, weights, biases

    def dotp(self, m1, m2):
        if len(m1) > len(m2):
            largerM = m1
            shorterM = m2
        else:
            largerM = m2
            shorterM = m1

        res = 0
        for lM in largerM:
            for sM in shorterM:
                res += lM * sM

        return res

    def fcm_1d(self, inputs, weights, biases):
        res = 0
        for i in inputs:
            for idx, w in enumerate(weights):
                res += (i*w)+biases[idx]
        return res

    def fcm_2d(self, inputs, weights, biases):
        c = []
        for i in range(len(inputs)):
            row = []
            for j in range(len(weights)):
                element = (inputs[i][j] * weights[i][j]) + biases[i][j]
                row.append(element)

            c.append(sum(row))

        return self.flatten(c)

    def get_position(self):
        return [(rounded(self.body.centerX) // CELL_SIZE) % CELL_SIZE, (rounded(self.body.centerY) // CELL_SIZE) % CELL_SIZE]

    def get_positions(self):
        self_pos = []
        agents_pos = []
        pred_pos = []
        food_pos = []
        empty_pos = []

        for row in range(len(self.board)):
            for col in range(len(self.board)):
                if self.board[row][col] == self.body:
                    self_pos.append(row + 1)
                    self_pos.append(col + 1)
                elif self.board[row][col].fill == self.color:
                    agents_pos.append(row + 1)
                    agents_pos.append(col + 1)
                elif self.board[row][col] == "red":
                    pred_pos.append(row + 1)
                    pred_pos.append(col + 1)
                elif self.board[row][col].fill == "blue":
                    food_pos.append(row + 1)
                    food_pos.append(col + 1)
                elif self.board[row][col].fill == None:
                    empty_pos.append(row + 1)
                    empty_pos.append(col + 1)

        return self_pos, agents_pos, pred_pos, food_pos, empty_pos

    def one_hot_encode(self):
        encoded = []
        for row in range(len(self.board)):
            for col in range(len(self.board)):
                if self.board[row][col] == self.board[self.get_position()[0]][self.get_position()[1]]:
                    encoded.append(2)
                elif self.board[row][col].fill == self.color:
                    encoded.append(4)
                elif self.board[row][col].fill == "red":
                    encoded.append(3)
                elif self.board[row][col].fill == "blue":
                    encoded.append(1)
                elif self.board[row][col].fill == None:
                    encoded.append(0)

        return encoded

    def calcFitness(self):
        health_fitness = self.health / 100

        self_pos, agents_pos, pred_pos, food_pos, empty_pos = self.get_positions()

        pred_distance = math.sqrt((app.width**2)+(app.height**2))
        for i in range(0, len(pred_pos), 2):
            distance = math.sqrt((self_pos[0] - pred_pos[i]) ** 2 + (self_pos[1] - pred_pos[i + 1]) ** 2)
            pred_distance = min(pred_distance, distance)
        pred_fitness = 1 - (pred_distance / (len(self.board) * math.sqrt(2)))

        lifespan_fitness = len(self.memory) / len(self.memory) * 10

        fitness = 1 / (1 + sum([health_fitness, pred_fitness, lifespan_fitness]))

        return fitness

    def start_threads(self):
        future = self.executor.submit(self.move)
        future.result()

    def move(self):
        self.lock.acquire()

        inputs = self.rec_flatten([i + self.one_hot_encode() + [self.get_position()[0] + self.get_position()[1]] + [self.health] + self.memory for i in self.get_positions()])
        self.decision = self.get_decision(inputs)
        print(f"Prey: id:{self.id_num} health:{self.health}, action:{self.decision}")

        if self.decision == "up":
            if self.row > 0:
                new_tile = self.board[self.row - 1][self.col]
                self.eat(new_tile)
                if new_tile.fill == None:
                    self.board[self.row][self.col].fill = None
                    self.body = new_tile
                    self.row -= 1
                    new_tile.fill = self.color

        if self.decision == "down":
            if self.row < len(self.board) - 1:
                new_tile = self.board[self.row + 1][self.col]
                self.eat(new_tile)
                if new_tile.fill == None:
                    self.board[self.row][self.col].fill = None
                    self.body = new_tile
                    self.row += 1
                    new_tile.fill = self.color

        if self.decision == "left":
            if self.col > 0:
                new_tile = self.board[self.row][self.col - 1]
                self.eat(new_tile)
                if new_tile.fill == None:
                    self.board[self.row][self.col].fill = None
                    self.body = new_tile
                    self.col -= 1
                    new_tile.fill = self.color

        if self.decision == "right":
            if self.col < len(self.board[0]) - 1:
                new_tile = self.board[self.row][self.col + 1]
                self.eat(new_tile)
                if new_tile.fill == None:
                    self.board[self.row][self.col].fill = None
                    self.body = new_tile
                    self.col += 1
                    new_tile.fill = self.color

        if self.decision == "None":
            if (self.row and self.col):
                new_tile = self.board[self.row][self.col]
                self.eat(new_tile)
                if new_tile.fill == None:
                    self.board[self.row][self.col].fill = None
                    self.body = new_tile
                    self.row
                    self.col
                    new_tile.fill = self.color

        self.health -= 1
        self.id.centerX, self.id.centerY = self.body.centerX, self.body.centerY

        self.backprop()

        self.lock.release()

    def eat(self, tile):
        if tile.fill == "blue":
            tile.fill = None
            tile.border = "black"
            self.health += 100

            tile = ran.choice(ran.choice(board))
            while tile.fill != None:
                tile = ran.choice(ran.choice(board))
            else:
                tile.fill = "blue"

    def get_decision(self, inputs):
        output1 = self.ran_act1(self.relu(self.fcm_1d(inputs, self.brain1, self.bias1)))
        output2 = self.ran_act2(self.relu(self.fcm_1d(inputs, self.brain2, self.bias2)))
        output3 = self.ran_act3(self.relu(self.fcm_1d(inputs, self.brain3, self.bias3)))
        output4 = self.ran_act4(self.relu(self.fcm_1d(inputs, self.brain4, self.bias4)))

        #xor logic gate
        output1 = self.sigmoid(output1)
        output2 = self.sigmoid(output2)
        output3 = self.sigmoid(output3)
        output4 = self.sigmoid(output4)

        self.o1 = 0
        self.o2 = 0
        self.o3 = 0
        self.o4 = 0

        #if output is greater than 0.5 we use
        if output1 >= 0.5:
            self.o1 = 1
        if output2 >= 0.5:
            self.o2 = 1
        if output3 >= 0.5:
            self.o3 = 1
        if output4 >= 0.5:
            self.o4 = 1

        output1 = math.tanh(sum([self.o1 * self.fcm_1d(self.brain1, self.memory, self.bias1)]))
        output2 = math.tanh(sum([self.o2 * self.fcm_1d(self.brain2, self.memory, self.bias2)]))
        output3 = math.tanh(sum([self.o3 * self.fcm_1d(self.brain3, self.memory, self.bias3)]))
        output4 = math.tanh(sum([self.o4 * self.fcm_1d(self.brain4, self.memory, self.bias4)]))

        """
        output1 = math.tanh(self.o1 * self.dotp(self.memory, self.brain1))
        output2 = math.tanh(self.o2 * self.dotp(self.memory, self.brain2))
        output3 = math.tanh(self.o3 * self.dotp(self.memory, self.brain3))
        output4 = math.tanh(self.o4 * self.dotp(self.memory, self.brain4))
        """

        #xor logic gate
        so1 = self.sigmoid(output1)
        so2 = self.sigmoid(output2)
        so3 = self.sigmoid(output3)
        so4 = self.sigmoid(output4)

        self.so1 = 0
        self.so2 = 0
        self.so3 = 0
        self.so4 = 0

        #if output is greater then 0.5 we use
        if so1 >= 0.5:
            self.so1 = 1
        if so2 >= 0.5:
            self.so2 = 1
        if so3 >= 0.5:
            self.so3 = 1
        if so4 >= 0.5:
            self.so4 = 1

        outputs = [direction for output,direction in zip([self.so1,self.so2,self.so3,self.so4],[0.5,-1,-0.5,1]) if output == 1]
        if len(outputs) > 1:
            outputs = [ran.choice(outputs)]

        #print(f"id:{self.id_num},o:{outputs}")
        match outputs:
            case [0.5]:
                self.memory.append(self.o1)
                if self.row > 0 and self.board[self.row-1][self.col].fill != "black":
                    return "up"
            case [-1]:
                self.memory.append(self.o2)
                if self.col < len(self.board[0])-1 and self.board[self.row][self.col+1].fill != "black":
                    return "right"
            case [-0.5]:
                self.memory.append(self.o3)
                if self.row < len(self.board)-1 and self.board[self.row+1][self.col].fill != "black":
                    return "down"
            case [1]:
                self.memory.append(self.o4)
                if self.col > 0 and self.board[self.row][self.col-1].fill != "black":
                    return "left"
            case [0]:
                return "None"

    def custom_loss(self, y, o):
        epsilon = 1e-10  # Small constant to avoid math domain errors
        o = max(epsilon, min(1 - epsilon, o))  # Clip 'o' to avoid values too close to 0 or 1

        return -y * math.log(o) - (1 - y) * math.log(1 - o)

    def backprop(self):
        self.y = self.calcFitness()

        loss1 = self.custom_loss(self.y, self.o1)
        loss2 = self.custom_loss(self.y, self.o2)
        loss3 = self.custom_loss(self.y, self.o3)
        loss4 = self.custom_loss(self.y, self.o4)

        od1 = self.relu_deriv(self.o1)
        od2 = self.relu_deriv(self.o2)
        od3 = self.relu_deriv(self.o3)
        od4 = self.relu_deriv(self.o4)

        for w in self.brain1:
            w -= loss1 * self.o1 * od1 * self.learning_rate1

        for w in self.brain2:
            w -= loss2 * self.o2 * od2 * self.learning_rate2

        for w in self.brain3:
            w -= loss3 * self.o3 * od3 * self.learning_rate3

        for w in self.brain4:
            w -= loss4 * self.o4 * od4 * self.learning_rate4

BOARD_SIZE = app.width//25
CELL_SIZE = app.height//16
board = [[Rect(CELL_SIZE*row, CELL_SIZE*col, CELL_SIZE, CELL_SIZE, fill=None, border="black", borderWidth=0.25) for row in range(BOARD_SIZE)] for col in range(BOARD_SIZE)]
sleep(3)

for row in range(BOARD_SIZE):
    for col in range(BOARD_SIZE):
        if ran.random() < 0.2:
            board[row][col].fill = "blue"
sleep(3)

num_predators = 4
predators = [Predator(board, "red", i+1) for i in range(num_predators)]
[print(f"Pred_id:{p.id_num}") for p in predators]
app.deadPredators = []
sleep(3)

predText = Label("", 50, 0, fill="black", size=18)
alivePreds = Label(f"pred_alive:{len(predators)}",50,34,size=19,borderWidth=0.5)
deadPreds = Label(f"pred_dead:{len(app.deadPredators)}",50,56,size=18,borderWidth=0.5)
app.predGeneration = 0
predGen = Label(f"pred_Gen:{app.predGeneration}",50,76,size=18,borderWidth=0.5)
sleep(3)

num_prey = 8
preys = [Prey(board, "forestGreen", i+1) for i in range(num_prey)]
[print(f"Prey_id:{p.id_num}") for p in preys]
app.deadPreyators = []
sleep(3)

preyText = Label("", 50, 200, fill="black", size=18)
alivePreys = Label(f"prey_alive:{len(preys)}",50,254,size=19,borderWidth=0.5)
deadPreys = Label(f"prey_dead:{len(app.deadPreyators)}",50,276,size=18,borderWidth=0.5)
app.preysGeneration = 0
preysGen = Label(f"prey_Gen:{app.preysGeneration}",50,296,size=18,borderWidth=0.5)
sleep(3)

def mutatePreds():
    for row in range(len(board)):
        for col in range(len(board)):
            tile = board[row][col]
            if tile.fill == "red" or "blue":
                tile.fill = None
                tile.border = "black"

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if ran.random() < 0.2:
                board[row][col].fill = "blue"

    app.deadPredators.sort(key=lambda p: len(p.memory), reverse=True)
    for i, dp in enumerate(app.deadPredators):
        p = Predator(board, "red", (dp.id_num + 1))

        p.brain1 = dp.brain1
        p.brain2 = dp.brain2
        p.brain3 = dp.brain3
        p.brain4 = dp.brain4

        p.ran_act1 = ran.choice(p.activations)
        p.ran_act2 = ran.choice(p.activations)
        p.ran_act3 = ran.choice(p.activations)
        p.ran_act4 = ran.choice(p.activations)

        p.ran_act1_idx = p.activations.index(p.ran_act1)
        p.ran_act2_idx = p.activations.index(p.ran_act2)
        p.ran_act3_idx = p.activations.index(p.ran_act3)
        p.ran_act4_idx = p.activations.index(p.ran_act4)

        p.ran_act1_deriv = p.activations_derivs[p.ran_act1_idx]
        p.ran_act2_deriv = p.activations_derivs[p.ran_act2_idx]
        p.ran_act3_deriv = p.activations_derivs[p.ran_act3_idx]
        p.ran_act4_deriv = p.activations_derivs[p.ran_act4_idx]

        p.learning_rate1 += ran.uniform(-1, 1)
        p.learning_rate2 += ran.uniform(-1, 1)
        p.learning_rate3 += ran.uniform(-1, 1)
        p.learning_rate4 += ran.uniform(-1, 1)

        p.memory = dp.memory

        print(f"Pred_id:{p.id_num}")

        predators.append(p)
        app.predGeneration += 1
        predGen.value = f"Gen:{app.predGeneration}"
        app.deadPredators = []

def mutatePreys():
    for row in range(len(board)):
        for col in range(len(board)):
            tile = board[row][col]
            if tile.fill == "forestGreen" or "blue":
                tile.fill = None
                tile.border = "black"

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if ran.random() < 0.2:
                board[row][col].fill = "blue"

    app.deadPreyators.sort(key=lambda p: len(p.memory), reverse=True)
    for i,dp in enumerate(app.deadPreyators):
        p = Prey(board, "forestGreen", (dp.id_num+1))
        p.brain1 = dp.brain1
        p.brain2 = dp.brain2
        p.brain3 = dp.brain3
        p.brain4 = dp.brain4

        p.ran_act1 = ran.choice(p.activations)
        p.ran_act2 = ran.choice(p.activations)
        p.ran_act3 = ran.choice(p.activations)
        p.ran_act4 = ran.choice(p.activations)

        p.ran_act1_idx = p.activations.index(p.ran_act1)
        p.ran_act2_idx = p.activations.index(p.ran_act2)
        p.ran_act3_idx = p.activations.index(p.ran_act3)
        p.ran_act4_idx = p.activations.index(p.ran_act4)

        p.ran_act1_deriv = p.activations_derivs[p.ran_act1_idx]
        p.ran_act2_deriv = p.activations_derivs[p.ran_act2_idx]
        p.ran_act3_deriv = p.activations_derivs[p.ran_act3_idx]
        p.ran_act4_deriv = p.activations_derivs[p.ran_act4_idx]

        p.learning_rate1 += ran.uniform(-1, 1)
        p.learning_rate2 += ran.uniform(-1, 1)
        p.learning_rate3 += ran.uniform(-1, 1)
        p.learning_rate4 += ran.uniform(-1, 1)

        p.memory = dp.memory

        print(f"Prey_id:{p.id_num}")

        preys.append(p)
        app.preysGeneration += 1
        preysGen.value = f"Gen:{app.preysGeneration}"
        app.deadPreyators = []

app.stepsPerSecond = 1
def onStep():
    if app._can_run:
        #clean up, if needed

        for predator in predators[:]:
            predator.start_threads()

            deadPreds.value = f"pred_dead:{len(app.deadPredators)}"
            alivePreds.value = f"pred_alive:{len(predators)}"

            if predator.health <= 0:
                app._can_run = False
                print(f"Predator {predator.id_num} died and its thread will be stopped.")

        for prey in preys[:]:
            prey.start_threads()

            deadPreys.value = f"prey_dead:{len(app.deadPreyators)}"
            alivePreys.value = f"prey_alive:{len(preys)}"

            if prey.health <= 0:
                app._can_run = False
                print(f"Prey {prey.id_num} died and its thread will be stopped.")

    if not app._can_run:
        for predator in predators[:]:
            if predator.health <= 0:
                # Add the predator to deadPredators
                app.deadPredators.append(predator)
                print(f"Added Predator {predator.id_num} to deadPredators.")

                # Stop the thread for this predator
                predator.executor.shutdown()
                print(f"Stopped thread for Predator {predator.id_num}")

                # Remove the predator from the list of active predators
                predators.remove(predator)
                app.group.remove(predator.body)
                app.group.remove(predator.id)
                predator.board[predator.get_position()[0]][predator.get_position()[1]].fill = None
                print(f"Predator {predator.id_num} has been removed and will be mutated.")

        for prey in preys[:]:
            if prey.health <= 0:
                # Add the prey to deadPreys
                app.deadPreyators.append(prey)
                print(f"Added Predator {prey.id_num} to deadPreys.")

                # Stop the thread for this prey
                prey.executor.shutdown()
                print(f"Stopped thread for Prey {prey.id_num}")

                # Remove the predator from the list of active preys
                preys.remove(prey)
                app.group.remove(prey.body)
                app.group.remove(prey.id)
                prey.board[prey.get_position()[0]][prey.get_position()[1]].fill = None
                print(f"Prey {prey.id_num} has been removed and will be mutated.")

        if len(predators) == 0:
            mutatePreds()
        if len(preys) == 0:
            mutatePreys()

        app._can_run = True
        print("Resuming app")

try:
    cmu_graphics.run()
except Exception as e:
    print(f"In the event of deploying an online Integrated Development Environment (IDE) at 'https://academy.cs.cmu.edu/>TIC123'\n\nIt is imperative to omit the line of code: cmu_graphics.run().\n\nAlternatively, if the deployment is intended for a personal device,\n\nit is crucial to ensure the presence of the 'cmu_graphics' library folder,\n\nobtainable either from the official website, CMU CS Academy Desktop, or the internal package library accessible through pip.")
