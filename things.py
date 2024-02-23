import random as ran
import math
try:
    from time import sleep
except Exception as e:
    pass

from cmu_graphics import *
import queue
from threading import Thread, Lock

try:
    app.width, app.height = 800,800
except Exception as e:
    print(f"App.width is readonly\nApp.height is readonly\n")

app.setMaxShapeCount(10_000)
app.background = gradient("black",rgb(229,229,229))
app._can_run = True

class Predator:
    def __init__(self, board, color, id_num):
        self.board = board
        self.color = color

        available_tiles = []
        for row in range(len(self.board)):
            for col in range(len(self.board)):
                tile = self.board[row][col]
                if tile.fill == None:
                    available_tiles.append(board[row][col])

        self.body = ran.choice(available_tiles)
        self.body.fill = self.color
        self.id_num = id_num
        self.id = Label(self.id_num,self.body.centerX,self.body.centerY,size=10)
        self.health = 100

        self.row, self.col = self.get_position()

        self.num_threads = 1
        self.thread_queue = queue.LifoQueue()
        self.lock = Lock()

        self.brain_mutation_rate = ran.uniform(-1,1)
        self.learning_rate = ran.uniform(0,1)

        self.brain1 = [ran.uniform(-1, 1) * self.brain_mutation_rate for _ in range(len(self.board)*len(self.board))]
        self.brain2 = [ran.uniform(-1, 1) * self.brain_mutation_rate for _ in range(len(self.board)*len(self.board))]
        self.brain3 = [ran.uniform(-1, 1) * self.brain_mutation_rate for _ in range(len(self.board)*len(self.board))]
        self.brain4 = [ran.uniform(-1, 1) * self.brain_mutation_rate for _ in range(len(self.board)*len(self.board))]

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

        self.memory = [0]

    def softmax(self, logits):
        exp_logits = [math.exp(logit) for logit in logits]
        sum_exp_logits = sum(exp_logits)
        softmax_probs = [logit / sum_exp_logits for logit in exp_logits]
        return softmax_probs

    def get_position(self):
        return (rounded(self.body.centerX) // CELL_SIZE) % CELL_SIZE, (rounded(self.body.centerY) // CELL_SIZE) % CELL_SIZE

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
                if self.board[row][col] == self.board[self.get_position()[1]][self.get_position()[1]]:
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
        for _ in range(self.num_threads):
            thread = Thread(target=self.move, daemon=True)
            self.thread_queue.put(thread)

        while not self.thread_queue.empty():
            thread = self.thread_queue.get()
            thread.start()
            thread.join()

    def move(self):
        self.lock.acquire()

        inputs = [i + self.one_hot_encode() + [self.get_position()[0] + self.get_position()[1]] + [self.health] + self.memory for i in self.get_positions()]
        self.decision = self.get_decision(inputs)
        print(self.id_num, self.health, self.decision)

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
                if prey.body == tile:
                    prey.health = 0
                    prey.thread_queue.queue.clear()
                    deadPreyators.append(prey)
                    preys.remove(prey)
                    app.group.remove(prey.body)
                    app.group.remove(prey.id)
                    tile.fill = None
                    print(f"Prey {prey.id_num} has been eaten and removed.")
                    self.health += 100

                    tile = ran.choice(ran.choice(self.board))
                    while tile.fill != None:
                        tile = ran.choice(ran.choice(self.board))
                    else:
                        tile.fill = "forestGreen"
                        break

    def get_decision(self, inputs):
        input_vectors = []
        for i in inputs:
            for j in range(len(i)):
                input_vectors.append(i[j])

        output1 = self.ran_act1(self.relu(sum([self.brain1[i] * input_vectors[i] for i in range(len(self.brain1)) for j in range(len(inputs))])))
        output2 = self.ran_act2(self.relu(sum([self.brain2[i] * input_vectors[i] for i in range(len(self.brain2)) for j in range(len(inputs))])))
        output3 = self.ran_act3(self.relu(sum([self.brain3[i] * input_vectors[i] for i in range(len(self.brain3)) for j in range(len(inputs))])))
        output4 = self.ran_act4(self.relu(sum([self.brain4[i] * input_vectors[i] for i in range(len(self.brain4)) for j in range(len(inputs))])))

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

        output1 = math.tanh(sum([self.o1 * brain1 * mem for brain1,mem in zip(self.brain1, self.memory)]) * self.brain_mutation_rate)
        output2 = math.tanh(sum([self.o2 * brain2 * mem for brain2,mem in zip(self.brain2, self.memory)]) * self.brain_mutation_rate)
        output3 = math.tanh(sum([self.o3 * brain3 * mem for brain3,mem in zip(self.brain3, self.memory)]) * self.brain_mutation_rate)
        output4 = math.tanh(sum([self.o4 * brain4 * mem for brain4,mem in zip(self.brain4, self.memory)]) * self.brain_mutation_rate)

        #xor logic gate
        so1 = self.sigmoid(output1)
        so2 = self.sigmoid(output2)
        so3 = self.sigmoid(output3)
        so4 = self.sigmoid(output4)

        self.so1 = 0
        self.so2 = 0
        self.so3 = 0
        self.so4 = 0

        #if output is greater than 0.5 we use
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
        if outputs == [0.5]:
            self.memory.append(self.o1)
            if self.row > 0 and self.board[self.row-1][self.col].fill != "black":
                return "up"
        elif outputs == [-1]:
            self.memory.append(self.o2)
            if self.col < len(self.board[0])-1 and self.board[self.row][self.col+1].fill != "black":
                return "right"
        elif outputs == [-0.5]:
            self.memory.append(self.o3)
            if self.row < len(self.board)-1 and self.board[self.row+1][self.col].fill != "black":
                return "down"
        elif outputs == [1]:
            self.memory.append(self.o4)
            if self.col > 0 and self.board[self.row][self.col-1].fill != "black":
                return "left"
        elif outputs == [0]:
            return "None"

    def custom_loss(self, y, o):
        epsilon = 1e-10
        o = max(-1 + epsilon, min(1 - epsilon, o))

        return -y * math.log((o + 1) / 2) - (1 - y) * math.log((1 - o) / 2)

    def backprop(self):
        self.y = self.calcFitness()

        error1 = self.custom_loss(self.y, self.o1)
        error2 = self.custom_loss(self.y, self.o2)
        error3 = self.custom_loss(self.y, self.o3)
        error4 = self.custom_loss(self.y, self.o4)

        od1 = self.tanh_deriv(error1)
        od2 = self.tanh_deriv(error2)
        od3 = self.tanh_deriv(error3)
        od4 = self.tanh_deriv(error4)

        oe1 = [error1 * od1 * b1 for b1 in self.brain1]
        oe2 = [error2 * od2 * b2 for b2 in self.brain2]
        oe3 = [error3 * od3 * b3 for b3 in self.brain3]
        oe4 = [error4 * od4 * b4 for b4 in self.brain4]

        for i in range(len(self.brain1)):
            self.brain1[i] -= oe1[i] * self.learning_rate

        for i in range(len(self.brain2)):
            self.brain2[i] -= oe2[i] * self.learning_rate

        for i in range(len(self.brain3)):
            self.brain3[i] -= oe3[i] * self.learning_rate

        for i in range(len(self.brain4)):
            self.brain4[i] -= oe4[i] * self.learning_rate


class Prey:
    def __init__(self, board, color, id_num):
        self.board = board
        self.color = color

        available_tiles = []
        for row in range(len(self.board)):
            for col in range(len(self.board)):
                tile = self.board[row][col]
                if tile.fill == None:
                    available_tiles.append(self.board[row][col])

        self.body = ran.choice(available_tiles)
        self.body.fill = self.color
        self.id_num = id_num
        self.id = Label(self.id_num,self.body.centerX,self.body.centerY,size=10)
        self.health = 100

        self.row, self.col = self.get_position()

        self.num_threads = 1
        self.thread_queue = queue.LifoQueue()
        self.lock = Lock()

        self.brain_mutation_rate = ran.uniform(-1,1)
        self.learning_rate = ran.uniform(0,1)

        self.brain1 = [ran.uniform(-1, 1) * self.brain_mutation_rate for _ in range(len(self.board)*len(self.board))]
        self.brain2 = [ran.uniform(-1, 1) * self.brain_mutation_rate for _ in range(len(self.board)*len(self.board))]
        self.brain3 = [ran.uniform(-1, 1) * self.brain_mutation_rate for _ in range(len(self.board)*len(self.board))]
        self.brain4 = [ran.uniform(-1, 1) * self.brain_mutation_rate for _ in range(len(self.board)*len(self.board))]

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

        self.memory = [0]

    def softmax(self, logits):
        exp_logits = [math.exp(logit) for logit in logits]
        sum_exp_logits = sum(exp_logits)
        softmax_probs = [logit / sum_exp_logits for logit in exp_logits]
        return softmax_probs

    def get_position(self):
        return (rounded(self.body.centerX) // CELL_SIZE) % CELL_SIZE, (rounded(self.body.centerY) // CELL_SIZE) % CELL_SIZE

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
                if self.board[row][col] == self.board[self.get_position()[1]][self.get_position()[1]]:
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
        for _ in range(self.num_threads):
            thread = Thread(target=self.move, daemon=True)
            self.thread_queue.put(thread)

        while not self.thread_queue.empty():
            thread = self.thread_queue.get()
            thread.start()
            thread.join()

    def move(self):
        self.lock.acquire()

        inputs = [i + self.one_hot_encode() + [self.get_position()[0] + self.get_position()[1]] + [self.health] + self.memory for i in self.get_positions()]
        self.decision = self.get_decision(inputs)
        print(self.id_num, self.health, self.decision)

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
            self.health += 100

            tile = ran.choice(ran.choice(board))
            while tile.fill != None:
                tile = ran.choice(ran.choice(board))
            else:
                tile.fill = "blue"

    def get_decision(self, inputs):
        input_vectors = []
        for i in inputs:
            for j in range(len(i)):
                input_vectors.append(i[j])

        output1 = self.ran_act1(self.relu(sum([self.brain1[i] * input_vectors[i] for i in range(len(self.brain1)) for j in range(len(inputs))])))
        output2 = self.ran_act2(self.relu(sum([self.brain2[i] * input_vectors[i] for i in range(len(self.brain2)) for j in range(len(inputs))])))
        output3 = self.ran_act3(self.relu(sum([self.brain3[i] * input_vectors[i] for i in range(len(self.brain3)) for j in range(len(inputs))])))
        output4 = self.ran_act4(self.relu(sum([self.brain4[i] * input_vectors[i] for i in range(len(self.brain4)) for j in range(len(inputs))])))

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

        output1 = math.tanh(sum([self.o1 * brain1 * mem for brain1,mem in zip(self.brain1, self.memory)]) * self.brain_mutation_rate)
        output2 = math.tanh(sum([self.o2 * brain2 * mem for brain2,mem in zip(self.brain2, self.memory)]) * self.brain_mutation_rate)
        output3 = math.tanh(sum([self.o3 * brain3 * mem for brain3,mem in zip(self.brain3, self.memory)]) * self.brain_mutation_rate)
        output4 = math.tanh(sum([self.o4 * brain4 * mem for brain4,mem in zip(self.brain4, self.memory)]) * self.brain_mutation_rate)

        #xor logic gate
        so1 = self.sigmoid(output1)
        so2 = self.sigmoid(output2)
        so3 = self.sigmoid(output3)
        so4 = self.sigmoid(output4)

        self.so1 = 0
        self.so2 = 0
        self.so3 = 0
        self.so4 = 0

        #if output is greater than 0.5 we use
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
        if outputs == [0.5]:
            self.memory.append(self.o1)
            if self.row > 0 and self.board[self.row-1][self.col].fill != "black":
                return "up"
        elif outputs == [-1]:
            self.memory.append(self.o2)
            if self.col < len(self.board[0])-1 and self.board[self.row][self.col+1].fill != "black":
                return "right"
        elif outputs == [-0.5]:
            self.memory.append(self.o3)
            if self.row < len(self.board)-1 and self.board[self.row+1][self.col].fill != "black":
                return "down"
        elif outputs == [1]:
            self.memory.append(self.o4)
            if self.col > 0 and self.board[self.row][self.col-1].fill != "black":
                return "left"
        elif outputs == [0]:
            return "None"

    def custom_loss(self, y, o):
        epsilon = 1e-10
        o = max(-1 + epsilon, min(1 - epsilon, o))

        return -y * math.log((o + 1) / 2) - (1 - y) * math.log((1 - o) / 2)

    def backprop(self):
        self.y = self.calcFitness()

        error1 = self.custom_loss(self.y, self.o1)
        error2 = self.custom_loss(self.y, self.o2)
        error3 = self.custom_loss(self.y, self.o3)
        error4 = self.custom_loss(self.y, self.o4)

        od1 = self.tanh_deriv(error1)
        od2 = self.tanh_deriv(error2)
        od3 = self.tanh_deriv(error3)
        od4 = self.tanh_deriv(error4)

        oe1 = [error1 * od1 * b1 for b1 in self.brain1]
        oe2 = [error2 * od2 * b2 for b2 in self.brain2]
        oe3 = [error3 * od3 * b3 for b3 in self.brain3]
        oe4 = [error4 * od4 * b4 for b4 in self.brain4]

        for i in range(len(self.brain1)):
            self.brain1[i] -= oe1[i] * self.learning_rate

        for i in range(len(self.brain2)):
            self.brain2[i] -= oe2[i] * self.learning_rate

        for i in range(len(self.brain3)):
            self.brain3[i] -= oe3[i] * self.learning_rate

        for i in range(len(self.brain4)):
            self.brain4[i] -= oe4[i] * self.learning_rate

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
[print(f"Pred_id:{p.id_num},lr:{p.learning_rate},bmr:{p.brain_mutation_rate}\n") for p in predators]
deadPredators = []
sleep(3)

predText = Label("", 50, 0, fill="black", size=18)
alivePreds = Label(f"pred_alive:{len(predators)}",50,34,size=19,borderWidth=0.5)
deadPreds = Label(f"pred_dead:{len(deadPredators)}",50,56,size=18,borderWidth=0.5)
predGeneration = 0
predGen = Label(f"pred_Gen:{predGeneration}",50,76,size=18,borderWidth=0.5)
sleep(3)

num_prey = 8
preys = [Prey(board, "forestGreen", i+1) for i in range(num_prey)]
[print(f"Prey_id:{p.id_num},lr:{p.learning_rate},bmr:{p.brain_mutation_rate}\n") for p in preys]
deadPreyators = []
sleep(3)

preyText = Label("", 50, 200, fill="black", size=18)
alivePreys = Label(f"prey_alive:{len(preys)}",50,254,size=19,borderWidth=0.5)
deadPreys = Label(f"prey_dead:{len(deadPreyators)}",50,276,size=18,borderWidth=0.5)
preysGeneration = 0
preysGen = Label(f"prey_Gen:{preysGeneration}",50,296,size=18,borderWidth=0.5)
sleep(3)

def mutatePreds():
    for row in range(len(board)):
        for col in range(len(board)):
            board[row][col].fill = None

    deadPredators.sort(key=lambda p: len(p.memory), reverse=True)
    for i,dp in enumerate(deadPredators):
        p = Predator(board, "red", (i+1))
        p.brain1 = dp.brain1
        p.brain2 = dp.brain2
        p.brain3 = dp.brain3
        p.brain4 = dp.brain4

        p.brain_mutation_rate += ran.uniform(-1,1)
        p.learning_rate += ran.uniform(-1,1)
        p.memory = dp.memory

        f"id:{p.id_num},lr:{p.learning_rate},bmr:{p.brain_mutation_rate}"

        predators.append(p)
        predGen.value = f"Gen:{predGeneration + 1}"

def mutatePreys():
    for row in range(len(board)):
        for col in range(len(board)):
            board[row][col].fill = None

    deadPreyators.sort(key=lambda p: len(p.memory), reverse=True)
    for i,dp in enumerate(deadPreyators):
        p = Prey(board, "forestGreen", (i+1))
        p.brain1 = dp.brain1
        p.brain2 = dp.brain2
        p.brain3 = dp.brain3
        p.brain4 = dp.brain4

        p.brain_mutation_rate += ran.uniform(-1,1)
        p.learning_rate += ran.uniform(-1,1)
        p.memory = dp.memory

        f"id:{p.id_num},lr:{p.learning_rate},bmr:{p.brain_mutation_rate}"

        preys.append(p)
        preysGen.value = f"Gen:{preysGeneration + 1}"

app.stepsPerSecond = 1
def onStep():
    if app._can_run:
        #clean up, if needed

        for predator in predators[:]:
            predator.start_threads()

            deadPreds.value = f"pred_dead:{len(deadPredators)}"
            alivePreds.value = f"pred_alive:{len(predators)}"

            if predator.health <= 0:
                app._can_run = False
                print(f"Predator {predator.id_num} died and its thread will be stopped.")

        for prey in preys[:]:
            prey.start_threads()

            deadPreys.value = f"prey_dead:{len(deadPreyators)}"
            alivePreys.value = f"prey_alive:{len(preys)}"

            if prey.health <= 0:
                app._can_run = False
                print(f"Prey {prey.id_num} died and its thread will be stopped.")

    if not app._can_run:
        for predator in predators[:]:
            if predator.health <= 0:
                # Add the predator to deadPredators
                deadPredators.append(predator)
                print(f"Added Predator {predator.id_num} to deadPredators.")

                # Stop the thread for this predator
                predator.thread_queue.queue.clear()  # Clear the queue to stop the thread
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
                deadPreyators.append(prey)
                print(f"Added Predator {prey.id_num} to deadPreys.")

                # Stop the thread for this prey
                prey.thread_queue.queue.clear()  # Clear the queue to stop the thread
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
