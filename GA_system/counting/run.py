import math
import random
from GA_system.counting.chromosome import Chromosome
import numpy as np
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, pyqtSlot
import shapely.geometry as sp
import descartes
from copy import deepcopy


class RunSignals(QObject):
    result = pyqtSignal(object)


class CarRunning(QRunnable):
    """ work thread, will execute "run" first """

    def __init__(self, data, filename, traindata, traindataname, list_l):
        super(CarRunning, self).__init__()
        # for return result
        self.signals = RunSignals()
        # read data file
        self.data = data[filename]
        self.train_data = traindata[traindataname]
        # adjust parameters
        self.iteration_times = list_l[0]
        self.population_num = list_l[1]
        self.reproduction_choose = list_l[2]
        self.crossover_prob = list_l[3]
        self.mutation_prob = list_l[4]
        self.neurl_j = list_l[5]
        if traindataname.find('4') != -1:
            self.dim_i = 3
        elif traindataname.find('6') != -1:
            self.dim_i = 5
        self.upbound_of_SD = list_l[7]
        self.tournament_num = list_l[6]
        # !!! could be set as variable
        self.mutation_tiny_adjust_var = 1
        # caculation of upbound in map
        max_px = max_py = -math.inf
        min_px = min_py = math.inf
        for i, j in zip(self.data.x[2:], self.data.y[2:]):
            max_px = max(i, max_px)
            min_px = min(i, min_px)
            max_py = max(j, max_py)
            min_py = min(j, min_py)

        self.upbound_of_map = ((max_px - min_px)**2 +
                               (max_py - min_py)**2)**(0.5)
        maxp = -math.inf
        minp = math.inf
        for i in self.train_data.v_x:
            for j in range(self.dim_i):
                maxp = max(maxp, i[j])
                minp = min(minp, i[j])
        self.range = (maxp, minp)
        # according pocket rule, saving the best
        self.best_individual = None
        # initialization the population
        self.populations = []
        for _ in range(self.population_num):
            self.populations.append(Chromosome(
                self.neurl_j, self.dim_i, self.range, self.upbound_of_SD))
        self.reproduction_pool = []

    @pyqtSlot()
    def run(self):
        """
        [the main function for caculation the GA]
        """
        # loop until reaching iteration times
        for iteration_time in range(self.iteration_times):
            # initialization the averge 0f adapt funct value and best individual
            avg_e = 0
            # loop for evaluate adapt funct.
            for k in range(self.population_num):
                self.populations[k].adapt_value = self.adaptation_funct(k)
                avg_e += self.populations[k].adapt_value
                if k == 0 and iteration_time == 0:
                    self.best_individual = deepcopy(self.populations[k])
                elif self.populations[k].adapt_value > self.best_individual.adapt_value:
                    self.best_individual = deepcopy(self.populations[k])
            avg_e = avg_e / self.population_num
            print(iteration_time, 1/self.best_individual.adapt_value, 1/avg_e)
            # use the evaluate value to counting the individual in reproduction pool
            if self.reproduction_choose == 'w':
                self.wheel_choose()
            elif self.reproduction_choose == 't':
                self.tournament_choose()

            # doing reproduction in reproduction pool

            self.reproduction_execution()
            # mutation
            self.mutation_execution()

            self.populations = deepcopy(self.reproduction_pool)

            self.reproduction_pool = []
            # check if out of bound

        self.run_trained_rbfn(self.dim_i, self.best_individual)

    def adaptation_funct(self, index):
        e_n = 0
        # [y_n - F(x_n)]^2
        for idx, expected_output in enumerate(self.train_data.wheel_angle):
            rbfn_value = self.rbfn_funct(self.train_data.v_x[idx], index)
            rbfn_value = max(-40, min(rbfn_value*40, 40))
            e_n += abs(expected_output - rbfn_value)
        # 1/N*(E_n)
        e_n = e_n/len(self.train_data.wheel_angle)
        # since the better parameters producing less e(n), we suppose let it
        # become bigger for reproduction.
        return 1 / e_n

    def rbfn_funct(self, input_vector, idx):
        f_x = self.populations[idx].theta[0]  # theta
        for j in range(self.neurl_j):
            gaussian = self.gaussian_funct(
                j, input_vector, self.populations[idx].means, self.populations[idx].sd[j])
            f_x = f_x + self.populations[idx].weight[j] * gaussian
        return f_x

    def gaussian_funct(self, jth_neurl, v_x, v_m, o):
        temp = 0
        means = np.array(
            v_m[len(v_x) * jth_neurl:len(v_x) * jth_neurl + self.dim_i])
        temp = (v_x - means).dot(v_x - means)
        return (math.exp(-temp / (2 * o ** 2)))

    def wheel_choose(self):
        """[This function is used for execute the selection in the way of wheel.]        
        Arguments:
            populations {[[], []]} -- [the population wait for put in reproduction pool,
            and [0] is individual's parameters, [1] corresponding evaluation value]
        EX :
            wheel_split_list is showing as below, and we randomly produced a
            value between it to determined which of it to put into reproduction
            pool.
            0  x0/sum    (x0+x1)/sum    1
            |___|__________|____________|
        """
        # making the wheel split list
        denominator = 0
        numerator = 0
        for i in self.populations:
            denominator += i.adapt_value
        wheel_split_list = []
        for i in self.populations:
            # first element is 0
            wheel_split_list.append(numerator/denominator)
            numerator += i.adapt_value
        # add the final split point
        wheel_split_list.append(numerator/denominator)

        # randomly produced a value between 0~1
        # and determined whether put it into reproduction pool
        choosen = False
        for i in range(self.population_num):
            ptr = random.uniform(0, 1)
            for p in range(1, len(wheel_split_list)):
                choosen = False
                if ptr < wheel_split_list[p]:
                    self.reproduction_pool.append(
                        deepcopy(self.populations[p-1]))
                    choosen = True
                    break
            if choosen is False:
                print('should never happen')
                self.reproduction_pool.append(deepcopy(self.populations[-1]))

    def tournament_choose(self):
        """[This function randomly choose individual in tournament_num, and 
        let the higest evaluation into reproduction pool]

        Arguments:
            populations {[[], []]} -- [the population wait for put in reproduction 
            pool, and [0] is individual's parameters, [1] corresponding evaluation 
            value]
        """
        def random_produce_nums(low, up, num):
            """[return amount of num random value and not same]

            Arguments:
                low {[int]} -- [lower bound, containing]
                up {[int]} -- [upper bound, not contain]
                num {[int]} -- [number amount]
            """
            result = []
            result.append(random.randrange(low, up))
            while(len(result) < num):
                temp = random.randrange(low, up)
                if temp not in result:
                    result.append(temp)
            return result
        for _ in range(self.population_num):
            competitors = random_produce_nums(
                0, self.population_num, self.tournament_num)
            best = competitors[0]
            for idx in competitors:
                if self.populations[idx].adapt_value > self.populations[best].adapt_value:
                    best = idx
            self.reproduction_pool.append(deepcopy(self.populations[best]))

    def reproduction_execution(self):
        """[As title, execute reproduction]

        Arguments:
            pool {individual, individual, ...} -- [the list of individual]
        """
        def random_crossover(a, b):
            adjust = random.uniform(-1, 1)
            temp_theta = adjust * \
                (self.reproduction_pool[a].theta -
                 self.reproduction_pool[b].theta)
            temp_weight = adjust * \
                (self.reproduction_pool[a].weight -
                 self.reproduction_pool[b].weight)
            temp_means = adjust * \
                (self.reproduction_pool[a].means -
                 self.reproduction_pool[b].means)
            temp_sd = adjust * \
                (self.reproduction_pool[a].sd-self.reproduction_pool[b].sd)
            self.reproduction_pool[a].theta += temp_theta
            self.reproduction_pool[b].theta -= temp_theta
            self.reproduction_pool[a].weight += temp_weight
            self.reproduction_pool[b].weight -= temp_weight
            self.reproduction_pool[a].means += temp_means
            self.reproduction_pool[b].means -= temp_means
            self.reproduction_pool[a].sd += temp_sd
            self.reproduction_pool[b].sd -= temp_sd

        if self.population_num % 2 == 1:
            for i in range(0, self.population_num-1, 2):
                if random.uniform(0, 1) < self.crossover_prob:
                    random_crossover(i, i+1)
        else:
            for i in range(0, self.population_num, 2):
                if random.uniform(0, 1) < self.crossover_prob:
                    random_crossover(i, i+1)
        for i in range(self.population_num):
            self.reproduction_pool[i].theta = np.clip(
                self.reproduction_pool[i].theta, -1, 1)
            self.reproduction_pool[i].weight = np.clip(
                self.reproduction_pool[i].weight, -1, 1)
            self.reproduction_pool[i].means = np.clip(
                self.reproduction_pool[i].means, self.range[0], self.range[1])
            self.reproduction_pool[i].sd = np.clip(
                self.reproduction_pool[i].sd, 0.001, math.inf)
            self.reproduction_pool[i].adapt_value = None

    def mutation_execution(self):
        for i in range(self.population_num):
            if random.uniform(0, 1) < self.mutation_prob:
                mutation_factor = Chromosome(
                    self.neurl_j, self.dim_i, self.range, self.upbound_of_SD)
                if random.uniform(0, 1) < 0.5:
                    self.reproduction_pool[i].theta += self.mutation_tiny_adjust_var * \
                        mutation_factor.theta
                    self.reproduction_pool[i].weight += self.mutation_tiny_adjust_var * \
                        mutation_factor.weight
                    self.reproduction_pool[i].means += self.mutation_tiny_adjust_var * \
                        mutation_factor.means
                    self.reproduction_pool[i].sd += self.mutation_tiny_adjust_var * \
                        mutation_factor.sd
                else:
                    self.reproduction_pool[i].theta -= self.mutation_tiny_adjust_var * \
                        mutation_factor.theta
                    self.reproduction_pool[i].weight -= self.mutation_tiny_adjust_var * \
                        mutation_factor.weight
                    self.reproduction_pool[i].means -= self.mutation_tiny_adjust_var * \
                        mutation_factor.means
                    self.reproduction_pool[i].sd -= self.mutation_tiny_adjust_var * \
                        mutation_factor.sd
        self.reproduction_pool[i].theta = np.clip(
            self.reproduction_pool[i].theta, -1, 1)
        self.reproduction_pool[i].weight = np.clip(
            self.reproduction_pool[i].weight, -1, 1)
        self.reproduction_pool[i].means = np.clip(
            self.reproduction_pool[i].means, self.range[0], self.range[1])
        self.reproduction_pool[i].sd = np.clip(
            self.reproduction_pool[i].sd, 0.001, None)

    def run_trained_rbfn(self, dim, best_parameters):
        def distance(points, car_loc):
            if isinstance(points, sp.MultiPoint):
                min_dis = ((points[0].x - car_loc[0])**2 +
                           (points[0].y - car_loc[1])**2)**(1/2)
                min_point = (points[0].x, points[0].y)
                for i in range(1, len(points)):
                    temp = ((points[i].x - car_loc[0])**2 +
                            (points[i].y - car_loc[1])**2)**(1/2)
                    if(temp < min_dis):
                        min_dis = temp
                        min_point = (points[i].x, points[i].y)
                l = [min_dis, min_point]
                return l
            elif isinstance(points, sp.Point):
                l = []
                l.append(
                    ((points.x - car_loc[0])**2 + (points.y - car_loc[1])**2)**(1/2))
                min_point = (points.x, points.y)
                l.append(min_point)
                return l

        def rbfn_funct(input_vector, parameters):
            f_x = parameters.theta[0]  # theta
            for j in range(self.neurl_j):
                gaussian = gaussian_funct(
                    j, input_vector, parameters.means, parameters.sd[j])
                f_x = f_x + parameters.weight[j] * gaussian
            return f_x

        def gaussian_funct(jth_neurl, v_x, v_m, o):
            temp = 0
            means = np.array(
                v_m[len(v_x) * jth_neurl:len(v_x) * jth_neurl + self.dim_i])
            temp = (v_x - means).dot(v_x - means)
            return (math.exp(-temp / (2 * o ** 2)))
        def se_rbfn_funct(input_vector, parameters):
            f_x = parameters.theta[0]  # theta
            for j in range(6):
                gaussian = gaussian_funct(
                    j, input_vector, parameters.means, parameters.sd[j])
                f_x = f_x + parameters.weight[j] * gaussian
            return f_x

        def se_gaussian_funct(jth_neurl, v_x, v_m, o):
            temp = 0
            means = np.array(
                v_m[len(v_x) * jth_neurl:len(v_x) * jth_neurl + 3])
            temp = (v_x - means).dot(v_x - means)
            return (math.exp(-temp / (2 * o ** 2)))
        """
        Run this function 
        """
        # trace data [0] = car center x, [1] = car center y, [2] = direction length,
        # [3] = right length, [4] = left length, [5] = thita, [6] = direct point on map line
        # [7] = right point on map line, [8] left point on map line, [9] the angle between dir car and horizontal
        trace_10d = []
        for i in range(10):
            trace_10d.append([])
        # creat end area by shapely
        end_area = []
        end_area.append((self.data.x[0], self.data.y[0]))
        end_area.append((self.data.x[1], self.data.y[0]))
        end_area.append((self.data.x[1], self.data.y[1]))
        end_area.append((self.data.x[0], self.data.y[1]))
        end_area = sp.Polygon(end_area)

        # creat map line by shapely
        map_line = []
        for i in range(2, len(self.data.x)):
            map_line.append([self.data.x[i], self.data.y[i]])
        map_line = sp.LineString(map_line)

        car_center = (self.data.start[0], self.data.start[1])
        car = sp.Point(*car_center).buffer(3)
        # main loop for computing through fuzzy architecture
        while(not car.intersection(map_line)):

            # let data list[1] be 1 longer as signal here !!
            if(end_area.contains(sp.Point(car_center))):
                trace_10d[1].append(0)
                break

            # creat car circle polygon by shapely and initial it, r, x, y
            if (len(trace_10d[0]) == 0):
                # count the distance
                r = self.upbound_of_map
                # initial x y fai
                x = self.data.start[0]
                y = self.data.start[1]
                fai = self.data.start[2]
                output = 0
            else:
                """update new point for computing """
                car_center = (car_center[0] + math.cos(math.radians(fai + output)) + math.sin(math.radians(fai))*math.sin(math.radians(output)),
                              car_center[1] + math.sin(math.radians(fai + output)) - math.sin(math.radians(output))*math.cos(math.radians(fai)))
                car = sp.Point(*car_center).buffer(3)
                x = car_center[0]
                y = car_center[1]
                fai = fai - \
                    math.degrees(math.asin(2*math.sin(math.radians(output))/6))
            ##
            trace_10d[0].append(x)
            trace_10d[1].append(y)
            trace_10d[9].append(fai)
            # dir, l, r line for counting intersection

            dir_line = [
                [x, y], [x + r * math.cos(math.radians(fai)), y + r * math.sin(math.radians(fai))]]
            l_line = [[x, y], [
                x + r * math.cos(math.radians(fai + 45)), y + r * math.sin(math.radians(fai + 45))]]
            r_line = [[x, y], [
                x + r * math.cos(math.radians(fai - 45)), y + r * math.sin(math.radians(fai - 45))]]

            # First, computing the dir, l, and r distance between car and wall
            temp = sp.LineString(dir_line).intersection(map_line)
            temp = distance(temp, car_center)
            dir_dist = temp[0]
            trace_10d[6].append(temp[1])
            temp = sp.LineString(r_line).intersection(map_line)
            temp = distance(temp, car_center)
            r_dist = temp[0]
            trace_10d[7].append(temp[1])
            temp = sp.LineString(l_line).intersection(map_line)
            temp = distance(temp, car_center)
            l_dist = temp[0]
            trace_10d[8].append(temp[1])

            ### record distace set in trace6d ###
            trace_10d[2].append(dir_dist)
            trace_10d[3].append(r_dist)
            trace_10d[4].append(l_dist)
            list4d = np.array([dir_dist, r_dist, l_dist])
            list6d = np.array([x, y, dir_dist, r_dist, l_dist])
            if(dim == 3):
                    output = rbfn_funct(np.array(list4d), best_parameters)
            if(dim == 5):
                output = rbfn_funct(np.array(list6d), best_parameters)
            output = max(-40, min(output * 40, 40))
            ### record wheel angle in trace6d ###
            trace_10d[5].append(output)
        self.signals.result.emit([trace_10d, best_parameters])
