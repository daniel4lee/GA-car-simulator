# GA-car-simulator

A car control simulator based on Genetic Algorithm

![preview](https://i.imgur.com/QUuc1PC.gif)

## Goal & Objective

Use genetic algorithm(GA) to train the radial base function network(RBFN).

The goal of the practice is to use the RBFN network which adjusted by GA to reach the end area without encountering any wall. Whether the car arrives safely or not, the program will display the result.

## Genetic Algorithm design detail

The inputs of GA are the 3 parameters(w, m, σ) of RBFN, and the RBFN structure diagram is as below.

![structure](https://i.imgur.com/ljrATMV.jpg)


![function](https://i.imgur.com/ctaMMwj.png)

I set the fitness function as the mean variance between the expected output of the input and the RBFN output with the input. 
``` python
fitness function: (1 / total number of training data) * sum(abs(d.expected_output - RBFN(d.input)) for d in training_dataset)
```
Hence, the RBFN parameters causing the lowest fitness value is the optimal parameters.

See [here](https://docs.google.com/document/d/1s8VtIBMynZoHHiBqRs_G6dmlKGg-_phJg71lLPQXR6M/edit?usp=sharing) for more details about experiments and analysis.
## Installation

1. Clone the project

```git bash
    git clone https://github.com/daniel4lee/fuzzy-system.git
```

2. Change directory to the root of the project and run with Python interpreter

``` bash
    python main.py
```

## Test on Customized Map

### Map File location and format

The map should be `*.txt` format and put in `/map_data` location.

### Example Format

![example map](https://i.imgur.com/oHiqTMr.jpg)

``` python
0,0,90  # x, y, degree(the initial position coordinate and direction angle of the car
18,40   # x, y (the top-left coordinate of the ending area)
30,37   # x, y (the bottom-right coordinate of the ending area)
-6,0   # the first point of the wall in map
-6,22
18,22
18,50
30,50
30,10
6,10
6,0
-6,0   # the last point of the wall in map
```

The coordinates after the third line are the corner points of the walls in the map.


## Training Data

### `train4D.txt` Format

|        Input (Distances)       |Output (Wheel Angle)|
|:------------------------------:|:------------------:|
|`21.1292288 9.3920089 7.7989045`|    `-14.7971418`   |

``` python
# Front_Distance Right_Distance Left_Distance Wheel_Angle

22.0000000 8.4852814 8.4852814 -16.0709664
21.1292288 9.3920089 7.7989045 -14.7971418
20.3973643 24.4555821 7.2000902 16.2304876
19.1995799 25.0357595 7.5129743 16.0825385
18.1744869 42.5622911 8.0705896 15.5075777
```

### `train6D.txt` Format

|        Input (Distances)                           |Output (Wheel Angle)|
|:--------------------------------------------------:|:------------------:|
|`0.0000000 0.0000000 22.0000000 8.4852814 8.4852814`|    `-16.0709664`   |

``` python
# X Y Front_Distance Right_Distance Left_Distance Wheel_Angle

0.0000000 0.0000000 22.0000000 8.4852814 8.4852814 -16.0709664
0.0000000 0.9609196 21.1292288 9.3920089 7.7989045 -14.7971418
-0.0892157 1.9236307 20.3973643 24.4555821 7.2000902 16.2304876
-0.2588831 2.8686659 19.1995799 25.0357595 7.5129743 16.0825385
-0.3398267 3.8261141 18.1744869 42.5622911 8.0705896 15.5075777
-0.3319909 4.7896773 17.2922349 8.1967401 8.9258102 -14.6592172
```


## Dependencies

- [numpy](http://www.numpy.org/)

- [PyQt5](https://pypi.org/project/PyQt5/)

- [Shapely](https://pypi.org/project/Shapely/)

- [descartes](https://pypi.org/project/descartes/)

- [matplotlib](https://matplotlib.org/)