# Filter design by optimization
This code finds the factors of finite impulse respone (FIR) digital filters transfer functions using Traditional Particle Swarm and Tribes variation optimization algorithms. It plots the error convergency over iterations and both desired and obtained filters response over normalized angular frequency.

## Instalation

This code only uses three libraries, [matplotlib](https://matplotlib.org/3.5.1/index.html), [numpy](https://numpy.org/) and [scipy](https://docs.scipy.org/doc/scipy/index.html). Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```bash
python3 -m pip install -r requirements.txt
```

## To Do
- Create classes to goal filter generation and particle swarm;
- More filter types (butter, cheby, etc);
- Apply optimization to Infinite Impulse Response (IIR) filters;

## References
- FIR filter PSO design in MATLAB ([code](https://github.com/zypher606/ParticleSwarmOperation-FIR))
- Swarm Intelligence - James Kennedy and Russell C. Eberhart ([book](https://www.sciencedirect.com/book/9781558605954/swarm-intelligence))
- A review of Different Optimization Algorithms for a Linear phase FIR filter design problem ([article](https://ieeexplore.ieee.org/document/8378122))
- Tribe-PSO: A novel global optimization algorithm and its application in molecular docking ([article](https://www.sciencedirect.com/science/article/abs/pii/S0169743905001346))
- Multipopulation Ensemble Particle Swarm Optimizer for Engineering Design Problems ([article](https://www.hindawi.com/journals/mpe/2020/1450985/))


## Author
- Heitor Teixeira de Azambuja

## License

MIT License

Copyright (c) 2022 Heitor Teixeira de Azambuja

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.