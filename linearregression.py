def errorSquare(dataX, dataY, b, m):
  error = 0
  for i in range(0, len(dataX)):
    error += error(dataY[i], b + m * dataX[i]) ** 2

  return error/float(len(dataX))

def step_descend(dataX, dataY, b_current, m_current, rate):
  N = float(len(dataX))
  b_gradient = 0
  m_gradient = 0
  for i in range(len(dataX)):
    b_gradient += -(2/N) * (dataY[i] - ((m_current * dataX[i]) + b_current))
    m_gradient += -(2/N) * dataX[i] * (dataY[i] - ((m_current * dataX[i]) + b_current))

  newB = b_current - (b_gradient * rate)
  newM = m_current - (m_gradient * rate)
  return newB, newM

dataX = [-5,-4,-3,-2,-1,0,1,2,3,4,5];
dataY = [-4,-3,-1,-1, 0,1,0,2,2,5,4];

rate = 0.001
b, m = 0, 0
for i in range(0, 2000):
  b, m = step_descend(dataX, dataY, b, m, rate)

print(str(b)+" + "+str(m)+"x")
