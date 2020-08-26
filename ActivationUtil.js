//Written by Aa C. (ProgrammingAac@gmail.com)
class ActivationUtil{
  //leaky ReLU
  static relu(x) {
    return x < 0 ? 0.001*x : x;
  }

  //gradient for leaky ReLU
  static reluGrad(x) {
    return x <= 0 ? 0.001 : 1;
  }

  //sigmoid function
  static sigmoid(x) {
    let ex = Math.exp(x);
    return ex / (ex + 1);
  }

  //gradient for sigmoid function
  static sigmoidGrad(x) {
    let ex = Math.exp(x);
    let y = ex / (ex + 1);
    return y * (1-y);
  }

  //tanh function
  static tanh(x) {
    let ex = Math.exp(x);
    let eNegX = 1/ex;
    return (ex-eNegX) / (ex+eNegX);
  }

  //gradient for tanh function
  static tanhGrad(x) {
    let tanhX = ActivationUtil.tanh(x);
    return 1 - tanhX*tanhX;
  }

  //return the gradient function accroding to the activation function
  static getGradFunc(activation){
    switch (activation){
      case ActivationUtil.relu:
        return ActivationUtil.reluGrad;
      case ActivationUtil.sigmoid:
        return ActivationUtil.sigmoidGrad;
      case ActivationUtil.tanh:
        return ActivationUtil.tanhGrad;
    }
  }
}

if(typeof process === 'object'){
  module.exports = ActivationUtil;
}