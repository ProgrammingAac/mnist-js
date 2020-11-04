/**
 * @module ActivationUtil
 */

/**
 * @author Aa C.
 * @class Utility class that provides common neuron activation functions and their respective gradients
 */
class ActivationUtil{
  /**
   * @param {number} x The number to be passed through the ReLU function (leaky)
   * @returns {number} The result of ReLU(x)
   */
  static relu(x) {
    return x < 0 ? 0.001*x : x;
  }

  /**
   * @param {number} x The number to be passed through the ReLU (leaky) gradient function 
   * @returns {number} The gradient of the ReLU function at x
   */
  static reluGrad(x) {
    return x <= 0 ? 0.001 : 1;
  }

  /**
   * @param {number} x The number to be passed through the Sigmoid function 
   * @returns {number} The result of Sigmoid(x)
   */
  static sigmoid(x) {
    let ex = Math.exp(x);
    return ex / (ex + 1);
  }

  /**
   * @param {number} x The number to be passed through the Sigmoid gradient function 
   * @returns {number} The gradient of the Sigmoid function at x
   */
  static sigmoidGrad(x) {
    let ex = Math.exp(x);
    let y = ex / (ex + 1);
    return y * (1-y);
  }

  /**
   * @param {number} x The number to be passed through the tanh function
   * @returns {number} The result of tanh(x)
   */
  static tanh(x) {
    let ex = Math.exp(x);
    let eNegX = 1/ex;
    return (ex-eNegX) / (ex+eNegX);
  }

  /**
   * @param {number} x The number to be passed through the tanh gradient function 
   * @returns {number} The gradient of the tanh function at x
   */
  static tanhGrad(x) {
    let tanhX = ActivationUtil.tanh(x);
    return 1 - tanhX*tanhX;
  }

  /**
   * @example
   * // return ActivationUtil.reluGrad
   * ActivationUtil.getGradFunc(ActivationUtil.relu)
   * @param {function} activation Activation function provided in ActivationUtil class
   * @returns {function} The gradient of the activation function passed
   */
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

module.exports = ActivationUtil;