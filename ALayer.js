/**
 * @module ALayer
 */

const Matrix = require('./Matrix.js');

/**
 * @author Aa C.
 * @class The representation of an average pooling layer in a convolutional neural network (CNN)
 */
class ALayer{
  
  /**
   * @constructs 
   * @param {number} kernelSize The length(or width) of the kernel 
   */
  constructor(kernelSize){
    this.kernelSize = kernelSize;
  }

  /**
   * Store the layer's information in a formatted string and returns that string
   * 
   * @returns {string} String containing the layer's information
   */
  serialize(){
    let ser = "";
    ser += "<" + this.constructor.name + ">";
    ser += "<" + this.kernelSize + ">";

    return ser;
  }

  /**
   * Extract information from the serialized string returned by serialize() function 
   * and return those information as an object literal
   * 
   * @param {string} ser String returned by serialize() function
   * @returns {object} object literal containing the layer's information
   */
  static getLayerInfo(ser){
    let infoRegex = /^<([A-Za-z]+)><([0-9]+)>/;
    let info = infoRegex.exec(ser);
    let layerType = info[1];
    let kernelSize = JSON.parse(info[2]);

    return {
      layerType: layerType,
      kernelSize: kernelSize
    };
  }

  /**
   * Create an instance of ALayer using the information in a serialized string (returned by serialize() function)
   * 
   * @param {string} ser String returned by serialize() function
   * @returns {object} an instance of ALayer
   */
  static deserialize(ser){
    let layerInfo = ALayer.getLayerInfo(ser);
    let kernelSize = layerInfo.kernelSize;

    let aLayer = new ALayer(kernelSize);
    return aLayer;
  }

  /**
   * @param {string} ser String returned by serialize() function
   * @returns {string} a brief description of the layer
   */
  static getLayerDescription(ser){
    let layerInfo = ALayer.getLayerInfo(ser);
    let layerType = layerInfo.layerType;
    let kernelSize = layerInfo.kernelSize;

    let description = "";
    description += layerType;
    description += " / kernelSize: " + kernelSize;

    return description;
  }

  /**
   * Called when creating a neural network instance, in order to establish the sequence of different layers in a model.
   * 
   * @param {object} prevLayer The layer to be placed before this layer
   * @param {object} nextLayer The layer to be placed after this layer
   */
  link(prevLayer, nextLayer){
    this.prevLayer = prevLayer;
    this.nextLayer = nextLayer;

    let stride = this.kernelSize;

    this.height = Math.ceil(prevLayer.height / stride);
    this.width = Math.ceil(prevLayer.width / stride);

    if (prevLayer.height % stride > 0){
      this.yPadding = stride - prevLayer.height%stride;
    } else this.yPadding = 0;

    if (prevLayer.width % stride > 0){
      this.xPadding = stride - prevLayer.width%stride;
    } else this.xPadding = 0;

    this.numMaps = prevLayer.numMaps;
  }

  /**
   * Performing the forward propagation.
   * Using the previous layer's output as the input.
   * The output is saved in instance's field (this.O)
   */
  forProp(){
    let I = this.prevLayer.O;
    this.I = I;
    let O = new Array(this.numMaps);
    for (let i = 0; i < this.numMaps; i++){
      let inputMap = I[i];
      O[i] = this.avgPool(inputMap);
    }
    
    this.O = O;
  }

  /**
   * Pad a matrix of values with zeros so as to output a Matrix with suitable dimensions to be used as input for this layer
   * 
   * @example
   * // inputMap = 3x3 Matrix; Pooling kernel = 2x2 --> return a 4x4 Matrix with padded zeros
   * 
   * @param {Matrix} inputMap Input map
   * @returns {Matrix} Input map matrix with padded zeros
   */
  padMap(inputMap){
    let paddedHeight = inputMap.row + this.yPadding;
    let paddedWidth = inputMap.col + this.xPadding;

    let paddedMap = new Matrix(paddedWidth, paddedHeight);

    for (let r = 0; r < this.yPadding; r++){
      for (let c = 0; c < paddedMap.col; c++){
        paddedMap[c][r] = 0;
      }
    }

    for (let r = this.yPadding; r < paddedMap.row; r++){
      for (let c = 0; c < this.xPadding; c++){
        paddedMap[c][r] = 0;
      }
      for (let c = this.xPadding; c < paddedMap.col; c++){
        paddedMap[c][r] = inputMap[c - this.xPadding][r - this.yPadding];
      }
    }

    return paddedMap;
  }

  /**
   * Perform average pooling on a value map. Pad the value map before the pooling if necessary
   * 
   * @param {Matrix} inputMap The value map to perform average pooling on
   * @returns {Matrix} The result pooled map as an Matrix instance
   */
  avgPool(inputMap){
    let paddedMap;
    let stride = this.kernelSize;

    if (this.xPadding > 0 || this.yPadding > 0){
      paddedMap = this.padMap(inputMap);
    } else paddedMap = inputMap;

    let result = new Matrix(this.width, this.height);

    //EQUATION (6)
    for (let c = 0; c < result.col; c++){
      for (let r = 0; r < result.row; r++){
        let sum = 0;
        for (let i = c*stride; i < (c+1)*stride; i++){
          for (let j = r*stride; j < (r+1)*stride; j++){
            sum += paddedMap[i][j];
          }
        }
        result[c][r] = sum / (this.kernelSize*this.kernelSize);
      }
    }

    return result;
  }
  
  /**
   * Remove some side columns and rows to reduce the dimensions
   * of a value map to be the same as previous layer's output's
   * 
   * @param {Matrix} outputMap The value map to be depadded
   * @returns {Matrix} The depadded value map
   */
  depadMap(outputMap){
    let depaddedWidth = outputMap.col - this.xPadding;
    let depaddedHeight = outputMap.row - this.yPadding;

    let result = new Matrix(depaddedWidth, depaddedHeight);
    for (let c = 0; c < result.col; c++){
      for (let r = 0; r < result.row; r++){
        result[c][r] = outputMap[c+this.xPadding][r+this.yPadding];
      }
    }

    return result;
  }

  /**
   * Averaging and spreading each gradient value 
   * in the argument map to an area of (stride * stride).  
   * Return the result of the above operation.
   * 
   * 
   * @param {Matrix} gradMap Matrix containing gradent values
   * @returns {Matrix} Matrix containing the spreaded and averaged gradient values
   */
  avgPoolGrad(gradMap){
    let stride = this.kernelSize;

    let paddedHeight = gradMap.row * stride;
    let paddedWidth = gradMap.col * stride;

    let result = new Matrix(paddedWidth, paddedHeight);

    //EQUATION (7)
    for (let c = 0; c < result.col; c++){
      for (let r = 0; r < result.row; r++){
        result[c][r] = gradMap[Math.floor(c/stride)][Math.floor(r/stride)];
        result[c][r] /= stride * stride;
      }
    }

    return this.depadMap(result);
  }

  /**
   * Backward propagation.  
   * Pass the gradient information from next layer to the previous layer
   */
  backProp(){
    let dEdO = this.nextLayer.dEdI;

    let dEdI = new Array(this.numMaps);
    for (let i = 0; i < this.numMaps; i++){
      dEdI[i] = this.avgPoolGrad(dEdO[i]);
    }
      
    this.dEdI = dEdI;
    return null;
  }

  /**
   * Place holder. ALayer does not have any trainable parameters
   */
  addDP(){
    return;
  }

  /**
   * Place holder. ALayer does not have any trainable parameters
   */
  updateParameters(){
    return;
  }
}

module.exports = ALayer;