/**
 * @module CLayer
 */

const ActivationUtil = require('./ActivationUtil.js');
const Matrix = require('./Matrix.js');

/**
 * @author Aa C.
 * @class The representation of a convolution layer in a convolutional neural network (CNN)
 */
class CLayer{

  /**
   * @constructs
   * @param {number} kernelSize The length(or width) of the kernel
   * @param {number} numMaps The number of value maps produced in forward propagation
   */
  constructor(kernelSize, numMaps){
    this.kernelSize = kernelSize;
    this.numMaps = numMaps;
  }

  /**
   * Store the layer's information in a formatted string, and return that string
   * 
   * @returns {string} String containing the layer's information
   */
  serialize(){
    let ser = "";
    ser += "<" + this.constructor.name + ">";
    ser += "<" + this.kernelSize + ">";
    ser += "<" + this.numMaps + ">";
    ser += JSON.stringify(this.kernels);

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
    let infoRegex = /^<([A-Za-z]+)><([0-9]+)><([0-9]+)>(.+)/;
    let info = infoRegex.exec(ser);
    let layerType = info[1];
    let kernelSize = JSON.parse(info[2]);
    let numMaps = JSON.parse(info[3]);
    let kernels = Matrix.from2DArray(JSON.parse(info[4]));
    for (let c = 0; c < kernels.length; c++){
      for (let r = 0; r < kernels[c].length; r++){
        kernels[c][r] = Matrix.from2DArray(kernels[c][r]);
      }
    }

    return {
      layerType: layerType,
      kernelSize: kernelSize,
      numMaps: numMaps,
      kernels: kernels
    };
  }

  /**
   * Create an instance of CLayer using the information in a serialized string (returned by serialize() function)
   * 
   * @param {string} ser String returned by serialize() function
   * @returns {object} an instance of CLayer
   */
  static deserialize(ser){
    let layerInfo = CLayer.getLayerInfo(ser);
    let kernelSize = layerInfo.kernelSize;
    let numMaps = layerInfo.numMaps;
    let kernels = layerInfo.kernels;

    let cLayer = new CLayer(kernelSize, numMaps);
    cLayer.kernels = kernels;
    return cLayer;
  }

  /**
   * @param {string} ser String returned by serialize() function
   * @returns {string} a brief description of the layer
   */
  static getLayerDescription(ser){
    let layerInfo = CLayer.getLayerInfo(ser);
    let layerType = layerInfo.layerType;
    let kernelSize = layerInfo.kernelSize;
    let numMaps = layerInfo.numMaps;
    
    let description = "";
    description += layerType;
    description += " / kernelSize: " + kernelSize;
    description += " / numMaps: " + numMaps;

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

    this.height = prevLayer.height - (this.kernelSize - 1);
    this.width = prevLayer.width - (this.kernelSize - 1);

    if (!this.kernels){
      this.kernels = new Matrix(prevLayer.numMaps, this.numMaps);
      for (let c = 0; c < this.kernels.col; c++){
        for (let r = 0; r < this.kernels.row; r++){
          this.kernels[c][r] = new Matrix(this.kernelSize, this.kernelSize);
        }
      }
      this._heInitialization();
    }

  }
  
  /**
   * Initialize the kernel values using the He initialization scheme
   */
  _heInitialization(){
    let mapInputs = this.prevLayer.numMaps;
    let mapOutputs = this.numMaps;
    let kernelSize = this.kernelSize;
    let upperLimit = Math.sqrt(mapOutputs/(mapInputs+mapOutputs)/Math.pow(kernelSize,2));
    let lowerLimit = -1 * upperLimit;
    for (let i = 0; i < this.kernels.length; i++){
      for (let j = 0; j < this.kernels[i].length; j++){
        this.kernels[i][j].randUni(lowerLimit, upperLimit);
      }
    }
  }

  /**
   * Perform forward propagation.
   * Use the previous layer's output as the input.
   * The output is saved in instance's field (this.O)
   */
  forProp(){
    let I = this.prevLayer.O;
    this.I = I;
    let Y = new Array(this.numMaps);

    for (let r = 0; r < this.numMaps; r++){
      for (let c = 0; c < I.length; c++){   //EQUATION (1)
        let inputMap = I[c];
        let filteredMap = inputMap.correlation(this.kernels[c][r]);
        if (typeof(Y[r])!=="undefined") {
          Y[r].add(filteredMap);
        } else {
          Y[r] = filteredMap;
        }
      }
    }

    this.Y = Y;
    let O = new Array(this.numMaps);
    for (let i = 0; i < O.length; i++){
      O[i] = Y[i].applyByElement(ActivationUtil.relu);    //EQUATION (2)
    }
    this.O = O;
  }

  /**
   * Backward propagation.  
   * Pass the gradient information from next layer to the previous layer
   */
  backProp(){
    let dEdO = this.nextLayer.dEdI;

    let dY = new Array(this.numMaps);
    let O = this.O;

    for (let j = 0; j < this.numMaps; j++){
      let activationGrad = O[j].applyByElement(ActivationUtil.reluGrad);
      dY[j] = dEdO[j].hP(activationGrad);
    }

    let I = this.I;
    let dEdK = new Matrix(this.kernels.col, this.kernels.row);
    let dEdI = new Array(I.length);
    for (let i = 0; i < dEdK.col; i++){
      for (let j = 0; j < dEdK.row; j++){
        dEdK[i][j] = I[i].correlation(dY[j]);   //EQUATION (5)

        //EQUATION (4)
        let k = this.kernels[i][j];
        let kR180 = k.rotate180();
        let temp = kR180.fullCorrelation(dY[j]);
        temp = temp.rotate180();
        if (typeof(dEdI[i]) === "undefined") {
          dEdI[i] = temp;
        } else {
          dEdI[i].add(temp);
        }
      }
    }
    this.dEdI = dEdI;
    return dEdK;
  }

  /**
   * Add the parameter gradient 
   * to the instance's parameter gradient (this.dP)
   * 
   * @param {array} dP 2D-array containing the kernel gradient map Matrices
   */
  addDP(dP){
    if (!this.dP){
      this.dP = dP;
      return;
    }

    for (let i = 0; i < dP.col; i++){
      for (let j = 0; j < dP.row; j++){
        this.dP[i][j].add(dP[i][j]);
      }
    }
  }

  /**
   * Update the layer's parameters with the instance's field this.dP
   * 
   * @param {number} a Learning rate
   */
  updateParameters(a){
    if (this.dP){
      for (let i = 0; i < this.dP.col; i++){
        for (let j = 0; j < this.dP.row; j++){
          this.kernels[i][j].add(this.dP[i][j].multiply(-1*a));
        }
      }
    }
    this.dP = null;
  }
}

module.exports = CLayer;