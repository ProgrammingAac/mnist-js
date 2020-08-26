//Written by Aa C. (ProgrammingAac@gmail.com)
class ALayer{
  constructor(kernelSize){
    this.kernelSize = kernelSize;
  }

  //return 
  serialize(){
    let ser = "";
    ser += "<" + this.constructor.name + ">";
    ser += "<" + this.kernelSize + ">";

    return ser;
  }

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

  static deserialize(ser){
    let layerInfo = ALayer.getLayerInfo(ser);
    let kernelSize = layerInfo.kernelSize;

    let aLayer = new ALayer(kernelSize);
    return aLayer;
  }

  static getLayerDescription(ser){
    let layerInfo = ALayer.getLayerInfo(ser);
    let layerType = layerInfo.layerType;
    let kernelSize = layerInfo.kernelSize;

    let description = "";
    description += layerType;
    description += " / kernelSize: " + kernelSize;

    return description;
  }

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

  backProp(){
    let dEdO = this.nextLayer.dEdI;

    let dEdI = new Array(this.numMaps);
    for (let i = 0; i < this.numMaps; i++){
      dEdI[i] = this.avgPoolGrad(dEdO[i]);
    }
      
    this.dEdI = dEdI;
    return null;
  }

  addDP(){
    return;
  }

  updateParameters(){
    return;
  }
}

if(typeof process === 'object'){
  ActivationUtil = require('./ActivationUtil.js');
  module.exports = ALayer;
}