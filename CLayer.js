//Written by Aa C. (ProgrammingAac@gmail.com)
class CLayer{
  constructor(kernelSize, numMaps){
    this.kernelSize = kernelSize;
    this.numMaps = numMaps;
  }

  serialize(){
    let ser = "";
    ser += "<" + this.constructor.name + ">";
    ser += "<" + this.kernelSize + ">";
    ser += "<" + this.numMaps + ">";
    ser += JSON.stringify(this.kernels);

    return ser;
  }

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

  static deserialize(ser){
    let layerInfo = CLayer.getLayerInfo(ser);
    let kernelSize = layerInfo.kernelSize;
    let numMaps = layerInfo.numMaps;
    let kernels = layerInfo.kernels;

    let cLayer = new CLayer(kernelSize, numMaps);
    cLayer.kernels = kernels;
    return cLayer;
  }

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

if(typeof process === 'object'){
  ActivationUtil = require('./ActivationUtil.js');
  module.exports = CLayer;
}