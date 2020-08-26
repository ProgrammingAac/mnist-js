//Written by Aa C. (ProgrammingAac@gmail.com)
class FLayer {    //Flatten Layer
  constructor() {
  }

  serialize() {
    let ser = "<" + this.constructor.name + ">";
    return ser;
  }

  static getLayerInfo(ser) {
    let infoRegex = /^<([A-Za-z]+)>/;
    let info = infoRegex.exec(ser);
    let layerType = info[1];
    return {
      layerType: layerType
    }
  }

  static deserialize(ser) {
    return new FLayer();
  }

  static getLayerDescription(ser) {
    let layerInfo = FLayer.getLayerInfo(ser);
    let layerType = layerInfo.layerType;

    let description = "";
    description += layerType;

    return description;
  }

  link(prevLayer, nextLayer) {

    if (prevLayer) {
      this.prevLayer = prevLayer;
    }
    if (nextLayer) {
      this.nextLayer = nextLayer;
      let nextType = nextLayer.constructor.name;
      if (nextType !== "NLayer" && nextType !== "LSTMLayer") {
        throw new Error("NLayer must be linked after FLayer or LSTMLayer");
      }
    }

    let prevType = prevLayer.constructor.name;
    if (prevType === "NLayer" || prevType === "FLayer") {
      throw new Error("FLayer must not be linked after NLayer or FLayer");
    } else {
      this.numNodes = prevLayer.numMaps * prevLayer.height * prevLayer.width;
    }

    
  }
  
  forProp() {
    let I = this.prevLayer.O;

    let arr = [];
    for (let i = 0; i < I.length; i++) {
      let mapArray = I[i].toArray();
      arr.push(...mapArray);
    }

    this.O = Matrix.fromArray(arr, 1);
  }

  backProp() {
    let dEdO = this.nextLayer.dEdI;
    let arr = dEdO.toArray();

    let numMaps = this.prevLayer.numMaps;
    let mapLength = this.prevLayer.width * this.prevLayer.height;
    let dEdI = new Array(numMaps);
    for (let i = 0; i < numMaps; i++) {
      let currStart = i * mapLength;
      let currEnd = (i+1) * mapLength;
      let mapArray = arr.slice(currStart, currEnd);
      dEdI[i] = Matrix.fromArray(mapArray, this.prevLayer.width);
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
  module.exports = FLayer;
}