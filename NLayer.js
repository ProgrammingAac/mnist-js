//Written by Aa C. (ProgrammingAac@gmail.com)
class NLayer{
  constructor(numNodes){
    this.numNodes = numNodes;
    this.activation = ActivationUtil.relu;
    this.activationGrad = ActivationUtil.getGradFunc(this.activation);
  }

  serialize(){
    let ser = "";
    ser += "<" + this.constructor.name + ">";
    ser += "<" + this.numNodes + ">";
    ser += JSON.stringify(this.W);
    return ser;
  }

  static getLayerInfo(ser){
    let infoRegex = /^<([A-Za-z]+)><([0-9]+)>(.+)/;
    let info = infoRegex.exec(ser);
    let layerType = info[1];
    let numNodes = JSON.parse(info[2]);
    let W = JSON.parse(info[3]);

    return {
      layerType: layerType,
      numNodes: numNodes,
      W: W
    };
  }

  static deserialize(ser){
    let layerInfo = NLayer.getLayerInfo(ser);
    let numNodes = layerInfo.numNodes;
    let W = Matrix.from2DArray(layerInfo.W);
    
    let nLayer = new NLayer(numNodes);
    nLayer.W = W;
    return nLayer;
  }

  static getLayerDescription(ser){
    let layerInfo = NLayer.getLayerInfo(ser);
    let layerType = layerInfo.layerType;
    let numNodes = layerInfo.numNodes;

    let description = "";
    description += layerType;
    description += " / numNodes: " + numNodes;

    return description;
  }

  link(prevLayer, nextLayer){
    if (prevLayer){
      this.prevLayer = prevLayer;
    }
    if (nextLayer){
      this.nextLayer = nextLayer;
    } else this.isEnd = true;
    
    if (!this.W){
      
      let prevType = prevLayer.constructor.name;
      if (prevType === "CLayer" || prevType === "ALayer") {
        throw new Error("CLayer or ALayer must not be linked before NLayer");
      } else if (prevType === "InLayer" && prevLayer.numMaps > 0) {
        throw new Error("Multiple channels must be flattened before linking to N-Layers.  Try inserting FLayer between InLayer and NLayer");
      }

      this.W = new Matrix(prevLayer.numNodes, this.numNodes);

      this._heInitialization();
    }
    
  }

  forProp(){
    this.I = this.prevLayer.O;

    this.S = this.W.dot(this.I);      //EQUATION (1)
    this.O = this.S.applyByElement(this.activation);

    if (this.isEnd){
      this.O = this.O.toArray();
    }
  }

  backProp(networkError){
    let dEdO;
    if (this.isEnd){
      if (typeof(networkError) === "undefined") throw new Error("Network error is undefined");
      dEdO = new Matrix(1, networkError.length);
      for (let i = 0; i < networkError.length; i++){
        dEdO[0][i] = networkError[i];
      }
    } else {
      dEdO = this.nextLayer.dEdI;
    }

    let dPhi = this.S.applyByElement(this.activationGrad);
    let dEdS = dEdO.hP(dPhi);     //EQUATION (2)

    let iT = this.I.transpose();

    let wT = this.W.transpose();
    
    let dW = dEdS.dot(iT);        //EQUATION (4)

    this.dEdI = wT.dot(dEdS);     //EQUATION (3)

    return dW;
  }

  addDP(dW){
    if (this.dW) {
      this.dW.add(dW);
    } else {
      this.dW = dW;
    }
  }

  updateParameters(a){
    if (this.dW){
      this.W.add(this.dW.multiply(-1*a));
    }
    this.dW = null;
  }

  _randomInitialization(){
    this.W.randUni(-0.5, 0.5);
  }

  _heInitialization(){
    let numInputs = this.W.col;
    let upperLimit = Math.sqrt(2 / numInputs);
    let lowerLimit = -1 * upperLimit;

    this.W.randUni(lowerLimit, upperLimit);
  }
}


if(typeof process === 'object'){
  ActivationUtil = require('./ActivationUtil.js');
  module.exports = NLayer;
}