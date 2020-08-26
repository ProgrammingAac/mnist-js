//Written by Aa C. (ProgrammingAac@gmail.com)
class RLayer{
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
    let layerInfo = RLayer.getLayerInfo(ser);
    let numNodes = layerInfo.numNodes;
    let W = Matrix.from2DArray(layerInfo.W);
    
    let rLayer = new RLayer(numNodes);
    rLayer.W = W;
    return rLayer;
  }

  static getLayerDescription(ser){
    let layerInfo = RLayer.getLayerInfo(ser);
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
    
    let prevType = prevLayer.constructor.name;
    if (prevType === "CLayer" || prevType === "ALayer") {
      throw new Error("CLayer or ALayer must not be linked before RLayer");
    } else if (prevType === "InLayer" && prevLayer.numMaps > 0) {
      throw new Error("Multiple channels must be flattened before linking to R-Layers.  Try inserting FLayer between InLayer and RLayer");
    }
    
    let col = prevLayer.numNodes;
    this.col = col;

    if (!this.W){
      this.W = new Matrix(col + this.numNodes, this.numNodes);

      this._heInitialization();
    } 
      
    this.Wi = new Matrix(col, this.numNodes);
    for (let i = 0; i < col; i++) {
      this.Wi[i] = this.W[i];
    }

    this.Wr = new Matrix(this.numNodes, this.numNodes);
    for (let i = col; i < col + this.numnodes; i++) {
      this.Wr[i-col] = this.W[i];
    }
    
    this.Is = [];
    this.Ss = [];
    this.Os = [];
  }


  forProp(){

    let Ii = this.prevLayer.O;
    
    let O_last;
    if (this.Os.length > 0) {
      O_last = this.Os[this.Os.length-1];
    } else {
      O_last = new Matrix(1, this.numNodes);
    }
    let Ir = O_last;

    let I = Matrix.concatVectors(Ii, Ir);
    this.Is.push(I.copy());

    let S = this.W.dot(I);
    this.Ss.push(S.copy());

    this.O = S.applyByElement(this.activation);
    this.Os.push(this.O.copy());

    if (this.isEnd){
      this.O = this.O.toArray();
    }
  }



  backProp(networkError){
    let dEdO;
    if (this.isEnd){
      if (typeof(networkError) === "undefined") {
        throw new Error("Network error is undefined");
      }
      dEdO = new Matrix(1, networkError.length);
      for (let i = 0; i < networkError.length; i++){
        dEdO[0][i] = networkError[i];
      }
    } else {
      dEdO = this.nextLayer.dEdI;
    }

    let steps = this.Ss.length;
    let tMax = steps - 1;

    let dEdW = new Matrix(this.W.col, this.W.row);

    let dEdI = new Matrix(1, this.prevLayer.numNodes + this.numNodes);

    let dEdOs = new Array(steps);
    dEdOs[tMax] = dEdO;

    let W_T = this.W.transpose();

    for (let t = tMax; t >= 0 ; t--) {
      let dPhi = this.Ss[t].applyByElement(this.activationGrad);
      let dEdS = dEdOs[t].hP(dPhi);

      let I_T = this.Is[t].transpose();
      dEdW.add(dEdS.dot(I_T));

      let dEdI_curr = W_T.dot(dEdS);
      dEdI.add(dEdI_curr);

      if (t-1 >= 0) {
        let pos1 = this.prevLayer.numNodes;
        let pos2 = pos1 + this.numNodes;

        dEdOs[t-1] = dEdI_curr.sliceVector(pos1, pos2);
      }
    }

    let pos1 = 0;
    let pos2 = this.prevLayer.numNodes;
    this.dEdI = dEdI.sliceVector(pos1, pos2);

    return dEdW;
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

  resetMemory() {
    this.Is = [];
    this.Ss = [];
    this.Os = [];
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
  module.exports = RLayer;
}