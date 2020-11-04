/**
 * @module NLayer
 */

const ActivationUtil = require('./ActivationUtil.js');
const Matrix = require('./Matrix.js');

/**
 * @author Aa C.
 * @class The representation of the basic layer in a neural network
 */
class NLayer{

  /**
   * @constructs
   * @param {number} numNodes The number of neuron nodes in the layer
   */
  constructor(numNodes){
    this.numNodes = numNodes;
    this.activation = ActivationUtil.relu;
    this.activationGrad = ActivationUtil.getGradFunc(this.activation);
  }

  /**
   * Store the layer's information in a formatted string and returns that string
   * 
   * @returns {string} String containing the layer's information
   */
  serialize(){
    let ser = "";
    ser += "<" + this.constructor.name + ">";
    ser += "<" + this.numNodes + ">";
    ser += JSON.stringify(this.W);
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

  /**
   * Create an instance of NLayer using the information in a serialized string (returned by serialize() function)
   * 
   * @param {string} ser String returned by serialize() function
   * @returns {object} an instance of NLayer
   */
  static deserialize(ser){
    let layerInfo = NLayer.getLayerInfo(ser);
    let numNodes = layerInfo.numNodes;
    let W = Matrix.from2DArray(layerInfo.W);
    
    let nLayer = new NLayer(numNodes);
    nLayer.W = W;
    return nLayer;
  }

  /**
   * @param {string} ser String returned by serialize() function
   * @returns {string} a brief description of the layer
   */
  static getLayerDescription(ser){
    let layerInfo = NLayer.getLayerInfo(ser);
    let layerType = layerInfo.layerType;
    let numNodes = layerInfo.numNodes;

    let description = "";
    description += layerType;
    description += " / numNodes: " + numNodes;

    return description;
  }

  /**
   * Called when creating a neural network instance, in order to establish the sequence of different layers in a model.
   * 
   * @param {object} prevLayer The layer to be placed before this layer
   * @param {object} nextLayer The layer to be placed after this layer
   */
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

  /**
   * Performing the forward propagation.
   * Using the previous layer's output as the input.
   * The output is saved in instance's field (this.O)
   */
  forProp(){
    this.I = this.prevLayer.O;

    this.S = this.W.dot(this.I);      //EQUATION (1)
    this.O = this.S.applyByElement(this.activation);

    if (this.isEnd){
      this.O = this.O.toArray();
    }
  }


  /**
   * Backward propagation.  
   * Pass the gradient information from next layer to the previous layer
   * 
   * @param {array} networkError Optional, only requires when the layer is the last layer in the model
   */
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

  /**
   * Add the wegihts gradient
   * to the instance's weights gradient (this.dW)
   * 
   * @param {Matrix} dW Matrix containing the weights gradient
   */
  addDP(dW){
    if (this.dW) {
      this.dW.add(dW);
    } else {
      this.dW = dW;
    }
  }

  /**
   * Update the layer's connection weights with the instance's field this.dW
   * 
   * @param {number} a Learning rate
   */
  updateParameters(a){
    if (this.dW){
      this.W.add(this.dW.multiply(-1*a));
    }
    this.dW = null;
  }

  /**
   * Initialize the connection weights using the random uniform distribution scheme
   */
  _randomInitialization(){
    this.W.randUni(-0.5, 0.5);
  }

  /**
   * Initialize the connection weights using the He initialization scheme
   */
  _heInitialization(){
    let numInputs = this.W.col;
    let upperLimit = Math.sqrt(2 / numInputs);
    let lowerLimit = -1 * upperLimit;

    this.W.randUni(lowerLimit, upperLimit);
  }
}

module.exports = NLayer;