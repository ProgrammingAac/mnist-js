/**
 * @module FLayer
 */

const Matrix = require('./Matrix.js');

/**
 * @author Aa C.
 * @class The representation of a flatten/fully-connected layer in a convolutional neural network (CNN)
 */
class FLayer {

  constructor() {
  }

  /**
   * Store the layer's information in a formatted string, and return that string
   * 
   * @returns {string} String containing the layer's information
   */
  serialize() {
    let ser = "<" + this.constructor.name + ">";
    return ser;
  }

  /**
   * Extract information from the serialized string returned by serialize() function 
   * and return those information as an object literal
   * 
   * @param {string} ser String returned by serialize() function
   * @returns {object} object literal containing the layer's information
   */
  static getLayerInfo(ser) {
    let infoRegex = /^<([A-Za-z]+)>/;
    let info = infoRegex.exec(ser);
    let layerType = info[1];
    return {
      layerType: layerType
    }
  }

  /**
   * Create an instance of FLayer using the information in a serialized string (returned by serialize() function)
   * 
   * @param {string} ser String returned by serialize() function
   * @returns {object} an instance of FLayer
   */
  static deserialize(ser) {
    return new FLayer();
  }

  /**
   * @param {string} ser String returned by serialize() function
   * @returns {string} a brief description of the layer
   */
  static getLayerDescription(ser) {
    let layerInfo = FLayer.getLayerInfo(ser);
    let layerType = layerInfo.layerType;

    let description = "";
    description += layerType;

    return description;
  }

  /**
   * Called when creating a neural network instance, in order to establish the sequence of different layers in a model.
   * 
   * @param {object} prevLayer The layer to be placed before this layer
   * @param {object} nextLayer The layer to be placed after this layer
   */
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
  
  /**
   * Perform forward propagation.
   * Use the previous layer's output as the input.
   * The output is saved in instance's field (this.O)
   */
  forProp() {
    let I = this.prevLayer.O;

    let arr = [];
    for (let i = 0; i < I.length; i++) {
      let mapArray = I[i].toArray();
      arr.push(...mapArray);
    }

    this.O = Matrix.fromArray(arr, 1);
  }

  /**
   * Backward propagation.  
   * Pass the gradient information from next layer to the previous layer
   */
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

  /**
   * Place holder. FLayer does not have any trainable parameters
   */
  addDP(){
    return;
  }

  /**
   * Place holder. FLayer does not have any trainable parameters
   */
  updateParameters(){
    return;
  }
}

module.exports = FLayer;