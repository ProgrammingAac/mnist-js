/**
 * @module InLayer
 */

const Matrix = require('./Matrix.js');

/**
 * @author Aa C.
 * @class The representation of an Input layer in a neural network
 */
class InLayer{

  /**
   * @constructs
   * @param {number} height The height dimension of the input
   * @param {number} width The width dimension of the input. Default is 1.
   * @param {number} numMaps The number of input maps.
   */
  constructor(height, width, numMaps){
    this.height = height;

    if (width && numMaps && numMaps > 0) {       //map(s) of input, CNN
      this.width = width;
      this.numMaps = numMaps;
    } else {
      this.numNodes = height;
      this.width = 1;
      this.numMaps = 0;
    }
  }

  /**
   * Store the layer's information in a formatted string and returns that string
   * 
   * @returns {string} String containing the layer's information
   */
  serialize(){
    let ser = "";
    ser += "<" + this.constructor.name + ">";
    ser += "<" + this.height + ">";
    ser += "<" + this.width + ">";
    ser += "<" + this.numMaps + ">";

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
    let infoRegex = /^<([A-Za-z]+)><([0-9]+)><([0-9]+)><([0-9]+)>/;
    let info = infoRegex.exec(ser);
    let layerType = info[1];
    let height = JSON.parse(info[2]);
    let width = JSON.parse(info[3]);
    let numMaps = JSON.parse(info[4]);

    return {
      layerType: layerType,
      height: height,
      width: width,
      numMaps: numMaps
    };
  }

  /**
   * Create an instance of InLayer using the information in a serialized string (returned by serialize() function)
   * 
   * @param {string} ser String returned by serialize() function
   * @returns {object} an instance of InLayer
   */
  static deserialize(ser){
    let layerInfo = InLayer.getLayerInfo(ser);
    let height = layerInfo.height;
    let width = layerInfo.width;
    let numMaps = layerInfo.numMaps;
    
    let inLayer = new InLayer(height, width, numMaps);
    return inLayer;
  }

  /**
   * @param {string} ser String returned by serialize() function
   * @returns {string} a brief description of the layer
   */
  static getLayerDescription(ser){
    let layerInfo = InLayer.getLayerInfo(ser);
    let layerType = layerInfo.layerType;
    let height = layerInfo.height;
    let width = layerInfo.width;
    let numMaps = layerInfo.numMaps;

    let description = "";
    description += layerType;
    if (numMaps > 0) {
      description += " / height: " + height;
      description += " / width: " + width;
      description += " / numMaps: " + numMaps;
    } else {
      description += " / numNodes: " + height;
    }
    
    return description;
  }

  /**
   * Called when creating a neural network instance, in order to establish the sequence of different layers in a model.
   * 
   * @param {object} prevLayer The layer to be placed before this layer
   * @param {object} nextLayer The layer to be placed after this layer
   */
  link(nextLayer){
    this.nextLayer = nextLayer;
  }

  /**
   * Performing the forward propagation.
   * Using the the passed argument as the input.
   * The output is saved in instance's field (this.O)
   * 
   * @param {array} input The input values to the neural network
   */
  forProp(input){
    let I = input;
    let O;
    if (this.numMaps <= 1){      //ANN or CNN with 1-channel input
      if (I.length !== this.height * this.width) throw new Error("Wrong input dimensions.");
      if (this.numMaps === 0) {   //ANN
        O = Matrix.fromArray(I, this.width);
      } else {    //CNN with 1-channel input
        O = [Matrix.fromArray(I, this.width)];
      }
    } else {     //CNN with multiple-channel input
      if (I.length !== this.numMaps) throw new Error("Wrong number of channels.");
      O = new Array(this.numMaps);
      for (let i = 0; i < I.length; i++){
        let channelInput = I[i];
        if (channelInput.length !== this.height * this.width) throw new Error("Wrong input dimensions.");
        O[i] = Matrix.fromArray(channelInput, this.width);
      }
    }
    this.I = I;
    this.O = O;
  }

  /**
   * Place holder. InLayer is the first layer in a neural network and does not need to propagate the error further back.
   */
  backProp() {
  }
}

module.exports = InLayer;