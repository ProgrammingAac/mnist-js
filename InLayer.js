//Written by Aa C. (ProgrammingAac@gmail.com)
class InLayer{
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

  serialize(){
    let ser = "";
    ser += "<" + this.constructor.name + ">";
    ser += "<" + this.height + ">";
    ser += "<" + this.width + ">";
    ser += "<" + this.numMaps + ">";

    return ser;
  }

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

  static deserialize(ser){
    let layerInfo = InLayer.getLayerInfo(ser);
    let height = layerInfo.height;
    let width = layerInfo.width;
    let numMaps = layerInfo.numMaps;
    
    let inLayer = new InLayer(height, width, numMaps);
    return inLayer;
  }

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

  link(nextLayer){
    this.nextLayer = nextLayer;
  }

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

  backProp() {
    
  }
}

if(typeof process === 'object'){
  module.exports = InLayer;
}