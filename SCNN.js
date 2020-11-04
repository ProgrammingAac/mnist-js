/**
 * @module Model
 */

const InLayer = require('./InLayer.js');
const CLayer = require('./CLayer.js');
const ALayer = require('./ALayer.js');
const FLayer = require('./FLayer.js');
const RLayer = require('./RLayer.js');
const NLayer = require('./NLayer.js');

/**
 * @author Aa C.
 * @class The representation of a neural network
 */
class Model {

  /**
   * @constructs
   * @param {number} a Learning rate
   * @param {array} layers Array containing the layers constituting the neural network
   * @param {string} errFunc String that indicate the type of error function used. "MSE" means mean-squared-error. Default is "MSE".
   */
  constructor(a, layers, errFunc){
    this.a = a;

    this.isRNN = false;

    this.layers = new Array(layers.length);
    for (let i = 0; i < layers.length; i++){
      if (layers[i] instanceof RLayer) {
        this.isRNN = true;
      }
      this.layers[i] = layers[i];
    }

    //Linking the layers
    this.layers[0].link(this.layers[1]);
    for (let i = 1; i < this.layers.length - 1; i++){
      this.layers[i].link(this.layers[i-1], this.layers[i+1]);
    }
    let lastLayer = this.layers[this.layers.length - 1];
    lastLayer.link(this.layers[this.layers.length - 2], null);
  
    switch(errFunc){
      case "MSE":
        this.errGrad = this._mseGrad;
        break;

      case undefined:
        this.errGrad = this._mseGrad;
        break;
    }
  }

  /**
   * Store the neural network model's information in a formatted string, and return that string
   * 
   * @returns {string} String containing the neura network model's information
   */
  serialize(){
    let ser = "";
    ser += this.a.toString();
    ser += "|";
    ser += this.layers[0].serialize();
    for (let i = 1; i < this.layers.length; i++){
      ser += "/";
      ser += this.layers[i].serialize();
    }
    return ser;
  }

  /**
   * Create an instance of Model using the information in a serialized string (returned by serialize() function)
   * 
   * @param {string} ser String returned by serialize() function
   * @returns {object} an instance of Model
   */
  static deserialize(ser){
    let sections = ser.split("|");

    let a = sections[0];
    a = Number.parseFloat(a);

    let layerInfos = sections[1].split("/");
    let typeRegex = /^<([A-Za-z]+)>.?/;
    let layers = [];
    for (let i = 0; i < layerInfos.length; i++){
      let layerType = typeRegex.exec(layerInfos[i])[1];
      let layer;
      switch(layerType){
        case "InLayer":
          layer = InLayer.deserialize(layerInfos[i]);
          break;
        case "CLayer":
          layer = CLayer.deserialize(layerInfos[i]);
          break;
        case "ALayer":
          layer = ALayer.deserialize(layerInfos[i]);
          break;
        case "FLayer":
          layer = FLayer.deserialize(layerInfos[i]);
          break;
        case "NLayer":
          layer = NLayer.deserialize(layerInfos[i]);
          break;
        case "RLayer":
          layer = RLayer.deserialize(layerInfos[i]);
          break;
      }
      layers.push(layer);
    }

    let model = new Model(a, layers);
    return model;
  }

  /**
   * Reset the memory of the R-Layers inside this neural network model.
   * It is a RNN-exclusive function.
   */
  resetMemory() {
    if (!this.isRNN) {
      throw new Error("Only RNNs have access to resetMemory()");
    }
    let layers = this.layers;
    for (let i = 0; i < layers.length; i++) {
      let layer = layers[i];
      if (layer instanceof RLayer) {
        layer.resetMemory();
      }
    }
  }

  /**
   * Perform forward propagation,
   * The output is saved in instance's field (this.output)
   * 
   * @param {array} input Array containing the input to be used for forward propagation
   */
  forProp(input) {
    let layers = this.layers;
    layers[0].forProp(input);
    for (let i = 1; i < layers.length; i++){
      layers[i].forProp();
    }
    this.output = layers[layers.length-1].O;
    return this.output;
  }

  /**
   * Perform forawrd propagation with RNN. 
   * For simplicity, this is only the "many-to-one" case. 
   * It is a RNN-exclusive function.
   * The output is saved in instance's field (this.output)
   * 
   * @param {array} inputs Array that contains sequence of arrays of network input data
   */
  forPropSeries(inputs) {
    if (!this.isRNN) {
      throw new Error("Only RNNs have access to forPropSeries()");
    }
    this.resetMemory();
    for (let i = 0; i < inputs.length; i++) {
      let input = inputs[i];
      this.forProp(input);
    }
    return this.output;
  }

  /**
   * Perform backward propagation.
   * Pass the gradient information from the last layer backwards each layer at a time,
   * until reaching the first layer
   * 
   * @param {array} input Array of input data
   * @param {array} target The target/desired network output for calculating the network error
   * @returns {array} Array containing the parameter gradient for each layer
   */
  backProp(input, target) {
    console.log("Target:", target);
    let output = this.forProp(input);
    console.log("ForProp:", output);
    console.log("");
    let layers = this.layers;
    let dP = new Array(layers.length);
    let err = this.errGrad(output, target);
    dP[layers.length - 1] = layers[layers.length - 1].backProp(err);
    for (let l = layers.length - 2; l > 0; l--){
      dP[l] = layers[l].backProp();
    }
    return dP;
  }

  /**
   * Perform backward propagation through time with RNN. 
   * For simplicity, this is only the "many-to-one" case. 
   * It is a RNN-exclusive function.
   * 
   * @param {array} inputs Array that contains sequence of arrays of network input data
   * @param {array} target The target/desired network output for calculating the network error
   * @returns {array} Array containing the parameter gradient for each layer
   */
  backPropSeries(inputs, target) {
    if (!this.isRNN) {
      throw new Error("Only RNNs have access to backPropSeries()");
    }
    
    let layers = this.layers;

    this.resetMemory();
    
    let output = this.forPropSeries(inputs);
    console.log("Target:", target);
    console.log("Output:", output);
    console.log("");
    let err = this._mseGrad(output, target);
    let dP = new Array(layers.length);
    dP[layers.length - 1] = layers[layers.length - 1].backProp(err);
    for (let l = layers.length - 2; l > 0; l--) {
      dP[l] = layers[l].backProp();
    }

    return dP;
  }

  /**
   * Calcuate the error between the 
   * network output and the target/desired output 
   * using the mean-squared-error approach
   * 
   * @param {array} output Output from neural network
   * @param {array} target The target/desired network output for calculating the network error
   * @returns {array} Array of the calculated output error
   */
  _mseGrad(output, target){
    if (output.length !== target.length) throw new Error("output length does not match target length");
    let err = new Array(output.length);
    for (let i = 0; i < output.length; i++){
      err[i] = output[i] - target[i];
    }
    return err;
  }

  /**
   * Train the neural network model through a input/target-output pair dataset
   * 
   * @param {array} inputs Array of arrays containing the network input
   * @param {array} targets Array of arrays containing the target/desired output.
   * @param {number} batch The interval between updating the model's parameters
   */
  train(inputs, targets, batch){
    for (let j = 0; j < inputs.length; j+=batch){
      for (let b = 0; b < batch; b++){
        if (j + b < inputs.length){
          // console.log((j+b+1) + " / " + inputs.length);

          let dP;
          if (!this.isRNN) {
            dP = this.backProp(inputs[j+b], targets[j+b]);
          } else {
            dP = this.backPropSeries(inputs[j+b], targets[j+b]);
          }
          for (let l = 0; l < this.layers.length; l++){
            if(dP[l]) this.layers[l].addDP(dP[l]);
          }
        } else break;
      }
      this.updateParameters();
    }
  }

  /**
   * Test the accuracy of the model with a input/target-output pair dataset
   * 
   * @param {array} inputs Array of arrays containing the network input
   * @param {array} targets Array of arrays containing the target/desired output.
   * @returns {object} Object literal containing testSamples (the number of test samples), and correctCount (the number of correct output count)
   */
  test(inputs, targets){
    let correctCount = 0;
    for (let i = 0; i < inputs.length; i++) {
      let target = targets[i];
      let input = inputs[i];

      let output;
      if (!this.isRNN) {
        output = this.forProp(input);
      } else {
        output = this.forPropSeries(input);
      }

      let sortedOutput = output.slice();
      sortedOutput.sort((a,b) => {
        if (a < b) {return 1;}
        else if (a === b) {return 0;}
        else {return -1;}
      });

      let targetIndices = new Array();
      for (let j = 0; j < target.length; j++) {
        if (target[j] === 1) targetIndices.push(j);
      }

      let maxIndices = new Array(targetIndices.length);
      for (let j = 0; j < targetIndices.length; j++) {
        maxIndices[j] = output.indexOf(sortedOutput[j]);
      }
      let isCorrect = true;
      for (let j = 0; j < maxIndices.length; j++) {
        if (target[maxIndices[j]] !== 1) {
          isCorrect = false;
          break;
        }
      }
      if (isCorrect) correctCount++;
    }
    return {
      testSamples: inputs.length,
      correctCount: correctCount
    };
  }

  /**
   * Update the parameters for each layer in the neural network
   */
  updateParameters(){
    for (let i = 1; i < this.layers.length; i++){
      this.layers[i].updateParameters(this.a);
    }
  }

  /**
   * @returns {string} a description string of the model's structure.
   */
  get structureDescription() {
    let ser = this.serialize();
    let sections = ser.split("|");
    
    let result = "========================================\n";
    result += "Structure:\n";
    let layerData = sections[1].split("/");
    const nameRegex = /^<([A-za-z]+)>.?/;
    for (let i = 0; i < layerData.length; i++){
      let layerName = nameRegex.exec(layerData[i])[1];
      let layerDescription;
      switch(layerName){
        case "InLayer":
          layerDescription = InLayer.getLayerDescription(layerData[i]);
          break;
        case "CLayer":
          layerDescription = CLayer.getLayerDescription(layerData[i]);
          break;
        case "ALayer":
          layerDescription = ALayer.getLayerDescription(layerData[i]);
          break;
        case "FLayer":
          layerDescription = FLayer.getLayerDescription(layerData[i]);
          break;
        case "NLayer":
          layerDescription = NLayer.getLayerDescription(layerData[i]);
          break;
        case "RLayer":
          layerDescription = RLayer.getLayerDescription(layerData[i]);
          break;
      }
      result += "\t" + layerDescription + "\n";
      
    }
    result += "========================================\n";

    return result;
  }

}

module.exports = Model;

if (process.argv[1] === __dirname + "\\SCNN.js"){

  const genTrainingReport = function(hyperParamsDict, performanceDict) {
    let a = hyperParamsDict.a;
    let batch = hyperParamsDict.batch ? hyperParamsDict.batch : 1;
    let epoch = hyperParamsDict.epoch ? hyperParamsDict.epoch : 1;
    
    let result = "----------------------------------------\n";
    result += "a: " + JSON.stringify(a) + "\n";
    result += "batch: " + JSON.stringify(batch) + "\n";
    result += "epoch: " + JSON.stringify(epoch) + "\n";

    for (let key in performanceDict) {
      result += "-> " + JSON.stringify(key) + ": " + JSON.stringify(performanceDict[key]) + "\n";
    }

    result += "----------------------------------------\n";

    return result;
  }

  const getCurrTimeStamp = function() {
    let currDate = new Date();
    let currYear = currDate.getFullYear().toString();
    let currMonth = (currDate.getMonth() + 1).toString().padStart(2,"0");
    let currDay = currDate.getDate().toString().padStart(2,"0");
    let currHour = currDate.getHours().toString().padStart(2,"0");
    let currMin = currDate.getMinutes().toString().padStart(2,"0");
    timeStamp = currYear + currMonth + currDay + currHour + currMin;

    return timeStamp;
  }

  const readMNISTimages = function(imagePath) {
    const fs = require('fs');
    let imageFileBuffer = fs.readFileSync(imagePath);
    let images = [];
    for (let imgIndex = 16; imgIndex < imageFileBuffer.length; imgIndex+=28*28) {
      let imgPixs = new Array(28*28);

      for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
          let currImgPix = x + (y * 28);
          imgPixs[currImgPix] = imageFileBuffer[imgIndex + currImgPix]/256;
        }
      }
      
      images.push(imgPixs);
    }

    return images;
  }

  const readMNISTlabels = function(labelPath) {
    const fs = require('fs');
    let labelFileBuffer = fs.readFileSync(labelPath);
    let labels = [];

    for (let lblIndex = 8; lblIndex < labelFileBuffer.length; lblIndex++) {
      labels.push(labelFileBuffer[lblIndex]);
    }

    return labels;
  }
  
  if (process.argv.length < 2) {
    throw new Error("Invalid Input");
  }


  const fs = require('fs');
  const path = require('path');

  let runArgs = process.argv.slice(2);
  let runType = runArgs[0];

  let resultFolder = __dirname + "/Training Results/";
  if (!fs.existsSync(resultFolder)) {
    fs.mkdirSync(resultFolder);
  }

  resultFolder += runType + "/";
  if (!fs.existsSync(resultFolder)) {
    fs.mkdirSync(resultFolder);
  }

  const mnistTrainImagePath = __dirname + '/mnist/train-images.idx3-ubyte';
  const mnistTrainLabelPath = __dirname + '/mnist/train-labels.idx1-ubyte';
  const mnistTestImagePath = __dirname + '/mnist/t10k-images.idx3-ubyte';
  const mnistTestLabelPath = __dirname + '/mnist/t10k-labels.idx1-ubyte';

  let trainImages = readMNISTimages(mnistTrainImagePath);
  let trainLabels = readMNISTlabels(mnistTrainLabelPath);
  let testImages = readMNISTimages(mnistTestImagePath);
  let testLabels = readMNISTlabels(mnistTestLabelPath);

  let resultFilePath;
  let isNewRun;

  let timeStamp;
  let runStamp;

  let currEpoch;

  let a;
  let batch = 16;

  let model;

  const prepareLabels = function(labels) {
    for (let i = 0; i < labels.length; i++) {
      let label = labels[i];
      let resultLabel = new Array(10);
      for (let j = 0; j < resultLabel.length; j++) {
        resultLabel[j] = label === j ? 1 : 0;
      }
      labels[i] = resultLabel;
    }
  }

  const prepareRNNInputs = function(inputs) {
    for (let i = 0; i < inputs.length; i++) {
      let input = inputs[i];
      let resultInput = new Array(4);     //四格漫畫
      for (let j = 0; j < resultInput.length; j++) {
        resultInput[j] = new Array((28/2)*(28/2));
        let startRow = j < 2 ? 0 : 28/2;
        let startCol = j%2 === 0 ? 0 : 28/2; 
        for (let r = startRow; r < startRow + 28/2; r++) {
          for (let c = startCol; c < startCol + 28/2; c++) {
            let srcIndex = c + (r * 28);
            let resultIndex =  (c-startCol) + ((r-startRow)*(28/2));
            resultInput[j][resultIndex] = input[srcIndex];
          }
        }
      }
      inputs[i] = resultInput;
    }
  }


  prepareLabels(trainLabels);
  prepareLabels(testLabels);

  if (runArgs[1]) {     //continue from existing run
    isNewRun = false;
    timeStamp = runArgs[1];
    runStamp = timeStamp + "_" + runType;
    console.log("runStamp:", runStamp);

    let ser;
    for (let i = 1; i < 2048; i++) {
      let serialFilePath = __dirname + "/Training Results/" + runType + "/" + runStamp + "_epoch" + i + "_serial.txt";
      try {
        ser = fs.readFileSync(serialFilePath, "utf8");
      } catch (error) {
        console.log("Loaded: ", 
          path.resolve(__dirname + "/Training Results/" + runType + "/" + runStamp + "_epoch" + (i-1) + "_serial.txt"));
        currEpoch = i-1;
        break;
      }
    }

    model = Model.deserialize(ser);
    a = model.a;
  }

  if (!runArgs[1]) {      //new run
    isNewRun = true;
    runStamp = getCurrTimeStamp() + "_" + runType;
  }

  if (runType === "rnn") {
    prepareRNNInputs(trainImages);
    prepareRNNInputs(testImages);
  }

  resultFilePath = resultFolder + runStamp + "_result.txt";

  if (isNewRun) {
    let layers = [];

    if (runType === "ann") {
      a = 0.01;
      batch = 16;
      layers.push(new InLayer(784));
      layers.push(new NLayer(110));
      layers.push(new NLayer(10));
    }

    if (runType === "cnn") {
      a = 0.01;
      batch = 16;
      layers.push(new InLayer(28,28,1));
      layers.push(new CLayer(7,32));
      layers.push(new ALayer(2));
      layers.push(new CLayer(5,16));
      layers.push(new ALayer(2));
      layers.push(new FLayer());
      layers.push(new NLayer(10));
    }

    if (runType === "rnn") {
      a = 0.01;
      batch = 16;
      layers.push(new InLayer(196));  
      layers.push(new NLayer(784));
      layers.push(new RLayer(110));
      layers.push(new NLayer(10));
    }

    model = new Model(a, layers);
    currEpoch = 0;

    fs.appendFileSync(resultFilePath, model.structureDescription);
  }

  console.log("Run stamp:", runStamp);
  console.log();
  console.log("Model:", model);
  console.log();
  console.log("----------------------------------");

  let accuracy;

  model.train(trainImages, trainLabels, batch);
  
  let testResult = model.test(testImages, testLabels);
  accuracy = testResult.correctCount / testResult.testSamples;

  let performanceDict = {
    Accuracy: accuracy
  }
  
  let hyperParams = {
    a: a,
    batch: batch,
    epoch: currEpoch + 1
  }

  fs.appendFileSync(resultFilePath, genTrainingReport(hyperParams, performanceDict));
  
  let serialFilePath = resultFolder + runStamp + "_epoch" + (currEpoch+1) + "_serial.txt";
  fs.appendFileSync(serialFilePath, model.serialize());

  console.log("Training for epoch " + (currEpoch+1) + " has been completed.");
  console.log("Result saved to:", resultFolder);

  console.log("----------------------------------");


  const fork = require("child_process").fork;
  if (isNewRun) {
    fork(path.resolve(__dirname + "/SCNN.js"), [...runArgs, timeStamp]);
  } else {
    fork(path.resolve(__dirname + "/SCNN.js"), runArgs);
  }
}