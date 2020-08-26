//Written by Aa C. (ProgrammingAac@gmail.com)
class Matrix extends Array{
  constructor(col, row){
    super(col);
    for (let c = 0; c < col; c++){
      this[c] = new Array(row);
      for (let r = 0; r < row; r++){
        this[c][r] = 0;
      }
    }

    this.col = col;
    this.row = row;
  }

  static fromArray(arr, col){
    let row = arr.length / col;
    let result = new Matrix(col, row);

    for (let i = 0; i < arr.length; i++){
      result[Math.floor(i/row)][i%row] = arr[i];
    }

    return result;
  }

  static from2DArray(arr){
    let col = arr.length;
    let row = arr[0].length;
    let result = new Matrix(col, row);
    for (let c = 0; c < result.col; c++){
      for (let r = 0; r < result.row; r++){
        result[c][r] = arr[c][r];
      }
    }

    return result;
  }

  dot(m){
    let result = new Matrix(m.col, this.row);
    for (let cI = 0; cI < result.col; cI++){
      for (let rI = 0; rI < result.row; rI++){
        let sum = 0;
        for (let i = 0; i < this.col; i++){
          sum += this[i][rI] * m[cI][i];
        }
        result[cI][rI] = sum;
      }
    }
    return result;
  }

  vectorize(){
    let result = new Matrix(1, this.col * this.row);
    for (let i = 0; i < this.col*this.row; i++){
      result[0][i] = this[Math.floor(i/this.col)][i%this.col];
    }
    return result;
  }

  static concatVectors(v1, v2) {
    if (v1.col !== 1) throw new Error("v1 is not a vector");
    if (v2.col !== 1) throw new Error("v2 is not a vector");

    let result = new Matrix(1, v1.row + v2.row);

    for (let i = 0; i < v1.row; i++) {
      result[0][i] = v1[0][i];
    }
    for (let i = v1.row; i < v1.row + v2.row; i++) {
      result[0][i] = v2[0][i - v1.row];
    }

    return result;
  }

  sliceVector(pos1, pos2) {
    let arr = this.toArray();
    let result = Matrix.fromArray(arr.slice(pos1,pos2), 1);
    
    return result;
  }

  matrixize(numOfVectors){
    let col = numOfVectors;
    let row = this.col * this.row / col;
    let result = new Matrix(col, row);

    for (let i = 0; i < col*row; i++){
      result[i/col][i%col] = this[Math.floor(i/this.col)][i%this.col];
    }

    return result;
  }

  hP(m){
    if (this.col !== m.col) throw new Error("The col length do not match");
    if (this.row !== m.row) throw new Error("The row length do not match");

    let result = new Matrix(this.col, this.row);

    for (let c = 0; c < this.col; c++){
      for (let r = 0; r < this.row; r++){
        result[c][r] = this[c][r] * m[c][r];
      }
    }

    return result;
  }

  transpose(){
    let result = new Matrix(this.row, this.col);

    for (let c = 0; c < result.col; c++){
      for (let r = 0; r < result.row; r++){
        result[c][r] = this[r][c];
      }
    }

    return result;
  }

  add(m){
    if (this.col !== m.col) throw new Error("The col length do not match");
    if (this.row !== m.row) throw new Error("The row length do not match");

    for (let c = 0; c < this.col; c++){
      for (let r = 0; r < this.row; r++){
        this[c][r] = this[c][r] + m[c][r];
      }
    }

  }

  multiply(a){
    let result = new Matrix(this.col, this.row);

    for (let c = 0; c < result.col; c++){
      for (let r = 0; r < result.row; r++){
        result[c][r] = a * this[c][r];
      }
    }

    return result;
  }

  toArray(){
    let result = new Array(this.col * this.row);
    
    for (let i = 0; i < result.length; i++){
      result[i] = this[Math.floor(i/this.row)][i%this.row];
    }

    return result;
  }

  correlation(k){
    if (!(k instanceof Matrix)) throw Error("kernel is not an instance of Matrix");
    let downSize = k.col - 1;
    let rowLength = this.row - downSize;
    let colLength = this.col - downSize;
    let result = new Matrix(colLength, rowLength);

    for (let c = 0; c < result.col; c++){
      for (let r = 0; r < result.row; r++){
        let sum = 0;
        for (let kC = 0; kC < k.col; kC++){
          for (let kR = 0; kR < k.row; kR++){
            sum += this[c+kC][r+kR] * k[kC][kR];
          }
        }
        result[c][r] = sum;
      }
    }
    
    return result;
  }

  fullCorrelation(k){
    let pad = k.col - 1;
    let padMCol = this.col + pad*2;
    let padMRow = this.row + pad*2;
    let padM = new Matrix(padMCol, padMRow);

    for (let r = 0; r < pad; r++){
      for (let c = 0; c < padM.col; c++){
        padM[c][r] = 0;
      }
    }

    for (let r = pad; r < pad+this.row; r++){
      for (let c = 0; c < pad; c++) padM[c][r] = 0;
      for (let c = pad; c < pad+this.col; c++){
        let originalC = c-pad;
        let originalR = r-pad;
        padM[c][r] = this[originalC][originalR];
      }
      for (let c = 0; c < pad; c++) padM[c][r] = 0;
    }

    for (let r = 0; r < pad; r++){
      for (let c = 0; c < padM.col; c++){
        padM[c][r] = 0;
      }
    }

    return padM.correlation(k);
  }

  rotate180(){
    let result = new Matrix(this.col, this.row);

    for (let c = 0; c < result.col; c++){
      let mCol = this.col - c - 1;
      for (let r = 0; r < result.row; r++){
        let mRow = this.row - r - 1;
        result[c][r] = this[mCol][mRow];
      }
    }

    return result;
  }

  rand() {
    for (let c = 0; c < this.col; c++){
      for (let r = 0; r < this.row; r++){
        this[c][r] = Math.random() - 0.5;
      }
    }
  }

  randUni(lowerLimit, upperLimit){
    let numInputs = this.col;
    let numOutputs = this.row;
    let numIntervals = numInputs * numOutputs;
    let interval = (upperLimit - lowerLimit) / numIntervals;
    let pool = new Array(numIntervals);
    for (let i = 0; i < numIntervals; i++){
      pool[i] = lowerLimit + i * interval;
    }
    for (let c = 0; c < this.col; c++){
      for (let r = 0; r < this.row; r++){
        let fromPool = pool.splice(Math.floor(Math.random() * (pool.length-1)), 1)[0];
        this[c][r] = fromPool;
      }
    }
  }

  applyByElement(func){
    let result = new Matrix(this.col, this.row);

    for (let c = 0; c < this.col; c++){
      for (let r = 0; r < this.row; r++){
        result[c][r] = func(this[c][r]);
      }
    }

    return result;
  }

  copy() {
    let m = new Matrix(this.col, this.row);
    for (let c = 0; c < this.col; c++) {
      for (let r = 0; r < this.row; r++) {
        m[c][r] = this[c][r];
      }
    }
    return m;
  }

}

if(typeof process === 'object'){
  module.exports = Matrix;
}