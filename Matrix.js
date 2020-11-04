/**
 * @author Aa C.
 * @class The representation of a Matrix
 */
class Matrix extends Array{

  /**
   * @constructs
   * @param {number} col The number of columns in the matrix
   * @param {number} row The number of rows in the matrix
   */
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

  /**
   * Static method to create a Matrix instance from a 1-D array and a specified column number
   * 
   * @param {array} arr The 1-D array to be converted into a matrix
   * @param {number} col The column number of the matrix
   * @returns {Matrix} The matrix converted from the 1-D array
   */
  static fromArray(arr, col){
    let row = arr.length / col;
    let result = new Matrix(col, row);

    for (let i = 0; i < arr.length; i++){
      result[Math.floor(i/row)][i%row] = arr[i];
    }

    return result;
  }

  /**
   * Static method to create a Matrix instance from a 2-D array
   * 
   * @param {array} arr The 2-D array to be converted into a matrix
   * @returns {Matrix} The matrix converted from the 2-D array
   */
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

  /**
   * The dot operation.
   * The first operand is the Matrix instance itself.
   * 
   * @param {Matrix} m The other operand in the dot operation
   * @returns {Matrix} The result matrix of the dot operation
   */
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

  /**
   * Flatten/ vectorize the current matrix into a 1-D matrix (vector).
   * Then, return that 1-D matrix (vector).
   * 
   * @returns The vectorized 1-D matrix from the current matrix
   */
  vectorize(){
    let result = new Matrix(1, this.col * this.row);
    for (let i = 0; i < this.col*this.row; i++){
      result[0][i] = this[Math.floor(i/this.col)][i%this.col];
    }
    return result;
  }

  /**
   * concatenate vector v2 after v1 to form a longer 1-D matrix (vector)
   * 
   * @param {Matrix} v1 1-D matrix (vector)
   * @param {Matrix} v2 1-D matrix (vector)
   * @returns The concatenated 1-D matrix (vector), with v2 after v1
   */
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

  /**
   * Slice a portion of a vector.
   * 
   * @param {number} pos1 inclusive start position to slice
   * @param {number} pos2 exclusive end position to slice
   * @returns {Matrix} The sliced vector
   */
  sliceVector(pos1, pos2) {
    let arr = this.toArray();
    let result = Matrix.fromArray(arr.slice(pos1,pos2), 1);
    
    return result;
  }

  /**
   * Create a matrix from the current matrix and a specified number of vectors
   * 
   * @param {number} numOfVectors The number of vectors in the result matrix, i.e., the column number
   * @returns {Matrix} The result matrix
   */
  matrixize(numOfVectors){
    let col = numOfVectors;
    let row = this.col * this.row / col;
    let result = new Matrix(col, row);

    for (let i = 0; i < col*row; i++){
      result[i/col][i%col] = this[Math.floor(i/this.col)][i%this.col];
    }

    return result;
  }

  /**
   * Hadamard product operation.
   * The first operand is the Matrix instance itself
   * 
   * @param {Matrix} The other operand in the Hadamard product operation
   * @returns {Matrix} The result of the Hadamard product operation
   */
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

  /**
   * @returns {Matrix} Matrix transposed from the current Matrix instance
   */
  transpose(){
    let result = new Matrix(this.row, this.col);

    for (let c = 0; c < result.col; c++){
      for (let r = 0; r < result.row; r++){
        result[c][r] = this[r][c];
      }
    }

    return result;
  }

  /**
   * Element-wise Matrix addition.
   * The first operand is the Matrix instance itself.
   * 
   * @param {Matrix} m The other operand in the addition operation
   * @returns {Matrix} The result matrix of the Matrix addition operation
   */
  add(m){
    if (this.col !== m.col) throw new Error("The col length do not match");
    if (this.row !== m.row) throw new Error("The row length do not match");

    for (let c = 0; c < this.col; c++){
      for (let r = 0; r < this.row; r++){
        this[c][r] = this[c][r] + m[c][r];
      }
    }
  }

  /**
   * Scalar multiplication on the matrix instance
   * 
   * @param {number} a scalar in the operation
   * @returns {Matrix} The result matrix of the scalar multiplication
   */
  multiply(a){
    let result = new Matrix(this.col, this.row);

    for (let c = 0; c < result.col; c++){
      for (let r = 0; r < result.row; r++){
        result[c][r] = a * this[c][r];
      }
    }

    return result;
  }

  /**
   * @returns {array} Array obtained from converting the matrix instance into a 1-D array
   */
  toArray(){
    let result = new Array(this.col * this.row);
    
    for (let i = 0; i < result.length; i++){
      result[i] = this[Math.floor(i/this.row)][i%this.row];
    }

    return result;
  }

  /**
   * Correlation operation
   * 
   * @param {Matrix} k Kernel
   * @returns {Matrix} The result of the correlation operation
   */
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

  /**
   * Full correlation operation. 
   * Used in backward propagation of CLayer
   * 
   * @param {Matrix} k Kernel
   * @returns {Matrix} The result of the full correlation operation
   */
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

  /**
   * Rotate the matrix by 180 degress and return the rotated matrix
   * 
   * @returns {Matrix} The matrix instance but rotated by 180 degrees
   */
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

  /**
   * Fill the matrix elements by 
   * random numbers within the range of
   * [-0.5, 0.5]
   */
  rand() {
    for (let c = 0; c < this.col; c++){
      for (let r = 0; r < this.row; r++){
        this[c][r] = Math.random() - 0.5;
      }
    }
  }

  /**
   * Fill the matrix elements by the random uniform
   * distribution scheme
   * 
   * @param {number} lowerLimit The lower numerical limit of each value in the matrix
   * @param {number} upperLimit The upper numerical limit of each value in the matrix
   */
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

  /**
   * Pass each element in the matrix through a function and return the result
   * 
   * @param {function} func The function to be used
   * @returns {Matrix} The result matrix
   */
  applyByElement(func){
    let result = new Matrix(this.col, this.row);

    for (let c = 0; c < this.col; c++){
      for (let r = 0; r < this.row; r++){
        result[c][r] = func(this[c][r]);
      }
    }

    return result;
  }

  /**
   * @returns {Matrix} A deep copy of the matrix instance
   */
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

module.exports = Matrix;