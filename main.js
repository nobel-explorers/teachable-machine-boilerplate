// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import "@babel/polyfill";
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import * as tmImage from '@teachablemachine/image';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

// Number of classes to classify
const NUM_CLASSES = 3;
// Webcam Image size. Must be 227. 
const IMAGE_SIZE = 227;
// K value for KNN
const TOPK = 10;


class Main {
  constructor() {
    // Initiate variables
    this.infoTexts = [];
    this.training = -1; // -1 when no class is being trained

    // Initiate deeplearn.js math and knn classifier objects
    this.bindPage();

    // this function from the tmImage library returns a video element that
    // shows a video feed from the webcam
    this.webcam = new tmImage.Webcam(200, 200, true); //width, height, flipped

    this.webcamSetup();

    

    const input = document.createElement('div');
    document.body.appendChild(input);
    
    // add the video element to the page
   
    

    this.imgUpload =document.createElement('input');
    this.imgUpload.setAttribute("type", "file");
    this.imgUpload.setAttribute("accept", "image/*");

    //add file input to DOM
    input.appendChild(this.imgUpload);

    const predict = document.createElement('button');
    predict.innerText = "Predict";
    document.body.appendChild(predict);
    this.predText = document.createElement('span')
    predText.innerText = " No Prediction";
    div.appendChild(predText);


    // Create training buttons and info texts    
    for (let i = 0; i < NUM_CLASSES; i++) {
      const div = document.createElement('div');
      document.body.appendChild(div);
      div.style.marginBottom = '10px';

      // Create training button
      const button = document.createElement('button')
      button.innerText = "Train " + i;
      div.appendChild(button);

      // Listen for mouse events when clicking the button
      button.addEventListener('mousedown', () => this.training = i);
      button.addEventListener('mouseup', () => this.training = -1);

      // Create info text
      const infoText = document.createElement('span')
      infoText.innerText = " No examples added";
      div.appendChild(infoText);
      this.infoTexts.push(infoText);
    }


   

  }

  async bindPage() {
    this.knn = knnClassifier.create();
    this.mobilenet = await mobilenet.load();

  }

  async webcamSetup(){

    await this.webcam.setup(); // request access to the webcam
    document.body.appendChild(this.webcam.canvas);
    this.webcam.play();
    requestAnimationFrame(this.loop.bind(this));
    
  }


  async loop(){
    // update the webcam frame
    this.webcam.update();
    // Get image data from video element
    if (this.training != -1) {
      
      await this.train(this.training);

    }
    // then call loop again
    requestAnimationFrame(this.loop.bind(this));
  }

  async train(i){
    const image = tf.fromPixels(this.webcam.canvas);

    let logits;
    // 'conv_preds' is the logits activation of MobileNet.
    const infer = () => this.mobilenet.infer(image, 'conv_preds');

    // Train class if one of the buttons is held down
    logits = infer();

    // Add current image to classifier
    this.knn.addExample(logits, i);

  

    // The number of examples for each class
    const exampleCount = this.knn.getClassExampleCount();

    // Update info text
    if (exampleCount[i] > 0) {
      this.infoTexts[i].innerText = ` ${exampleCount[i]} examples `
    }
      
    
      image.dispose();
      if (logits != null) {
        logits.dispose();
      }

  }


  async animate() {
      // Get image data from video element
      const image = tf.fromPixels(webcam.canvas);

      let logits;
      // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenet.infer(image, 'conv_preds');

      // Train class if one of the buttons is held down
      if (this.training != -1) {
        logits = infer();

        // Add current image to classifier
        this.knn.addExample(logits, this.training)
      }

      const numClasses = this.knn.getNumClasses();
      if (numClasses > 0) {

        // If classes have been added run predict
        logits = infer();
        const res = await this.knn.predictClass(logits, TOPK);

        for (let i = 0; i < NUM_CLASSES; i++) {

          // The number of examples for each class
          const exampleCount = this.knn.getClassExampleCount();

          // Make the predicted class bold
          if (res.classIndex == i) {
            this.infoTexts[i].style.fontWeight = 'bold';
          } else {
            this.infoTexts[i].style.fontWeight = 'normal';
          }

          // Update info text
          if (exampleCount[i] > 0) {
            this.infoTexts[i].innerText = ` ${exampleCount[i]} examples - ${res.confidences[i] * 100}%`
          }
        }
      }

      // Dispose image when done
      image.dispose();
      if (logits != null) {
        logits.dispose();
      }
    
  }
}

window.addEventListener('load', () => new Main());