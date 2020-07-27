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
    this.buttons = [];
    this.training = -1; // -1 when no class is being trained

    // Initiate deeplearn.js math and knn classifier objects
    this.bindPage();

    this.webcamDiv= document.createElement('div');
    document.body.appendChild(this.webcamDiv);
    
    // add the video element to the page
    // this function from the tmImage library returns a video element that
    // shows a video feed from the webcam
    this.webcam = new tmImage.Webcam(200, 200, true); //width, height, flipped    
    this.webcamSetup();
    
    //create div to contain the file input and webcam
    this.inputDiv = document.createElement('div');
    document.body.appendChild(this.inputDiv);

    //initialize the file input element 
    this.imgUpload =document.createElement('input');
    this.imgUpload.setAttribute("type", "file");
    this.imgUpload.setAttribute("accept", "image/*");
    this.imgUpload.multiple = true;

    this.imgList = [];

    this.imgUpload.addEventListener("change", this.upload.bind(this));
    
    //add file input to DOM
    this.inputDiv.appendChild(this.imgUpload);


    this.classSelection = [];

    for(let i = 0; i < NUM_CLASSES; i++){

      const radioID = "Class"+i;
      
      const radio = document.createElement('input');
      radio.setAttribute("type", "radio");
      radio.setAttribute("name", "class");
      radio.setAttribute("id", radioID);
      this.inputDiv.appendChild(radio);
      this.classSelection.push(radio);

      const radioLabel = document.createElement('label');
      radioLabel.setAttribute("for",radioID);
      radioLabel.innerText = radioID;
      this.inputDiv.appendChild(radioLabel);


    }

    this.trainULButton = document.createElement('button');
    this.trainULButton.innerText = "Train with Uploaded";

    this.uploadedImages = document.createElement('div');
    document.body.appendChild(this.uploadedImages);

    this.trainULButton.addEventListener('click', this.trainUL.bind(this));
    this.inputDiv.appendChild(this.trainULButton);
    //instatiate div for predict button + output text
    const predDiv = document.createElement('div');
    document.body.appendChild(predDiv);

    //instatiate predict button for webcam (disabled until model and  is running)
    this.predWC = document.createElement('button');
    this.predWC.innerText = "Predict Webcam";
    this.predWC.disabled = true;
    this.predWC.addEventListener("click", this.predict.bind(this));
    predDiv.appendChild(this.predWC);

    //prediction output    
    this.predText = document.createElement('span')
    this.predText.innerText = "No Prediction";
    predDiv.appendChild(this.predText);


    // Create training buttons and info texts    
    for (let i = 0; i < NUM_CLASSES; i++) {
      const div = document.createElement('div');
      document.body.appendChild(div);
      div.style.marginBottom = '10px';

      // Create training button
      const button = document.createElement('button')
      button.innerText = "Train " + i;
      button.disabled=true;
      div.appendChild(button);
      this.buttons.push(button);

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

  async trainUL(){

    var selectedClass = -1;
        for(var i = 0; i < this.classSelection.length; i++){
          if(this.classSelection[i].checked){
              selectedClass = i;
              break;
          }
        }

     for(var i = 0; i < this.imgList.length; i++){
        var img = this.imgList[i];
        let imgEl = document.createElement('img');
        imgEl.src = URL.createObjectURL(img);
        imgEl.setAttribute("height",200);
        imgEl.setAttribute("width",200);

        
        await this.train(selectedClass,imgEl);
       

        //remove items from the DOM
        

      }

     while(this.uploadedImages.firstChild){
          this.uploadedImages.removeChild(this.uploadedImages.firstChild);
        }

  }

  upload() {
     this.imgList = [];
      
      for(var i = 0; i < this.imgUpload.files.length; i++){
        this.imgList.push(this.imgUpload.files[i]);
        const imgEl = document.createElement('img');
        imgEl.src = URL.createObjectURL(this.imgUpload.files[i]);
        imgEl.setAttribute("height", 50);
        this.uploadedImages.appendChild(imgEl);

      }
    

  }



  async bindPage() {

    this.knn = knnClassifier.create();
    this.mobilenet = await mobilenet.load();
    this.buttons.forEach(button => button.disabled=false);


  }

  async webcamSetup(){

    await this.webcam.setup(); // request access to the webcam
    this.webcamDiv.appendChild(this.webcam.canvas);
    this.webcam.play();
    requestAnimationFrame(this.loop.bind(this));

    
  }


  async loop(){
    // update the webcam frame
    this.webcam.update();
    // Get image data from video element
    if (this.training != -1) {

      const image = this.webcam.canvas;
      await this.train(this.training, image);
      

    }

    const numClasses = this.knn.getNumClasses();

    if(numClasses > 0){
      this.predWC.disabled=false;
    }

    // then call loop again
    requestAnimationFrame(this.loop.bind(this));
  }

  async train(i, image){

    console.log(image);
    image = tf.browser.fromPixels(image);
    console.log(image);
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
      
    //dispose of the image and logits to free memory
    image.dispose()
      if (logits != null) {
        logits.dispose();
      }

  }

  async predict(){

    const image = tf.browser.fromPixels(this.webcam.canvas);

    let logits;
    // 'conv_preds' is the logits activation of MobileNet.
    const infer = () => this.mobilenet.infer(image, 'conv_preds');


    // If classes have been added run predict
    logits = infer();
    const res = await this.knn.predictClass(logits, TOPK);

    for (let i = 0; i < NUM_CLASSES; i++) {

      if (res.classIndex == i){
        this.predText.innerText = "Class" + i;
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