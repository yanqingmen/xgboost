/*
 Copyright (c) 2014 by Contributors 

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
    
 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */
package org.dmlc.xgboost4j.demo;

import org.dmlc.xgboost4j.Booster;
import org.dmlc.xgboost4j.DMatrix;
import org.dmlc.xgboost4j.util.Params;
import org.dmlc.xgboost4j.util.Trainer;

/**
 * example for training binary classifier
 * @author hzx
 */
public class TrainBinaryClassifierExample {
    public static void main(String[] args) {
    //load train mat (svmlight format)
        DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
        //load valid mat (svmlight format)
        DMatrix validMat = new DMatrix("../../demo/data/agaricus.txt.test");
        
        //set params
        //set params
        Params param = new Params() {
            {
                put("eta", "1.0");
                put("max_depth", "3");
                put("silent", "1");
                put("nthread", "6");
                put("objective", "binary:logistic");
                put("gamma", "1.0");
                put("eval_metric", "error");
            }
        };
        
        //set round
        int round = 2;
        
        //set evaluation data
        DMatrix[] dmats = new DMatrix[] {trainMat, validMat};
        String[] evalNames = new String[] {"train", "valid"};
        
        //train a booster
        System.out.println("begin to train the booster model");        
        Booster booster = Trainer.train(param, trainMat, round, dmats, evalNames);
        
        //save model to modelPath
        String modelPath = "./data/xgb_model.bin";
        booster.saveModel(modelPath);
        
        System.out.println("training complete!!!!!!!!!!!!");
    }
}
