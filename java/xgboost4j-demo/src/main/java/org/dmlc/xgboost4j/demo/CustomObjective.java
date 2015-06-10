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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.dmlc.xgboost4j.Booster;
import org.dmlc.xgboost4j.IEvaluation;
import org.dmlc.xgboost4j.DMatrix;
import org.dmlc.xgboost4j.IObjective;
import org.dmlc.xgboost4j.util.Params;
import org.dmlc.xgboost4j.util.Trainer;

/**
 * an example user define objective and eval
 * NOTE: when you do customized loss function, the default prediction value is margin
 * this may make buildin evalution metric not function properly
 * for example, we are doing logistic loss, the prediction is score before logistic transformation
 * he buildin evaluation error assumes input is after logistic transformation
 * Take this in mind when you use the customization, and maybe you need write customized evaluation function
 * @author hzx
 */
public class CustomObjective {
    /**
     * loglikelihoode loss obj function
     */
    public static class LogRegObj implements IObjective {
        /**
         * simple sigmoid func
         * @param input
         * @return 
         * Note: this func is not concern about numerical stability, only used as example
         */
        public float sigmoid(float input) {
            float val = (float) (1/(1+Math.exp(-input)));
            return val;
        }
        
        public float[][] transform(float[][] predicts) {
            int nrow = predicts.length;
            float[][] transPredicts = new float[nrow][1];
            
            for(int i=0; i<nrow; i++) {
                transPredicts[i][0] = sigmoid(predicts[i][0]);
            }
            
            return transPredicts;
        }

        @Override
        public List<float[]> getGradient(float[][] predicts, DMatrix dtrain) {
            int nrow = predicts.length;
            List<float[]> gradients = new ArrayList<>();
            float[] labels = dtrain.getLabel();
            float[] grad = new float[nrow];
            float[] hess = new float[nrow];
            
            float[][] transPredicts = transform(predicts);
            
            for(int i=0; i<nrow; i++) {
                float predict = transPredicts[i][0];
                grad[i] = predict - labels[i];
                hess[i] = predict * (1 - predict);
            }
            
            gradients.add(grad);
            gradients.add(hess);
            return gradients;
        }        
    }
    
    /**
     * user defined eval function.
     * NOTE: when you do customized loss function, the default prediction value is margin
     * this may make buildin evalution metric not function properly
     * for example, we are doing logistic loss, the prediction is score before logistic transformation
     * the buildin evaluation error assumes input is after logistic transformation
     * Take this in mind when you use the customization, and maybe you need write customized evaluation function
     */
    public static class EvalError implements IEvaluation {
        String evalMetric = "custom_error";
        
        public EvalError() {
        }
        
        @Override
        public String getMetric() {
            return evalMetric;
        }

        @Override
        public float eval(float[][] predicts, DMatrix dmat) {
            float error = 0f;
            float[] labels = dmat.getLabel();
            int nrow = predicts.length;
            for(int i=0; i<nrow; i++) {
                if(labels[i]==0f && predicts[i][0]>0) {
                    error++;
                }
                else if(labels[i]==1f && predicts[i][0]<=0) {
                    error++;
                }
            }
            
            return error/labels.length;
        }
    }
    
    public static void main(String[] args) {
        //load train mat (svmlight format)
        DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
        //load valid mat (svmlight format)
        DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");
        
        //set params
        //set params
        Params param = new Params() {
            {
                put("eta", "1.0");
                put("max_depth", "2");
                put("silent", "1");
            }
        };
        
        //set round
        int round = 2;
        
        //set evaluation data
        DMatrix[] dmats = new DMatrix[] {trainMat, testMat};
        String[] evalNames = new String[] {"train", "eval"};
        
        //user define obj and eval
        IObjective obj = new LogRegObj();
        IEvaluation eval = new EvalError();
        
        //train a booster
        System.out.println("begin to train the booster model");        
        Booster booster = Trainer.train(param, trainMat, round, dmats, evalNames, obj, eval);
    }
}
